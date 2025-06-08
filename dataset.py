from __future__ import annotations
import os
import numpy as np
from tqdm import tqdm
from typing import (
    Optional,
    Dict,
    List,
    Tuple
)

from args import DatasetArgs, get_args
from utils.logging_utils import setup_logger, get_logger
from utils.dataset_utils import make_dataset, preprocess_in_chunks


class TrainValTestSplit:

    def __init__(self, train_dataset, val_dataset, test_dataset) -> None:
        self.train: TinyRadarDataset = train_dataset
        self.val: TinyRadarDataset = val_dataset
        self.test: TinyRadarDataset = test_dataset

    def __repr__(self): return str(self)

    def __str__(self) -> str:
        string = f"'train':\n{self.train}\n'val':\n{self.val}\n'test':\n{self.test}"
        return string
    
    @classmethod
    def loaded_from(cls, directory_path):
        train = TinyRadarDataset.loaded_from(os.path.join(directory_path, 'train'))
        val = TinyRadarDataset.loaded_from(os.path.join(directory_path, 'val'))
        test = TinyRadarDataset.loaded_from(os.path.join(directory_path, 'test'))
        return cls(train, val, test)
    
    def unpack(self) -> Tuple[TinyRadarDataset, TinyRadarDataset, TinyRadarDataset]:
        return self.train, self.val, self.test


class TinyRadarDataset:

    def __init__(
            self,
            data: np.ndarray, txt_labels: np.ndarray,
            people: np.ndarray, _indices: Optional[np.ndarray]=None,
            label_to_int: Optional[dict] = None,
            clip_value = None, max_value = None
        ):

        self._data: np.ndarray = data
        self._txt_labels: np.ndarray = txt_labels
        self._people = people
        self._indices: Optional[np.ndarray] = _indices

        if label_to_int is None:
            all_labels = sorted(set(self._txt_labels))
            label_to_int = {label: i for i, label in enumerate(all_labels)}

        self.label_to_int: Dict[str, int] = label_to_int
        self.int_to_label: Dict[int, str] = {val: key for key, val in self.label_to_int.items()}
        self._labels: np.ndarray = np.array([self.label_to_int[label] for label in self._txt_labels])

        self.clip_value: Optional[np.float32] = clip_value
        self.max_value: Optional[np.float32] = max_value

    def save(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        data_path = os.path.join(dir_path, "preprocessed_inputs.npy")
        people_path = os.path.join(dir_path, "persons.npy")
        labels_path = os.path.join(dir_path, "txt_labels.npy")
        
        np.save(data_path, self._data)
        np.save(people_path, self._people)
        np.save(labels_path, self._txt_labels)
        self.save_meta(dir_path)

    def save_meta(self, dir_path: str):
        meta_data_path = os.path.join(dir_path, "meta.npz")
        meta_data = {
            "label_to_int": self.label_to_int,
            "clip_value": self.clip_value,
            "max_value": self.max_value
        }
        np.savez(meta_data_path, **meta_data) # type: ignore

    @classmethod
    def loaded_from(cls, dir_path):
        data_path = os.path.join(dir_path, "preprocessed_inputs.npy")
        labels_path = os.path.join(dir_path, "txt_labels.npy")
        meta_data_path = os.path.join(dir_path, "meta.npz")
        people_path = os.path.join(dir_path, 'persons.npy')

        inputs = np.load(data_path, mmap_mode='r')
        txt_labels = np.load(labels_path)
        people = np.load(people_path)
        meta_data = np.load(meta_data_path, allow_pickle=True)
        label_to_int = meta_data['label_to_int'].item()
        clip_value = meta_data['clip_value'].item()
        max_value = meta_data['max_value'].item()

        dataset = TinyRadarDataset(inputs, txt_labels, people, None, label_to_int, clip_value, max_value)
        return dataset
    
    def shuffle(self, seed=0):
        """
        Creates a new datase from this dataset, with shuffled data and labels.
        The shuffle is completely deterministic based on the provided seed.
        """
        rng = np.random.RandomState(seed)
        indices = np.arange(len(self))
        rng.shuffle(indices)
        self._indices = indices

    def train_val_test_split(self, args: DatasetArgs) -> Tuple[TinyRadarDataset, TinyRadarDataset, TinyRadarDataset]:
        """This should not be a method. Ewww. Fuck indices. Eww. Pfew. Why did I introduce indices?????????"""
        assert 0 <= args.train_split <= 1 and 0 <= args.val_split <= 1, "Percentages must be between 0 and 1"

        if args.test_split is None:
            args.test_split = 1.0 - args.train_split - args.val_split
        assert 0 <= args.test_split <= 1, "Invalid test percentage"
        assert args.train_split + args.val_split + args.test_split <= 1.0 + 1e6, "Percentages must sum to <= 1.0"
        
        num_samples = len(self._labels)
        indices = np.arange(num_samples)
        if args.shuffle:
            rng = np.random.RandomState(args.dataset_shuffle_seed)
            rng.shuffle(indices)

        train_end = int(args.train_split * num_samples)
        val_end = train_end + int(args.val_split * num_samples)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        to_dataset = lambda idx: TinyRadarDataset(
            self._data, self._txt_labels, self._people, idx, self.label_to_int,
            self.clip_value, self.max_value
        )

        return to_dataset(train_idx), to_dataset(val_idx), to_dataset(test_idx)

    def make_real(self):
        """
        Shuffle and train_val_test split work by only changin the indexing, and
        don't actually save a new array. This method does that and returns a new dataset.
        """
        if self._indices is not None:
            data = self._data[self._indices]
            txt_labels = self._txt_labels[self._indices]
            people = self._people[self._indices]
            return TinyRadarDataset(data, txt_labels, people, None, self.label_to_int, self.clip_value, self.max_value)
        return TinyRadarDataset(self._data, self._txt_labels, self._people, None, self.label_to_int, self.clip_value, self.max_value)

    def __len__(self):
        if self._indices is not None:
            return len(self._indices)
        else:
            return len(self._data)
    
    def __getitem__(self, idx):
        if self._indices is not None:
            real_indices = self._indices[idx]
            return self._data[real_indices], self._labels[real_indices]
        else:
            return self._data[idx], self._labels[idx]
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx:idx+1]

    def iterate(self, batch_size):
        for idx in range(0, len(self), batch_size):
            yield self[idx:idx+batch_size]
    
    def __repr__(self) -> str:
        return str(self)
    
    def __str__(self) -> str:
        return f"TinyRadarDataset:\n\tNum Samples: {len(self)}\n\tLabels: {set(self.label_to_int)}"
    
    def get_inputs(self):
        if self._indices is None:
            return self._data
        else:
            return self._data[self._indices]
        
    def get_txt_labels(self):
        if self._indices is None:
            return self._txt_labels
        else:
            return self._txt_labels[self._indices]
        
    def get_labels(self):
        if self._indices is None:
            return self._labels
        else:
            return self._labels[self._indices]
        
    def get_people(self):
        if self._indices is None:
            return self._people
        else:
            return self._people[self._indices] 


def train_val_test_split(dataset_args: DatasetArgs):
    """
    What a disgusting function.
    """
    dataset = TinyRadarDataset.loaded_from(dataset_args.built_dataset_path)
    if dataset_args.split_based_on_person:
        test_indices = np.where(dataset._people == dataset_args.test_person)[0]
        val_indices = np.where(dataset._people == dataset_args.val_person)[0]
        train_indices = [i for i in np.arange(len(dataset)) if i not in test_indices and i not in val_indices]
        make_dataset = lambda indices: TinyRadarDataset(
            dataset._data, dataset._txt_labels, dataset._people, indices, dataset.label_to_int,
            dataset.clip_value, dataset.max_value
        )
        train, val, test = make_dataset(train_indices), make_dataset(val_indices), make_dataset(test_indices)
        if dataset_args.shuffle:
            _ = train.shuffle(), val.shuffle(), test.shuffle()
    else:
        train, val, test = dataset.train_val_test_split(dataset_args)

    train, val, test = train.make_real(), val.make_real(), test.make_real()
    train.save(os.path.join(dataset_args.built_dataset_path, 'train'))
    val.save(os.path.join(dataset_args.built_dataset_path, 'val'))
    test.save(os.path.join(dataset_args.built_dataset_path, 'test'))
    return TrainValTestSplit(train, val, test)


def get_dataset(dataset_args: DatasetArgs, rebuild=False, preprocess_again=False, train_val_test_split_again=False) -> TrainValTestSplit:
    if rebuild:
        make_dataset(dataset_args) # I don't like that window logic is already in there. Should be done in preprocessing

    if rebuild or preprocess_again:
        preprocess_in_chunks(dataset_args, chunk_size=100)

    if rebuild or preprocess_again or train_val_test_split_again:
        return train_val_test_split(dataset_args) # I don't like the way this is implemented
    
    return TrainValTestSplit.loaded_from(dataset_args.built_dataset_path)


if __name__ == '__main__':
    args = get_args()
    logger = setup_logger(args.logging_args)
    dataset = get_dataset(args.dataset_args, rebuild=False, preprocess_again=False, train_val_test_split_again=True)
    pass