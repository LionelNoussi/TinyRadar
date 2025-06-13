import numpy as np
from dataset import get_dataset, TinyRadarDataset
from utils.dataset_utils import preprocess_in_chunks
from args import get_args

dataset = TinyRadarDataset.loaded_from('datasets/dataset_11G')
test = TinyRadarDataset.loaded_from('datasets/dataset_11G/test')

args = get_args("DEFAULT")
data_split = get_dataset(args.dataset_args)
train = data_split.test
frames = np.load('datasets/dataset_5G/test/frames.npy')

# PREPROCESSING
frames = frames.reshape(998, 5, 32, 492, 2)
frames = frames.reshape(frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3]//6, 6, frames.shape[4]).mean(axis=4)

# Downsample time steps
frames = frames.reshape(frames.shape[0], frames.shape[1], frames.shape[2] // 2, 2, frames.shape[3], frames.shape[4]).mean(axis=3)

# Apply FFT over axes 2
frames = np.fft.fftshift(np.fft.fft(frames, axis=2), axes=2)
# chunk_fft = chunk
frames = np.abs(frames).astype(np.float32).reshape(-1, 16, 82, 10)

z = np.log1p(train.max_value) # type: ignore
max_value = 1 / (np.real(z)/(np.abs(z)**2)) # type: ignore
max_value = np.expm1(max_value)
train.max_value = max_value

frames = np.log1p(frames) / np.log1p(train.max_value) # type: ignore
frames = frames.astype(np.float32)

assert np.all(np.isclose(frames, train._data))

train.save_meta('datasets/dataset_5G/test/')
data_split.val.max_value = train.max_value
data_split.val.save_meta('datasets/dataset_5G/val/')
data_split.test.max_value = train.max_value
data_split.test.save_meta('datasets/dataset_5G/train/')
pass