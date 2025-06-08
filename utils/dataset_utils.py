import os
import numpy as np
from args import DatasetArgs
from utils.logging_utils import get_logger
from typing import (
    Optional,
    List
)
from tqdm import tqdm


def extract_instance(args: DatasetArgs, session_path: str, instance: int) -> List[np.ndarray]:
    """
    This function takes the args, a path to session folder and an instance within that session,
    and returns the formatted data for that instance.

    This means doing the following things:
        1. Reading the info for that session and checking whether it is a valid
           session, which can be used.
        2. Loading the binary data for both sensors and combining them (data_binary_stacked)
        3. Splitting up the data into args.num_windows of equal args.sweeps_per_window with a certain stride

    At the end, the function returns a list of numpy arrays with the following shape:
    (num_windows, sweeps_per_window, range_points_per_sweep, 2)
    ~ (1, windows, time, range distances, which sensor)

    1 For the batch dimension, to allow for easy concatenation
    Each window basically corresponds to one temporal input for the Encoder later.
    Each window has an equal amount of sweeps, which will first get passed to the 2D CNN
    Each sweep has range_points_per_sweep number of datapoints, for reflection strengths of different distances
    There are two sensors.

    The list contains multiple extractions per recording session. So if a recording took 3 seconds, and our data
    is 1 second long, we extract 3 instances. If the recording is 2 seconds long, we also extract 3, but they will have some overlap.
    The number of extractions can be set via args.number_of_extractions.

    Parameters
    ----------
    args : DatasetArgs
        The DatasetArgs to use.
    session_path : str
        The relative or absolute path to the session's raw data folder
    instance : int
        The specific instance within that session

    Returns
    -------
    List[np.ndarray]
        List of the formatted data arrays with the following shape: 
        (1, args.num_windows, args.sweeps_per_window, args.range_points_per_sweep, 2)
    """

    # Properly format the paths
    info_path = os.path.join(session_path, f"{instance}_info.txt")
    sensor1_data_path = os.path.join(session_path, f"{instance}_s0.dat")
    sensor2_data_path = os.path.join(session_path, f"{instance}_s1.dat")

    # Read in and check that the info is as expected
    with open(info_path, "r") as f:
        info = [x.strip() for x in f.readlines()]

    number_of_sweeps = int(info[1])
    sweep_frequency = int(info[2])
    sensor1_range_points = int(info[14])
    sensor2_range_points = int(info[15])
    assert sweep_frequency == args.sweep_frequency
    assert sensor1_range_points == args.range_points_per_sweep and sensor2_range_points == args.range_points_per_sweep
    if args.min_sweeps > number_of_sweeps:
        return []
    
    # Read in and stack the data of the two sensors
    sensor_data1 = np.fromfile(sensor1_data_path, dtype=np.complex64)
    sensor_data2 = np.fromfile(sensor2_data_path, dtype=np.complex64)
    sensor_data1 = sensor_data1.reshape((number_of_sweeps, sensor1_range_points))
    sensor_data2 = sensor_data2.reshape((number_of_sweeps, sensor2_range_points))
    sensor_data_stacked = np.stack((sensor_data1, sensor_data2), axis=-1)   # (number_of_sweeps, range_points, 2)

    # Else create args.num_windows many windows. The datapoints of the windows might overlap.
    window_start_indices = [i for i in range(0, number_of_sweeps - args.sweeps_per_window + 1, args.stride)]

    # If less than required windows, but enough number_of_sweeps (> args.min_sweeps)
    # max(1, ) creates an instance with some windows empty
    # NOTE: sooooo stupid, but they did it that way. I would have just discarded them. 
    # To discard them, set args.min_sweeps to args.sweeps_per_window + args.stride * (args.num_windows - 1)
    
    available_extractions = max(1, len(window_start_indices) - args.num_windows)

    # Here we extract the instances instances
    instances = []
    number_of_extractions = min(available_extractions, args.number_of_extractions)
    for i in range(0, number_of_extractions):
        selected_indices = window_start_indices[i:i+args.num_windows]

        # Fill the output array with selected windows
        data = np.zeros((1, args.num_windows, args.sweeps_per_window, args.range_points_per_sweep, 2), dtype=np.complex64)
        for wdx, idx in enumerate(selected_indices):
            data[0, wdx, :, :, :] = sensor_data_stacked[idx:idx + args.sweeps_per_window]
        instances.append(data)

    return instances


def make_dataset(dataset_args: DatasetArgs):
    """
    This function creates the dataset from the collection of individual binary
    data files. It goes through every instance, and properly formats it into
    windows. It then saves the output array and corresponding label array to
    .npy files.
    """
    inputs = []
    persons = []
    txt_labels = []

    people = [f"0_{i}" for i in range(1, dataset_args.num_single_user_sessions + 1)]
    people.extend([f"{i}" for i in range(1, dataset_args.num_people + 1)])

    for person in people:
        print(f"Person: {person}")
        for gesture in dataset_args.gestures:
            print(f"  Gesture: {gesture}")
            for session in range(dataset_args.num_sessions):
                print(f"    Session: {session}")
                print(f"      Instance: ", end='')
                session_path = os.path.join(dataset_args.raw_dataset_path, f"p{person}/{gesture}/sess_{session}")
                for instance in range(dataset_args.num_instances):
                    print(f" {instance}", end='')
                    data_points = extract_instance(dataset_args, session_path, instance)
                    if data_points is not None:
                        num_extractions = len(data_points)
                        inputs.extend(data_points)
                        txt_labels.extend([gesture] * num_extractions)
                        persons.extend([person] * num_extractions)
                print()
    
    inputs = np.concatenate(inputs)
    os.makedirs(dataset_args.built_dataset_path, exist_ok=True)
    data_path = os.path.join(dataset_args.built_dataset_path, "input_data.npy")
    labels_path = os.path.join(dataset_args.built_dataset_path, "txt_labels.npy")
    persons_path = os.path.join(dataset_args.built_dataset_path, "persons.npy")
    np.save(labels_path, np.array(txt_labels))
    np.save(persons_path, np.array(persons))
    np.save(data_path, inputs)


def reservoir_sample(reservoir, stream, sample_size, seen):
    for value in stream:
        seen[0] += 1
        if len(reservoir) < sample_size:
            reservoir.append(value)
        else:
            j = np.random.randint(0, seen[0])
            if j < sample_size:
                reservoir[j] = value


def preprocess_in_chunks(dataset_args: DatasetArgs, chunk_size=100):
    """
    This function does preprocessing on the input data, by computing the
    FFT on the input data, taking the magnitude and normalizing. Optionally, it can also be clipped.

    The function saves the resulting preprocessed dataset to .npy files again.
    """
    dir_path = dataset_args.built_dataset_path
    clip = dataset_args.clip
    tdr = dataset_args.time_downsample_rate
    rdr = dataset_args.range_downsample_rate

    # CREATE ALL THE PATHS
    input_data_path = os.path.join(dir_path, "input_data.npy")
    txt_labels_path = os.path.join(dir_path, "txt_labels.npy")
    memmap_path = os.path.join(dir_path, 'doppler.memmap')
    results_path = os.path.join(dir_path, 'preprocessed_inputs.npy')

    # LOAD THE INPUT DATA
    input_data = np.load(input_data_path, mmap_mode='r')
    print("Loaded input array.")
    
    # CREATE MEMMAP FILE AND INSTANCE OF FINAL RESULT
    total_instances = input_data.shape[0]
    new_shape = (input_data.shape[0], input_data.shape[2] // tdr, input_data.shape[3] // rdr, input_data.shape[4] * input_data.shape[1])
    preprocessed_inputs = np.memmap(memmap_path, mode='w+', dtype=np.float32, shape=new_shape)

    # INITIATE VARIABLES TO KEEP TRACK OF MAX ACROSS CHUNKS
    global_max = -np.inf
    global_percentile = None
    if clip:
        reservoir = []
        seen = [0]  # mutable counter
        sample_limit = 1000
    
    # GO THROUGH CHUNKS ONE TIME AND CALCULATE FFT FOR EACH CHUNK AND SAVE
    for start_idx in tqdm(range(0, total_instances, chunk_size)):

        # load chunk into RAM
        end_idx = min(start_idx + chunk_size, total_instances)
        chunk = input_data[start_idx:end_idx]

        # Downsample range points of chunk
        chunk = chunk.reshape(chunk.shape[0], chunk.shape[1], chunk.shape[2], chunk.shape[3]//rdr, rdr, chunk.shape[4]).mean(axis=4)

        # Downsample time steps
        chunk = chunk.reshape(chunk.shape[0], chunk.shape[1], chunk.shape[2] // tdr, tdr, chunk.shape[3], chunk.shape[4]).mean(axis=3)

        # Apply FFT over axes 2
        chunk_fft = np.fft.fftshift(np.fft.fft(chunk, axis=2), axes=2)
        # chunk_fft = chunk
        preprocessed_inputs[start_idx:end_idx] = np.abs(chunk_fft).astype(np.float32).reshape(-1, *new_shape[1:])
        preprocessed_inputs.flush()

        # Update global max
        chunk_max = np.abs(chunk_fft).max()
        if chunk_max > global_max:
            global_max = chunk_max

        # sample values for percentile later:
        if clip:
            flat_chunk = chunk_fft.ravel()
            reservoir_sample(reservoir, flat_chunk, sample_limit, seen)

    if clip:
        global_percentile = np.percentile(reservoir, dataset_args.clip_percentile)
        global_max = min(global_max, global_percentile)

    # NORMALIZE ALL OF THE CHUNKS WITH LOG TO [0, 1] RANGE
    for start_idx in tqdm(range(0, total_instances, chunk_size)):
        end_idx = min(start_idx + chunk_size, total_instances)
        chunk = preprocessed_inputs[start_idx:end_idx]  # load chunk into RAM

        if clip:
            chunk = np.clip(chunk, a_min=0, a_max=global_percentile)

        normalized_chunk = np.log1p(chunk) / np.log1p(global_max)
        preprocessed_inputs[start_idx:end_idx] = normalized_chunk

    # SAVE THE FINAL ARRAY TO .npy FILE
    np.save(results_path, preprocessed_inputs)

    # SAVE META DATA
    txt_labels = np.load(txt_labels_path)
    all_labels = sorted(set(txt_labels))
    label_to_int = {label: i for i, label in enumerate(all_labels)}

    meta_data_path = os.path.join(dir_path, "meta.npz")
    meta_data = {
        "label_to_int": label_to_int,
        "clip_value": global_percentile,
        "max_value": global_max
    }
    np.savez(meta_data_path, **meta_data)