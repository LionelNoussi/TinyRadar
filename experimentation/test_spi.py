import time, os, sys
sys.path.insert(0, os.getcwd())
import numpy as np

from utils.CommandArgs import CommandLineArgs
from utils.uart_utils import MicrocontrollerInterface
from args import get_args
from dataset import get_dataset
from utils.general import cprint
from model import QuantizedModel


class CmdArgs(CommandLineArgs):
    pass


def compare_arrays(mcu_array, python_array):
    # uint8_python_array = np.frombuffer(python_array, dtype=np.uint8)
    # restored = np.frombuffer(mcu_array.astype(np.uint8), dtype=np.complex64).reshape(python_array.shape)
    restored = np.frombuffer(mcu_array.astype(np.int8), dtype=np.int8).reshape(python_array.shape)
    try:
        if np.all(np.isclose(python_array, restored)):
            return True
        # if np.max(np.abs(mcu_array - uint8_python_array)) <= 1:
        #     return True
        else:
            return False
    except:
        return False
    

def match_next_step(uint8_mcu_array, current_array, next_step_array):
    restored = np.frombuffer(uint8_mcu_array.astype(np.uint8), dtype=np.complex64)
    assert np.all(np.isclose(restored.reshape(current_array.shape), current_array)), "Failed to restore array"

    temp = np.zeros(2624, dtype=np.complex64)
    NUM_SENSORS = 2
    DOWNSAMPLED_POINTS = 82
    DOWNSAMPLED_TIME_STEPS = 16
    DOWNSAMPLE_FACTOR_T = 2
    for sensor in range(NUM_SENSORS):
        for _range in range(DOWNSAMPLED_POINTS):
            for t in range(DOWNSAMPLED_TIME_STEPS):
                t1 = sensor + _range * NUM_SENSORS + t * (DOWNSAMPLED_POINTS * NUM_SENSORS * DOWNSAMPLE_FACTOR_T)
                t2 = t1 + (DOWNSAMPLED_POINTS * NUM_SENSORS)
                val1 = restored[t1];
                val2 = restored[t2];

                real = (val1.real + val2.real) * 0.5;
                imag = (val1.imag + val2.imag) * 0.5;

                idx = sensor + _range * NUM_SENSORS + t * (DOWNSAMPLED_POINTS * NUM_SENSORS)
                temp[idx] = real + 1j * imag
    assert np.all(np.isclose(temp.reshape(16, 82, 2), next_step_array)), "Didn't manage to reproduce downsampling."
    print("Restoring")


def main(cmd_args: CmdArgs):

    args = get_args(cmd_args.config)
    dataset = get_dataset(args.dataset_args).test
    q_model = QuantizedModel(model_path=args.model_args.tflite_model_path)
    
    frames_path = os.path.join(args.dataset_args.built_dataset_path, 'test/frames.npy')
    frames = np.load(frames_path)
    
    interface = MicrocontrollerInterface(args.serial_args, verbose=True)

    sample_index = 40
    test_frame = frames[sample_index]
    labels = dataset._labels
    label = labels[sample_index]

    windowed_frame = test_frame.reshape(1, 5, 32, 492, 2)

    range_downsampled_frame = windowed_frame.reshape(1, 5, 32, 82, 6, 2).mean(axis=4)

    time_downsampled_frame = range_downsampled_frame.reshape(1, 5, 16, 2, 82, 2).mean(axis=3)

    fft_frame = np.fft.fft(time_downsampled_frame, axis=2)

    abs_frame = np.abs(fft_frame).astype(np.float32)

    norm_frame = np.log1p(abs_frame) / np.log1p(dataset.max_value) # type: ignore

    q_frame_temp = norm_frame / q_model.in_scale + q_model.in_zero_point
    q_frame = np.clip(np.round(q_frame_temp), -128, 127).astype(np.int8)

    shift_frame = np.fft.fftshift(q_frame, axes=2)

    final_frame = shift_frame.reshape(-1, 16, 82, 10).astype(np.float32)

    raw_frame, _ = dataset[sample_index:sample_index+1]
    actual_frame = q_model.quantize_inputs(raw_frame)

    if np.all(np.isclose(final_frame, actual_frame)):
        # cprint("Frames are equal! Continuing.", 'green')
        pass
    else:
        cprint("Frames are not exact in python! Aborting.", 'red')
        exit()
    
    freq = args.serial_args.mcu_clock_frequency
    while True:
        try:
            sample_idx = int(input(f"Receive output for sample index (0, {len(labels)}): "))
            # process_cycles = interface.read_uint32_integer()
            frame_cycles = interface.read_uint32_integer()
            window_cycles = interface.read_uint32_integer()
            comp_cycles = interface.read_uint32_integer()
            mcu_output_class = interface.read_uint8_integer()
            # print(f"downsampling one frame took {process_cycles} many cycles --> {process_cycles / args.serial_args.mcu_clock_frequency:.4f} seconds")
            print(f"Downsampling range points takes {frame_cycles} many cycles --> {frame_cycles / freq:.4f} seconds")
            print(f"Processing one window takes {window_cycles} many cycles --> {window_cycles / freq:.4f} seconds")
            print(f"Computation took {comp_cycles} many cycles --> {comp_cycles / freq:.4f} seconds")
            label = labels[sample_idx]
            sample = dataset._data[sample_idx:sample_idx+1]
            python_output_class = np.argmax(q_model.predict(sample), axis=-1)[0]
            print(f"Python Output: {python_output_class}. MCU Output: {mcu_output_class}. Label: {label}")
        except (TimeoutError, KeyboardInterrupt):
            print("Timed out...")

    # while True:
    #     input("Press `Enter` to read array.")
    #     try:
    #         array = interface.read_int_array(signed=False)
    #     except (KeyboardInterrupt, TimeoutError):
    #         pass
    #     else:
    #         if compare_arrays(array, final_frame[0]):
    #             cprint("MCU array matches python array! Test passed.", 'green')
    #             # match_next_step(array, range_downsampled_frame[0, 0], time_downsampled_frame[0, 0])
    #         else:
    #             cprint("MCU Array did not match python array... Test failed.", 'red')

    input("Debug")




if __name__ == '__main__':
    cmd_args = CmdArgs()
    main(cmd_args)