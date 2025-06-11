import argparse
from typing import (
    Optional
)
from utils.HierarchyArgs import HierarchyArgs


class ModelArgs(HierarchyArgs):
    load_model: bool = True
    load_model_path: str = 'saved_models/saved_model_87.h5'
    tflite_model_path: str = 'build/my_model_11G_86p67.tflite'
    # tflite_model_path: str = 'build/my_model.tflite'
    save_model: bool = True
    save_model_path: str = 'saved_models/saved_model.h5'
    checkpoint_dir: str = 'saved_models/checkpoints/'
    learning_rate: float = 1e-2
    batch_size: int = 32
    num_epochs: int = 100


class ConversionArgs(HierarchyArgs):
    quantize: bool = True
    out_dir: str = "build/"


class DatasetArgs(HierarchyArgs):
    raw_dataset_path: str = '/Volumes/LioDrive/Datasets/TinyRadar11G/data'
    built_dataset_path: str = './datasets/dataset_11G/'
    num_sessions: int = 5
    num_instances: int = 7
    sweep_frequency: int = 160
    range_points_per_sweep: int = 492
    stride: int = 32
    num_windows: int = 5
    sweeps_per_window: int = 32
    number_of_extractions: int = 1
    time_downsample_rate: int = 2
    range_downsample_rate: int = 6
    min_sweeps: int = 32
    num_people: int = 25
    num_single_user_sessions: int = 1
    gestures: list = ["PinchIndex", "PinchPinky", "FingerSlider", "FingerRub",
                      "SlowSwipeRL", "FastSwipeRL", "Push", "Pull", "PalmTilt",
                      "Circle", "PalmHold"] #, "NoHand", "RandomGesture"]
    # gestures: list = ["PinchIndex", "Push", "SlowSwipeRL", "NoHand", "Circle"]
    shuffle: bool = True
    clip: bool = False
    clip_percentile: float = 99.0
    dataset_shuffle_seed: int = 0
    split_based_on_person: bool = False
    test_person: str = "1"
    val_person: str = "2"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1


class LoggingArgs(HierarchyArgs):
    level: str = "DEBUG"  # or int like logging.INFO
    log_to_file: bool = True
    log_dir_path: str = "logs/"
    file_mode: str = "a"  # or "w"
    log_to_stdout: bool = True
    format: str = '%(asctime)s | %(name)s | %(levelname)s: %(message)s'
    datefmt: str = "%H:%M:%S"


class SerialArgs(HierarchyArgs):
    serial_port: str = '/dev/tty.usbmodem103'
    baudrate: int = 115200
    mcu_clock_frequency: int = 80000000
    array_size_in_bytes: int = 13120 # 52480    # how many bytes the microcontroller expects to receive at once.
    ser_timeout: int = 1        # seconds
    read_timeout: int = 20    # seconds
    start_signal: bytes = b'\x02'
    stop_header: bytes = b'\x01'
    inference_header: bytes = b'\x02'
    array_header: bytes = b'\x01'
    uint8_header: bytes = b'\x02'
    uint32_header: bytes = b'\x03'
    ready_header: bytes = b'\x04'

class SPIArgs(HierarchyArgs):
    max_speed_hz: int = 10_000_000
    mode: int = 0
    rate_hz: int = 160
    frame_size: int = 492 * 2 * 8
    chunk_limit: int = 4096


class Args(HierarchyArgs):
    model_args: ModelArgs = ModelArgs()
    conversion_args: ConversionArgs = ConversionArgs()
    dataset_args: DatasetArgs = DatasetArgs()
    serial_args: SerialArgs = SerialArgs()
    spi_args: SPIArgs = SPIArgs()
    logging_args: LoggingArgs = LoggingArgs()

def raspberry_pi_config():
    args = Args()
    args.dataset_args.raw_dataset_path = "/media/lionel/LioDrive/Datasets/TinyRadar11G/data"
    args.dataset_args.built_dataset_path = "/media/lionel/LioDrive/Datasets/TinyRadar11G/dataset_11G"
    return args


def generate_template_config(out_yaml_file_path: str = 'configs/template.yaml'):
    print(f"Generating template config at {out_yaml_file_path}")
    default_args = Args()
    default_args.to_yaml(out_yaml_file_path)
    return default_args


def get_args(config_file_name=None) -> Args:
    if config_file_name is None or config_file_name == 'DEFAULT':
       return Args()
    if config_file_name == "raspy":
        return raspberry_pi_config()
    args = Args.from_yaml(config_file_name)
    return args


if __name__ == '__main__':
    generate_template_config('configs/template.yaml')