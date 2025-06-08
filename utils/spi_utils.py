import serial, struct, time
from args import SPIArgs
from typing import List
import numpy as np
import concurrent.futures


class SPIInterface:
    
    def __init__(self, spi_args: SPIArgs, verbose: bool = False) -> None:
        self.args = spi_args
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)  # Bus 0, CE0
        self.spi.max_speed_hz = spi_args.max_speed_hz
        self.spi.mode = spi_args.mode

    def send_array(self, array: np.ndarray):
        assert len(array.shape) == 3

        delay = 1.0 / self.args.rate_hz  # time between chunks
        num_frames = array.shape[0]

        for i in range(num_frames):
            chunk = list(array[i].tobytes())
            self.spi.xfer2(chunk)
            time.sleep(delay)