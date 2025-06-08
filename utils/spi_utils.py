import spidev, time
from args import SPIArgs
import numpy as np


class SPIInterface:
    
    def __init__(self, spi_args: SPIArgs, verbose: bool = False) -> None:
        self.args = spi_args
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)  # Bus 0, CE0
        self.spi.max_speed_hz = spi_args.max_speed_hz
        self.spi.mode = spi_args.mode

    def send(self, message: bytearray):
        self.spi.xfer2(message)

    def send_array(self, array: np.ndarray):
        assert array.ndim == 3
        assert array.shape[2] == 2
        assert array.dtype == np.complex64
        print(array.shape)
        
        num_frames = array.shape[0]
        frame_size = array.shape[1] * array.shape[2] * 8  # 492 × 2 × 8 = 7872
        max_chunk_size = 4096
        delay_per_frame = 1.0 / self.args.rate_hz
        
        for i in range(num_frames):
            frame_bytes = array[i].astype(np.complex64).tobytes()
            # print(f"Sending frame {i}")
            offset = 0
            time.sleep(0.01)
            while offset < frame_size:
                chunk = frame_bytes[offset : offset + max_chunk_size]
                self.spi.xfer2(list(chunk))
                offset += len(chunk)

    def read_byte(self):
        response = self.spi.xfer2([0x00])[0]
        return response