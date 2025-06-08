import serial, struct, time
from args import SerialArgs
from typing import List
import numpy as np
import concurrent.futures


class AsyncHandle:
    def __init__(self, future):
        self._future = future

    def wait(self):
        return self._future.result()
    

class MicrocontrollerInterface:
    
    def __init__(self, serial_args: SerialArgs, verbose: bool = False) -> None:
        self.args = serial_args
        self.ser = serial.Serial(self.args.serial_port, self.args.baudrate, timeout=self.args.ser_timeout)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.verbose = verbose

    def start(self):
        self.ser.write(self.args.start_signal)

    def drain(self):
        self.ser.read_all()

    def readlines(self):
        lines = self.ser.readlines()
        print("MCU Printf Outputs:")
        for line in lines:
            print('\t' + line.decode().rstrip('\r\n'))

    def send_header(self, header: bytes):
        assert len(header) == 1
        self.ser.write(header)

    def _wait_for_header(self, target_header: bytes) -> bytes:
        buffer = b''
        
        start_time = time.time()
        while time.time() - start_time < self.args.read_timeout:

            header = self.ser.read(1)

            if header == target_header:
                break  # Found the header, proceed
            elif len(header) != 0:
                # If something else was read, store it in the buffer
                buffer += header

        if self.verbose and len(buffer):     # if verbose, print the buffered bytes which were read.
            try:
                formatted_buffer = '\t' + '\n\t'.join(buffer.decode().split('\n'))
            except:
                formatted_buffer = buffer
            print(f"While waiting for header, also read this from UART:\n{formatted_buffer}".rstrip('\n\t'))

        if not len(header):
            raise TimeoutError("No header received.")
        
        return header

    def send_array(self, array: np.ndarray):
        if array.dtype == np.float32:
            byte_array = np.array(array).astype('<f4').tobytes()
        elif array.dtype == np.int8:
            byte_array = np.array(array).astype(np.int8).tobytes()
        else:
            raise TypeError("Unsupported dtype")

        self.ser.write(self.args.inference_header)
        time.sleep(0.1)
        self.ser.write(byte_array)
        self.ser.flush()

    def send_array_async(self, array: np.ndarray) -> AsyncHandle:
        def write_func():
            if array.dtype == np.float32:
                byte_array = np.array(array).astype('<f4').tobytes()
            elif array.dtype == np.int8:
                byte_array = np.array(array).astype(np.int8).tobytes()
            else:
                raise TypeError("Unsupported dtype")

            self.ser.write(self.args.inference_header)
            time.sleep(0.1)
            self.ser.write(byte_array)
            self.ser.flush()

        future = self.executor.submit(write_func)
        return AsyncHandle(future)

    def read_uint8_integer(self) -> int:
        self._wait_for_header(self.args.uint8_header)

        # The value we are looking for will be one byte after
        # the single_header
        output = self.ser.read(1)
        if len(output) != 1:    # If no byte was read, raise TimeoutError
            raise TimeoutError("No or incomplete data received from MCU.")

        # Create an integer from the read byte
        return int.from_bytes(output, byteorder='little', signed=False)
    
    def read_uint32_integer(self) -> int:
        header = self._wait_for_header(self.args.uint32_header)

        # The value we are looking for will be one byte after
        # the single_header
        output = self.ser.read(4)
        if not len(output):    # If no byte was read, raise TimeoutError
            raise TimeoutError("No or incomplete data received from MCU.")

        # Create an integer from the read byte
        return int.from_bytes(output, byteorder='little', signed=False)
    
    def read_int8_array(self):
        self._wait_for_header(self.args.array_header)

        # When receiving an array, the MCU first sends the number of elements
        num_elements = self.ser.read(1)
        if len(num_elements) != 1:
            raise TimeoutError("No or incomplete data received from MCU.")
        
        num_elements = int.from_bytes(num_elements, byteorder='little', signed=False)
        
        outputs = []
        for i in range(num_elements):
            element = self.ser.read(1)
            integer = int.from_bytes(element, byteorder='little', signed=True)
            outputs.append(integer)

        return outputs