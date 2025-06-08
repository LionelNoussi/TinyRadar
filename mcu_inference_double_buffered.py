import serial, time
import random
import numpy as np
from tqdm import tqdm
import logging

from args import get_args
from utils.logging_utils import setup_logger
from model import get_model
from dataset import get_dataset
from utils.uart_utils import MicrocontrollerInterface
from utils.CommandArgs import CommandLineArgs
from model import QuantizedModel


class CmdLineArgs(CommandLineArgs):
    fp: bool = False
    dont_time: bool = False


def main(cmd_args: CmdLineArgs):
    # READ COMMAND LINE ARGUMENTS
    QUANTIZED = not cmd_args.fp
    TIME_COMPUTATION = not cmd_args.dont_time

    # get yaml args, setup logger, get model and dataset
    args = get_args(cmd_args.config)
    logger = setup_logger(args.logging_args)
    dataset = get_dataset(args.dataset_args).test

    VERBOSE = logger.level <= logging.DEBUG

    model = get_model(args.model_args)
    model.summary(print_fn=lambda x: logger.debug(x))
    if QUANTIZED:
        logger.info("Using quantized model.")
        q_model = QuantizedModel(model_path=args.model_args.tflite_model_path)
    
    interface = MicrocontrollerInterface(args.serial_args, verbose=VERBOSE)
    input("Press Enter to Start Experiment: ")
    interface.readlines()
    interface.start()

    total_cycles = 0
    num_python_correct = 0
    num_mcu_currect = 0
    num_python_is_mcu = 0
    num_samples = 0
    total_samples = len(dataset)
    golden_output = np.argmax(model.predict(dataset.get_inputs()), axis=-1)

    # Start sending the first input asynchronously
    first_inputs, _ = dataset[0]
    if QUANTIZED:
        first_inputs = q_model.quantize_inputs(first_inputs)
    send_handle = interface.send_array_async(first_inputs)

    for idx in tqdm(range(total_samples), disable=not VERBOSE):
        label = dataset._labels[idx]
        python_output = golden_output[idx]

        logger.debug("Waiting for input data to finish sending...")
        send_handle.wait()

        # Start sending the next input asynchronously if not last sample
        logger.debug("Finished sending. Starting to send next input asynchrously.")
        if idx < total_samples - 1:
            next_inputs = dataset._data[idx + 1]
            if QUANTIZED:
                next_inputs = q_model.quantize_inputs(next_inputs)
            send_handle = interface.send_array_async(next_inputs)
        else:
            interface.send_header(args.serial_args.stop_header)

        logger.debug("Waiting for output...")
        if TIME_COMPUTATION:
            cycles = interface.read_uint32_integer()
            total_cycles += cycles
        mcu_output = interface.read_uint8_integer()

        num_python_correct += python_output == label
        num_mcu_currect += mcu_output == label
        num_python_is_mcu += python_output == mcu_output
        num_samples += 1
        logger.debug(f"Sample {idx} / {total_samples}: Correct Output: {python_output} | MCU Output: {mcu_output} | Label: {label}")
        logger.debug(f"Sample {idx} / {total_samples}: Running MCU Accuracy: {num_mcu_currect / num_samples:.2f}")
        logger.debug(f"Total Cycles: {total_cycles} --> {total_cycles / args.serial_args.mcu_clock_frequency:.4f} seconds")

    logger.info(f"Python Accuracy: {num_python_correct/num_samples:.2f}")
    logger.info(f"MCU accuracy: {num_mcu_currect/num_samples:.2f}")
    logger.info(f"MCU matches python accuracy: {num_python_is_mcu/num_samples:.2f}")


if __name__ == '__main__':
    cmd_args = CmdLineArgs()
    main(cmd_args)