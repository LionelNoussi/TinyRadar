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
    if QUANTIZED:
        logger.info("Using quantized model.")
        q_model = QuantizedModel(model_path=args.model_args.tflite_model_path)

    model.summary(print_fn=lambda x: logger.debug(x))
    
    input("Press Enter to Start Experiment")

    interface = MicrocontrollerInterface(args.serial_args)
    interface.drain()
    interface.start()

    total_cycles = 0
    num_python_correct = 0
    num_mcu_currect = 0
    num_python_is_mcu = 0
    num_samples = 0
    total_samples = len(dataset)
    if not QUANTIZED:
        golden_output = np.argmax(model.predict(dataset.get_inputs()), axis=-1)

    for idx in tqdm(range(total_samples), disable=VERBOSE):
        inputs, label = dataset[idx]

        if QUANTIZED:
            python_output = np.argmax(q_model.predict(inputs.reshape(1, *inputs.shape)))
            inputs = q_model.quantize_inputs(inputs)
        else:
            python_output = golden_output[idx]

        logger.debug("Sending input data.")
        interface.send_array(inputs)

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
        logger.debug(f"Sample {idx} / {total_samples}: Running MCU Accuracy: {num_mcu_currect / num_samples*100:.2f}")
        logger.debug(f"Sample {idx} / {total_samples}: MCU matches python: {python_output == mcu_output}")
        logger.debug(f"Total Cycles: {total_cycles} --> {total_cycles / args.serial_args.mcu_clock_frequency:.4f} seconds")

    logger.info(f"Python Accuracy: {num_python_correct/num_samples*100:.2f}")
    logger.info(f"MCU accuracy: {num_mcu_currect/num_samples*100:.2f}")
    logger.info(f"MCU matches python accuracy: {num_python_is_mcu/num_samples*100:.2f}")


if __name__ == '__main__':
    cmd_args = CmdLineArgs()
    main(cmd_args)