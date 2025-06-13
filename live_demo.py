import serial, time
import random
import numpy as np
import argparse

from args import get_args
from utils.logging_utils import setup_logger
from model import get_model
from dataset import get_dataset
from utils.uart_utils import MicrocontrollerInterface
from model import QuantizedModel
from utils.CommandArgs import CommandLineArgs


class CmdArgs(CommandLineArgs):
    fp: bool = False
    full_output: bool = False
    dont_time: bool = False
    compare_to_float: bool = False


def main(cmd_args: CmdArgs):
    QUANTIZED = not cmd_args.fp
    FULL_OUTPUT = cmd_args.full_output
    TIME_COMPUTATION = not cmd_args.dont_time
    COMPARE_TO_FLOAT = cmd_args.compare_to_float

    # get yaml args, setup logger, get model and dataset
    args = get_args(cmd_args.config)
    logger = setup_logger(args.logging_args)
    dataset = get_dataset(args.dataset_args).test
    model = get_model(args.model_args)
    if QUANTIZED:
        q_model = QuantizedModel(model_path=args.model_args.tflite_model_path)

    # Get random sample input and get golden_output
    logger.info("Calculating golden output in python.")
    sample_index = random.randint(0, len(dataset))
    sample_input, sample_label = dataset[sample_index:sample_index+1]
    if QUANTIZED:
        q_sample_input = q_model.quantize_inputs(sample_input)
    
    if COMPARE_TO_FLOAT:
        python_output = model.predict(sample_input)
    else:
        q_python_output = q_model.q_predict(q_sample_input)
        python_output = q_model.dequantize_outputs(q_python_output)
    python_output_class = np.argmax(python_output[0])

    print(dataset)
    print(python_output_class)
    print(sample_label)
    input("Press Enter to Start Experiment")

    # Start serial communication with MCU and send start signal
    logger.info("Starting Microcontroller Interface")
    interface = MicrocontrollerInterface(args.serial_args, verbose=True)
    interface.readlines()

    # Send input array to MCU
    logger.info(f"Sending input data.")
    interface.send_array(q_sample_input[0])

    logger.info("Waiting For Output...")
    if TIME_COMPUTATION:
        cycles = interface.read_uint32_integer()
        logger.info(f"Computation took {cycles} many cycles --> {cycles / args.serial_args.mcu_clock_frequency:.4f} seconds")

    if FULL_OUTPUT:
        q_mcu_outputs = interface.read_int_array()
        mcu_outputs = q_model.dequantize_outputs(q_mcu_outputs)

    mcu_output_class = interface.read_uint8_integer()

    # Compore output with golden output
    if FULL_OUTPUT:
        formatter = {'float_kind': lambda x: f"{x:.4f}"}
        logger.info("Python Logits: %s", np.array2string(python_output[0], formatter=formatter, floatmode='fixed')) # type: ignore
        logger.info("MCU Logits: %s", np.array2string(mcu_outputs, formatter=formatter, floatmode='fixed')) # type: ignore
        logger.info("Python quantized Logits: %s", np.array2string(q_python_output[0], formatter=formatter, floatmode='fixed')) # type: ignore
        logger.info("MCU quantized Logits: %s", np.array2string(np.array(q_mcu_outputs), formatter=formatter, floatmode='fixed')) # type: ignore

    logger.info(f"Python Output: {python_output_class}. MCU Output: {mcu_output_class}. Label: {sample_label[0]}")


if __name__ == '__main__':
    cmd_args = CmdArgs()
    main(cmd_args)