import keras
import numpy as np
from tqdm import tqdm
from typing import (
    Union
)

from utils.CommandArgs import CommandLineArgs
from args import get_args
from utils.logging_utils import setup_logger
from model import get_model
from dataset import get_dataset
from model import QuantizedModel


class CmdArgs(CommandLineArgs):
    fp: bool = False
    qh5: bool = False
    _qh5_help: str = "If True, uses a a normal h5 model, but with tfmot quantization layers, used for quantized aware training"
    val: bool = False


def main(cmd_args: CmdArgs):
    QUANTIZED = (not cmd_args.fp) and (not cmd_args.qh5)
    VAL = cmd_args.val

    args = get_args(cmd_args.config)
    logger = setup_logger(args.logging_args)
    dataset = get_dataset(args.dataset_args)
    dataset = dataset.val if VAL else dataset.test

    if QUANTIZED:
        model = QuantizedModel(model_path=args.model_args.tflite_model_path)
        num_correct = 0
        for inputs, labels in tqdm(dataset):
            outputs = np.argmax(model.predict(inputs))
            num_correct += 1 if (outputs == labels) else 0
        logger.info(f"Final Accuracy with quantization is {num_correct/len(dataset)*100:.2f} %")
    else:
        model = get_model(args.model_args, quantized=cmd_args.qh5)
        _, final_accuracy = model.evaluate(dataset.get_inputs(), dataset.get_labels())
        logger.info(f"Final Accuracy without quantization is {final_accuracy*100:.2f} %")


if __name__ == '__main__':
    cmd_args = CmdArgs()
    main(cmd_args)