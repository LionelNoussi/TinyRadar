import subprocess, os
import tensorflow as tf
import numpy as np

from utils.CommandArgs import CommandLineArgs
from args import get_args
from utils.logging_utils import setup_logger
from model import get_model
from dataset import get_dataset


class CmdLineArgs(CommandLineArgs):
    model: bool = False # model
    dataset: bool = False # dataset
    both: bool = False # both


def main(cmd_args: CmdLineArgs):
    SETUP_MODEL = cmd_args.model or cmd_args.both
    SETUP_DATASET = cmd_args.dataset or cmd_args.both

    # get yaml args, setup logger, get model and get dataset
    args = get_args(cmd_args.config)
    logger = setup_logger(args.logging_args)
    model = get_model(args.model_args, quantized=False)
    dataset = get_dataset(args.dataset_args)

    if SETUP_MODEL:
        logger.info("Converting the model to tflite...")
        out_dir = args.conversion_args.out_dir
        tflite_model_path = os.path.join(out_dir, "my_model.tflite")
        
        # Convert Keras model to a tflite model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        if args.conversion_args.quantize:
            logger.info("Quantizing the model.")
            # Provide a representative dataset to ensure we quantize correctly.
            X_test = dataset.train._data
            def representative_dataset_gen():
                for i in range(100):
                    x = X_test[i:i+1]
                    yield [x.astype(np.float32)]

            converter.representative_dataset = representative_dataset_gen # type: ignore

            # Set the optimization flag.
            converter.optimizations = {tf.lite.Optimize.DEFAULT}

            # Enforce full-int8 quantization
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.target_spec.supported_types = [tf.int8]
            converter.inference_input_type = tf.int8  # or tf.uint8
            converter.inference_output_type = tf.int8  # or tf.uint8

        tflite_model = converter.convert()

        os.makedirs(out_dir, exist_ok=True)
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info("Generated tflite model.")

        logger.info("Converting model to a char array...")
        script_path = "utils/convert_tflite_to_cc.sh"
        result = subprocess.run([script_path, tflite_model_path, out_dir.rstrip('/')], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Error executing script:", result.stderr)
            logger.error("Conversion to char array was not succesfull.")
        else:
            logger.info("Succesfully converted model. Bash script output: \n" + result.stdout)
            logger.info("Done converting and saving the model.")

    if SETUP_DATASET:
        # TODO Is this really necessary, since my dataset won't fit on the MCU anyways?
        raise NotImplementedError()


if __name__ == '__main__':
    setup = CmdLineArgs()
    main(setup)