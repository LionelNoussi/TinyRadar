from typing import (
    Optional
)

from args import Args, ModelArgs
from utils.logging_utils import get_logger


import os
import heapq
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras import models
from keras import optimizers
from keras import regularizers
from keras.callbacks import Callback
import tensorflow_model_optimization as tfmot


class QuantizedModel:

    def __init__(self, model_path) -> None:
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.inputs_index = self.interpreter.get_input_details()[0]['index']
        self.outputs_index = self.interpreter.get_output_details()[0]['index']
        self.in_scale, self.in_zero_point = self.interpreter.get_input_details()[0]['quantization']
        self.out_scale, self.out_zero_point = self.interpreter.get_output_details()[0]['quantization']

    def quantize_inputs(self, inputs):
        inputs = inputs.astype(np.float32)
        inputs = inputs / self.in_scale + self.in_zero_point
        inputs = np.clip(np.round(inputs), -128, 127).astype(np.int8)
        return inputs
    
    def predict(self, float_inputs):
        quant_inputs = self.quantize_inputs(float_inputs)
        quant_outputs = self.q_predict(quant_inputs)
        float_outputs = self.dequantize_outputs(quant_outputs)
        return float_outputs
    
    def q_predict(self, quant_inputs):
        self.interpreter.set_tensor(self.inputs_index, quant_inputs)
        self.interpreter.invoke()
        quant_outputs = self.interpreter.get_tensor(self.outputs_index)
        return quant_outputs

    def dequantize_outputs(self, outputs):
        outputs = (np.array(outputs).astype(np.float32) - self.out_zero_point) * self.out_scale
        return outputs


def get_model(model_args: ModelArgs, quantized: bool = False) -> keras.Model:
    logger = get_logger()

    if model_args.load_model:
        logger.info(f"Loading model from {model_args.load_model_path}.")
        if not quantized:
            return keras.models.load_model(model_args.load_model_path, custom_objects={'dtype': tf.float32}) # type: ignore
        else:
            with tfmot.quantization.keras.quantize_scope():
                return keras.models.load_model(model_args.load_model_path) # type: ignore

    logger.info("Creating a new model from scratch.")

    input_shape = (16, 82, 10)
    # model = models.Sequential([
    #     layers.Input(shape=input_shape),
        
    #     layers.Conv2D(8, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
    #     layers.Conv2D(8, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
    #     layers.Conv2D(12, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
    #     layers.Conv2D(12, kernel_size=(1, 3), strides=(1, 2), padding='same', activation='relu'),
    #     layers.Conv2D(16, kernel_size=3, padding='same', activation='relu'),  # new small conv layer
        
    #     layers.Dropout(0.3),

    #     layers.GlobalAveragePooling2D(),
    #     layers.Dense(32, activation='relu'),  # tiny dense layer
    #     layers.Dense(5, activation='softmax')
    # ])
    model = models.Sequential([
        layers.Input(shape=(16, 82, 10)),

        # Less aggressive downsampling early
        layers.Conv2D(12, kernel_size=3, strides=1, padding='same', activation='relu'),
        layers.Conv2D(12, kernel_size=3, strides=2, padding='same', activation='relu'),
        layers.SpatialDropout2D(0.3),

        layers.Conv2D(16, kernel_size=3, strides=1, padding='same', activation='relu'),
        layers.Conv2D(16, kernel_size=3, strides=2, padding='same', activation='relu'),
        layers.SpatialDropout2D(0.3),

        layers.Conv2D(24, kernel_size=3, strides=1, padding='same', activation='relu'),
        layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'),

        layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),  # Deep feature layer

        layers.Dropout(0.4),

        layers.GlobalAveragePooling2D(),
        layers.Dense(96, activation='relu', kernel_regularizer=regularizers.l2(1e-5)),
        layers.Dense(11, activation='softmax')
    ])

    model.build(input_shape=(None, 16, 82, 10))
    if quantized:
        model = tfmot.quantization.keras.quantize_model(model)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


class CheckpointCallback(Callback):
    def __init__(self, dir_path, top_n=3):
        super().__init__()
        self.dir_path = dir_path  # Must include something like 'epoch={epoch:02d}-val_acc={val_accuracy:.4f}.h5'
        os.makedirs(self.dir_path, exist_ok=True)
        self.filename = "checkpoint_epoch_{epoch:02d}_acc_{val_accuracy:.4f}.h5"
        self.top_n = top_n
        self.top_scores = []  # Min-heap of (val_accuracy, filepath)
        self.model: keras.Model
        self.logger = get_logger()
    
    def on_epoch_end(self, epoch, logs: Optional[dict]=None):
        if logs is None:
            print("Warning: No logs at the end of epoch. Skipping checkpoint.")
            return
        
        val_acc = logs.get("val_accuracy")

        if val_acc is None:
            print("Warning: Validation accuracy not available. Skipping checkpoint.")
            return

        # If we have fewer than N, accept by default
        if len(self.top_scores) < self.top_n or val_acc > self.top_scores[0][0]:
            # Save current model
            fname = self.filename.format(epoch=epoch + 1, val_accuracy=val_acc)
            save_path = os.path.join(self.dir_path, fname)

            self.logger.info(f"Saving Checkpoint: {fname}")
            self.model.save(save_path)

            # Add to heap
            heapq.heappush(self.top_scores, (val_acc, save_path))

            # Remove worst if we exceed top_n
            if len(self.top_scores) > self.top_n:
                worst = heapq.heappop(self.top_scores)
                self.logger.info(f"Deleting Checkpoint {worst[1]} due to top_n={self.top_n}")
                if os.path.exists(worst[1]):
                    os.remove(worst[1])