from args import get_args
from utils.logging_utils import setup_logger, LoggingCallback
from dataset import get_dataset
from model import get_model, CheckpointCallback
from utils.CommandArgs import CommandLineArgs
from datetime import datetime
import wandb
from keras.callbacks import ReduceLROnPlateau
import tensorflow_model_optimization as tfmot
from keras import optimizers


class CmdLineArgs(CommandLineArgs):
    dont_train: bool = False
    dont_test: bool = False
    plot: bool = False
    quantize: bool = False


def main(cmd_args: CmdLineArgs):
    # READ COMMAND LINE ARGUMENTS
    TRAIN = not cmd_args.dont_train
    TEST = not cmd_args.dont_test
    PLOT = cmd_args.plot
    QUANTIZE = cmd_args.quantize

    # Read yaml args and setup logger
    args = get_args(cmd_args.config)
    logger = setup_logger(args.logging_args)
    
    date = datetime.now()
    logger.info(f"TRAINING LOG {date.day}.{date.month}.{date.year}")
    logger.info(f"Command Line args: {cmd_args}")
    logger.info(f"General Args:\n{args}")

    dataset = get_dataset(args.dataset_args)
    train_dataset, val_dataset, test_dataset = dataset.unpack()
    
    model = get_model(args.model_args, quantized=QUANTIZE)
    model.summary(print_fn=lambda x: logger.debug(x))

    logger.info("Model and Dataset were succesfully created.")

    # Trains the model on the training dataset
    if TRAIN:
        wandb.init(project="MLonMCU_FinalProject")
        logger.info("Starting training...")
        logger.info(f"Training for {args.model_args.num_epochs} epochs with batch size {args.model_args.batch_size}.")
        model.fit(
            train_dataset.get_inputs(),
            train_dataset.get_labels(),
            batch_size=args.model_args.batch_size,
            epochs=args.model_args.num_epochs,
            verbose=0, # type: ignore
            validation_data=(val_dataset.get_inputs(), val_dataset.get_labels()),
            callbacks=[
                LoggingCallback(logger),
                CheckpointCallback(args.model_args.checkpoint_dir),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6), # type: ignore
            ]
        )
        logger.info("Training finished succesfully!")

    # Does inference on the test dataset
    if TEST:
        logger.info("Evaluating model on test set...")
        loss, final_accuracy = model.evaluate(test_dataset.get_inputs(), test_dataset.get_labels())
        logger.info(f"Test Loss is {loss}")
        logger.info(f"Test Accuracy is {final_accuracy*100:.2f} %")

    # Saves the Model to the args.save_model_path and overrides
    # any already saved models with the same file_name. Make sure
    # to rename any models, which should be kept.
    if args.model_args.save_model:
        logger.info(f"Saving model to {args.model_args.save_model_path}")
        model.save(args.model_args.save_model_path, overwrite=True)

    if PLOT:
        from utils.visualize import plot_confusion_matrix
        plot_confusion_matrix(model, dataset)


if __name__ == '__main__':
    cmd_args = CmdLineArgs()
    main(cmd_args)