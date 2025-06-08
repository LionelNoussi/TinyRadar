from typing import Optional
from args import LoggingArgs
import os, sys, warnings
from datetime import datetime
import logging
from logging import StreamHandler, FileHandler, Formatter
from keras import callbacks
import wandb


LOGGER_NAME = 'MAConMIC'
LOGGER_AVAILABLE = False
_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    global LOGGER_AVAILABLE, _logger
    if LOGGER_AVAILABLE and _logger is not None:
        return _logger
    else:
        # Create default logger
        logging_args = LoggingArgs()
        logger = setup_logger(logging_args)
        return logger


def setup_logger(logging_args: LoggingArgs) -> logging.Logger:
    # set log dir of wandb
    os.environ["WANDB_DIR"] = logging_args.log_dir_path
    
    global LOGGER_AVAILABLE, _logger
    if LOGGER_AVAILABLE and _logger is not None:
        return _logger
    
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = Formatter(fmt=logging_args.format, datefmt=logging_args.datefmt)

    if logging_args.log_to_stdout:
        stream_handler = StreamHandler()
        stream_handler.setLevel(logging_args.level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if logging_args.log_to_file:
        os.makedirs(logging_args.log_dir_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{LOGGER_NAME}_{timestamp}.log"
        file_path = os.path.join(logging_args.log_dir_path, filename)

        file_handler = FileHandler(file_path, mode=logging_args.file_mode)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Allow normal exit on Ctrl+C
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    def warning_to_log(message, category, filename, lineno, file=None, line=None):
        logger.warning(f"{filename}:{lineno}: {category.__name__}: {message}")

    warnings.showwarning = warning_to_log

    _logger = logger
    LOGGER_AVAILABLE = True
    return logger


def deactivate_other_loggers(logging_args: LoggingArgs):
    # Get the root logger and all existing loggers
    logging.getLogger().setLevel(logging.CRITICAL)
    
    # Optionally, loop through all loggers and set their level to CRITICAL
    for logger_name in logging.root.manager.loggerDict:
        if logger_name != LOGGER_NAME:  # Replace 'my_logger' with your logger's name
            logging.getLogger(logger_name).setLevel(logging.CRITICAL)


class LoggingCallback(callbacks.Callback):
    def __init__(self, logger, log_every_n_batches=50):
        super().__init__()
        self.logger = logger
        self.log_every_n_batches = log_every_n_batches
        self.batch_count = 0

    def on_train_batch_end(self, batch, logs=None):
        self.batch_count += 1
        if self.batch_count % self.log_every_n_batches == 0 and logs is not None:
            wandb.log({
                "batch_loss": logs.get("loss"),
                "batch_accuracy": logs.get("accuracy")
            }, step=self.batch_count)
            msg = f"Step {self.batch_count}: " + ", ".join(f"{k}={v:.4f}" for k, v in logs.items())
            self.logger.info(msg)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = f"Epoch {epoch + 1}: " + ", ".join(f"{k}={v:.4f}" for k, v in logs.items())
        self.logger.info(msg)