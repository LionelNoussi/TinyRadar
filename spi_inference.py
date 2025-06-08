import spidev
import time
import os
from utils.CommandArgs import CommandLineArgs
from args import get_args
from utils.logging_utils import setup_logger


class CmdArgs(CommandLineArgs):
	fp: bool = False


def main(cmd_args):
	QUANTIZED = not cmd_args.fp
	# get yaml args, setup logger, get model and dataset
	args = get_args(cmd_args.config)
	logger = setup_logger(args.logging_args)
    
    frames_path = os.path.join(args.dataset_args.built_dataset_path, 'test/frames.npy')
	frames = np.load(frames_path)
	labels_path = os.path.join(args.dataset_args.built_dataset_path, 'test/txt_labels.npy')
	labels = np.load(labels_path)

	spi = spidev.SpiDev()
	spi.open(0, 0)  # Bus 0, CE0
	spi.max_speed_hz = 500000
	spi.mode = 0
	while True:
		input("Press enter to send signal:")
		logger.info("Sending signal via SPI!")
		spi.xfer2([0xAA])




if __name__ == '__main__':
	cmd_args = CmdArgs()
	main(cmd_args)