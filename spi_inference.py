import os, random, time
import numpy as np

from args import get_args
from utils.CommandArgs import CommandLineArgs
from utils.logging_utils import setup_logger
from utils.spi_utils import SPIInterface
from utils.general import cprint
from utils.visualize import animate_complex_heatmap


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

	# sample_index = random.randint(0, len(frames))
	sample_index = 40

	interface = SPIInterface(args.spi_args)

	while True:
		sample_index = int(input(f"Send sample with index (0, {len(frames)}): "))
		sample_input, sample_label = (frames[sample_index], labels[sample_index])
		animate_complex_heatmap(sample_input)
		interface.send([0x02])
		time.sleep(0.1)
		response = interface.read_byte()
		# print(f"Response is: {response}")
		interface.send_array(sample_input)
		time.sleep(0.1)
		response = interface.read_byte()
		if response == 5:
			cprint("Transfer was succesful!", 'green')
		else:
			cprint("Transfer was not succesfull...", 'red')
		# print(f"Response is: {response}")

	# while True:
	# 	input("Press enter to send signal:")
	# 	logger.info("Sending signal via SPI!")
	# 	interface.send([0xAA])




if __name__ == '__main__':
	cmd_args = CmdArgs()
	main(cmd_args)