import os, sys
sys.path.insert(0, os.getcwd())
from model import get_model
from args import get_args
from utils.visualize import visualize_model

args = get_args("raspy")
model = get_model(args.model_args)
visualize_model(model)