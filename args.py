# -*- coding: utf-8 -*-
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--use_gpu", type=int, default=1)
parser.add_argument("--fc_output_type", type=int, default=1)
parser.add_argument("--lstm_apply_batchnorm", type=int, default=0)
parser.add_argument("--text_dir", type=str, default="text")
parser.add_argument("--model_dir", type=str, default="model")
args = parser.parse_args()