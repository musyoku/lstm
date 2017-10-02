from __future__ import division
from __future__ import print_function
import argparse, pickle, os
import numpy as np
import chainer
from chainer import cuda, functions
from model import LSTM
from lstm.dataset import ID_EOS

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--gpu-device", "-g", type=int, default=0)
	parser.add_argument("--dropout-embedding-softmax", "-dos", type=float, default=0.5)
	parser.add_argument("--dropout-rnn", "-dor", type=float, default=0.2)
	parser.add_argument("--ndim-hidden", "-dh", type=int, default=640)
	parser.add_argument("--num-layers", "-nl", type=int, default=2)
	parser.add_argument("--num-to-generate", "-n", type=int, default=100)
	parser.add_argument("--model-filename", "-m", type=str, default="model.hdf5")
	parser.add_argument("--vocab-filename", "-v", type=str, default="vocab.pkl")
	args = parser.parse_args()

	assert args.num_layers > 0
	assert args.ndim_hidden > 0
	assert os.path.isfile(args.vocab_filename) is True

	with open(args.vocab_filename, "rb") as f:
		vocab_str_id = pickle.load(f)
		vocab_id_str = pickle.load(f)

	vocab_size = len(vocab_str_id)
	lstm = LSTM(vocab_size=vocab_size,
				ndim_hidden=args.ndim_hidden, 
				num_layers=args.num_layers,
				dropout_embedding_softmax=args.dropout_embedding_softmax, 
				dropout_rnn=args.dropout_rnn)
	assert lstm.load(args.model_filename)

	for n in range(args.num_to_generate):
		lstm.reset_state()
		x_sequence = np.asarray([ID_EOS]).astype(np.int32)[None, :]
		for t in range(1000):
			distribution = functions.softmax(lstm(x_sequence[:, t])).data[0]
			y_data = np.random.choice(np.arange(distribution.size), size=1, p=distribution).astype(np.int32)
			x_sequence = np.concatenate((x_sequence, y_data[None, :]), axis=1)
			if y_data[0] == ID_EOS:
				break
		tokens = []
		for t in range(1, x_sequence.size - 2):
			tokens.append(vocab_id_str[x_sequence[0, t]])
		print(" ".join(tokens))


if __name__ == "__main__":
	main()