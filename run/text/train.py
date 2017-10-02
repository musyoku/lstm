from __future__ import division
from __future__ import print_function
import argparse, sys, os, time, math, pickle
import numpy as np
import chainer
from chainer import cuda, serializers, functions
from model import LSTM
from lstm.optim import Optimizer
from lstm.dataset import read_data

def clear_console():
	printr("")

def printr(string):
	sys.stdout.write("\r\033[2K")
	sys.stdout.write(string)
	sys.stdout.flush()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--batchsize", "-b", type=int, default=64)
	parser.add_argument("--seq-length", "-l", type=int, default=35)
	parser.add_argument("--total-epochs", "-e", type=int, default=300)
	parser.add_argument("--gpu-device", "-g", type=int, default=0)
	parser.add_argument("--grad-clip", "-gc", type=float, default=5)
	parser.add_argument("--learning-rate", "-lr", type=float, default=1)
	parser.add_argument("--weight-decay", "-wd", type=float, default=0.000001)
	parser.add_argument("--dropout-embedding-softmax", "-dos", type=float, default=0.5)
	parser.add_argument("--dropout-rnn", "-dor", type=float, default=0.2)
	parser.add_argument("--momentum", "-mo", type=float, default=0.9)
	parser.add_argument("--optimizer", "-opt", type=str, default="msgd")
	parser.add_argument("--ndim-hidden", "-dh", type=int, default=640)
	parser.add_argument("--num-layers", "-nl", type=int, default=2)
	parser.add_argument("--lr-decay-epoch", "-lrd", type=int, default=20)
	parser.add_argument("--model-filename", "-m", type=str, default="model.hdf5")
	parser.add_argument("--vocab-filename", "-v", type=str, default="vocab.pkl")
	parser.add_argument("--train-filename", "-train", default=None)
	parser.add_argument("--dev-filename", "-dev", default=None)
	parser.add_argument("--test-filename", "-test", default=None)
	args = parser.parse_args()

	assert args.num_layers > 0
	assert args.ndim_hidden > 0

	dataset_train, dataset_dev, dataset_test, vocab_str_id, vocab_id_str = read_data(args.train_filename, args.dev_filename, args.test_filename)
	dataset_dev = np.asarray(dataset_dev, dtype=np.int32)
	dataset_test = np.asarray(dataset_test, dtype=np.int32)
	assert len(dataset_train) > 0

	if os.path.isfile(args.vocab_filename):
		with open(args.vocab_filename, "rb") as f:
			vocab_str_id = pickle.load(f)
			vocab_id_str = pickle.load(f)
	else:
		with open(args.vocab_filename, "wb") as f:
			pickle.dump(vocab_str_id, f)
			pickle.dump(vocab_id_str, f)

	print("#train = {}".format(len(dataset_train)))
	print("#dev = {}".format(len(dataset_dev)))
	print("#test = {}".format(len(dataset_test)))

	vocab_size = len(vocab_str_id)
	lstm = LSTM(vocab_size=vocab_size,
				ndim_hidden=args.ndim_hidden, 
				num_layers=args.num_layers,
				dropout_embedding_softmax=args.dropout_embedding_softmax, 
				dropout_rnn=args.dropout_rnn)
	lstm.load(args.model_filename)

	total_iterations_train = len(dataset_train) // (args.seq_length * args.batchsize)

	optimizer = Optimizer(args.optimizer, args.learning_rate, args.momentum)
	optimizer.setup(lstm.model)
	if args.grad_clip > 0:
		optimizer.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))
	if args.weight_decay > 0:
		optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

	using_gpu = False
	if args.gpu_device >= 0:
		cuda.get_device(args.gpu_device).use()
		lstm.model.to_gpu()
		using_gpu = True
	xp = lstm.model.xp

	training_start_time = time.time()
	for epoch in range(args.total_epochs):

		sum_loss = 0
		epoch_start_time = time.time()

		# training
		for itr in range(total_iterations_train):
			# sample minbatch
			batch_offsets = np.random.randint(0, len(dataset_train) - args.seq_length - 1, size=args.batchsize)
			x_batch = np.empty((args.batchsize, args.seq_length), dtype=np.int32)
			t_batch = np.empty((args.batchsize, args.seq_length), dtype=np.int32)
			for batch_index, offset in enumerate(batch_offsets):
				sequence = dataset_train[offset:offset + args.seq_length]
				teacher = dataset_train[offset + 1:offset + args.seq_length + 1]
				x_batch[batch_index] = sequence
				t_batch[batch_index] = teacher

			if using_gpu:
				x_batch = cuda.to_gpu(x_batch)
				t_batch = cuda.to_gpu(t_batch)

			# update model parameters
			with chainer.using_config("train", True):
				lstm.reset_state()
				loss = 0
				for t in range(args.seq_length):
					x_data = x_batch[:, t]
					t_data = t_batch[:, t]
					y_data = lstm(x_data)
					loss += functions.softmax_cross_entropy(y_data, t_data)

				lstm.model.cleargrads()
				loss.backward()
				optimizer.update()

				sum_loss += float(loss.data)
				assert sum_loss == sum_loss, "Encountered NaN!"

			printr("Training ... {:3.0f}% ({}/{})".format((itr + 1) / total_iterations_train * 100, itr + 1, total_iterations_train))

		lstm.save(args.model_filename)

		# evaluation
		perplexity = -1
		negative_log_likelihood = 0
		if epoch % 10 == 0:
			x_sequence = dataset_dev[:-1]
			t_sequence = dataset_dev[1:]
			seq_length_dev = len(x_sequence)

			if using_gpu:
				x_sequence = cuda.to_gpu(x_sequence)[None, :]
				t_sequence = cuda.to_gpu(t_sequence)[None, :]

			with chainer.no_backprop_mode() and chainer.using_config("train", False):
				lstm.reset_state()
				for t in range(seq_length_dev):
					x_data = x_sequence[:, t]
					t_data = t_sequence[:, t]
					y_data = lstm(x_data)
					negative_log_likelihood += float(functions.softmax_cross_entropy(y_data, t_data).data)

					printr("Computing perplexity ...{:3.0f}% ({}/{})".format((t + 1) / seq_length_dev * 100, t + 1, seq_length_dev))

			assert negative_log_likelihood == negative_log_likelihood, "Encountered NaN!"
			perplexity = math.exp(negative_log_likelihood / len(dataset_dev))

		clear_console()
		print("Epoch {} done in {} sec - loss: {:.6f} - log_likelihood: {} - ppl: {} - lr: {:.3g} - total {} min".format(
			epoch + 1, int(time.time() - epoch_start_time), sum_loss / total_iterations_train, 
			int(-negative_log_likelihood), int(perplexity), optimizer.get_learning_rate(),
			int((time.time() - training_start_time) // 60)))

		if epoch >= args.lr_decay_epoch:
			optimizer.decrease_learning_rate(0.98, final_value=1e-5)

if __name__ == "__main__":
	main()