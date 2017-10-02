import os, sys, uuid
import numpy as np
from chainer import serializers, functions
sys.path.append(os.path.join("..", ".."))
import lstm.nn as nn

class LSTM():
	def __init__(self, vocab_size, ndim_hidden, num_layers=2, dropout_embedding_softmax=0.5, dropout_rnn=0.2):
		self.vocab_size = vocab_size
		self.ndim_hidden = ndim_hidden
		self.num_layers = num_layers
		self.dropout_softmax = dropout_embedding_softmax
		self.dropout_rnn = dropout_rnn

		self.model = nn.Module()

		for _ in range(num_layers):
			self.model.add(nn.LSTM(ndim_hidden, ndim_hidden))

		self.model.embed = nn.EmbedID(vocab_size, ndim_hidden)
		self.model.dense = nn.Linear(ndim_hidden, vocab_size)

		for param in self.model.params():
			if param.name == "W" and param.data is not None:
				param.data[...] = np.random.normal(0, 0.01, param.data.shape)

		self.reset_state()

	def reset_state(self):
		for lstm in self.model.layers:
			lstm.reset_state()

	def __call__(self, x, flatten=False):
		batchsize = x.shape[0]
		# embedding layer
		embedding = self.model.embed(x)
		out_data = functions.dropout(embedding, self.dropout_softmax)

		# lstm layers
		for index, lstm in enumerate(self.model.layers):
			out_data = functions.dropout(out_data, self.dropout_rnn)
			out_data = lstm(out_data)

		# dense layers
		out_data = self.model.dense(functions.dropout(out_data, self.dropout_softmax))

		return out_data

	def save(self, filename):
		tmp_filename = str(uuid.uuid4())
		serializers.save_hdf5(tmp_filename, self.model)
		if os.path.isfile(filename):
			os.remove(filename)
		os.rename(tmp_filename, filename)

	def load(self, filename):
		if os.path.isfile(filename):
			print("Loading {} ...".format(filename))
			serializers.load_hdf5(filename, self.model)
			return True
		return False