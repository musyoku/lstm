# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
sys.path.append(os.path.split(os.getcwd())[0])
from args import args
from env import dataset, n_vocab, n_dataset, lstm, conf

n_epoch = 1000
n_train = 1000
batchsize = 64
total_time = 0

# 長すぎるデータはメモリに乗らないこともあります
max_length_of_chars = 100

# 学習初期は短い文章のみ学習し、徐々に長くしていきます。
# この機能が必要ない場合は最初から大きな値を設定します。
current_length_limit = 15
increasing_limit_interval = 1000

def make_batch():
	batch_array = []
	max_length_in_batch = 0
	for b in xrange(batchsize):
		length = current_length_limit + 1
		while length > current_length_limit:
			k = np.random.randint(0, n_dataset)
			data = dataset[k]
			length = len(data)
		batch_array.append(data)
		if length > max_length_in_batch:
			max_length_in_batch = length
	batch = np.full((batchsize, max_length_in_batch), -1.0, dtype=np.float32)
	for i, data in enumerate(batch_array):
		batch[i,:len(data)] = data
	return batch

def get_validation_data():
	max_length_in_batch = 0
	length = current_length_limit + 1
	while length > current_length_limit:
		k = np.random.randint(0, n_dataset)
		data = dataset[k]
		length = len(data)
	return data

for epoch in xrange(n_epoch):
	start_time = time.time()
	sum_loss = 0
	for t in xrange(n_train):
		batch = make_batch()
		sum_loss += lstm.train(batch)
		if t % 10 == 0:
			sys.stdout.write("\rLearning in progress...({:d} / {:d})".format(t, n_train))
			sys.stdout.flush()
	elapsed_time = time.time() - start_time
	total_time += elapsed_time
	sys.stdout.write("\r")
	print "epoch: {:d} loss: {:f} time: {:d} min total_time: {:d} min current_length_limit: {:d}".format(epoch, sum_loss / float(n_train), int(elapsed_time / 60), int(total_time / 60), current_length_limit)
	sys.stdout.flush()
	lstm.save(args.model_dir)
	if epoch % increasing_limit_interval == 0 and epoch != 0:
		current_length_limit = (current_length_limit + 5) if current_length_limit < max_length_of_chars else max_length_of_chars

	# Validation
	num_validation = 500
	correct = 0
	for i in xrange(num_validation):
		lstm.reset_state()
		phrase = get_validation_data()
		for n in xrange(len(phrase)):
			char = phrase[n]
			dist = lstm.predict(char, test=True)
			if char == dist:
				correct += 1
	print "validation: {:.3f}".format(correct / float(num_validation) / float(len(phrase)))

