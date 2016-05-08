# -*- coding: utf-8 -*-
import numpy as np
from chainer import cuda, Variable
from chainer import links as L

class EmbedID(L.EmbedID):

	# vec: Numpy / Cupy
	def cosine_similarity(self, vec):
		W = self.W.data
		xp = cuda.get_array_module(*(vec,))
		w_norm = xp.sqrt(xp.sum(W ** 2, axis=1))
		v_norm = xp.sqrt(xp.sum(vec ** 2, axis=1))
		inner_product = W.dot(vec.T)
		norm = w_norm.reshape(1, -1).T.dot(v_norm.reshape(1, -1)) + 1e-6
		# 最初の軸がIDに対応
		return inner_product / norm

	# vec: Numpy / Cupy
	def reverse(self, vec, sample=False, to_cpu=False):
		xp = cuda.get_array_module(*(vec,))
		if sample:
			result = self.reverse_sampling(vec)	# Numpy ndarray
			if to_cpu or xp is np:
				return result
			return cuda.to_cpu(result)
		else:
			# 最初の軸がIDに対応
			result = self.reverse_argmax(vec)	# Numpy ndarray or Cupy ndarray
			if to_cpu and xp is cuda.cupy:
				result = cuda.to_cpu(result)
			return result
			
	# vec: Numpy / Cupy
	# Returns xp
	def reverse_argmax(self, vec):
		xp = cuda.get_array_module(*(vec,))
		cos = self.cosine_similarity(vec)
		# 最初の軸がIDに対応する
		return xp.argmax(cos, axis=0)

	# vec: Numpy / Cupy
	# Returns np
	def reverse_sampling(self, vec):
		xp = cuda.get_array_module(*(vec,))
		cos = self.cosine_similarity(vec)
		cos = xp.exp(cos)
		sum = xp.sum(cos, axis=0)
		sum = sum.reshape(1, -1)
		softmax = cos / sum
		if xp is cuda.cupy:
			softmax = cuda.to_cpu(softmax)
		softmax = softmax.T
		n_vec = softmax.shape[0]
		n_ids = softmax.shape[1]
		result = np.empty((n_vec,), dtype=np.int32)
		for t in xrange(n_vec):
			id = np.random.choice(np.arange(n_ids), p=softmax[t])
			result[t] = id
		return result