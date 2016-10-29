import csv
from io import StringIO
import numpy as np
import math
from multiprocessing import Pool

class spam:
	def __init__(self, path, alpha, mom):
		self.filePath = path
		self.case_num = 0
		self.feature_num = 0
		self.label = []
		self.feature = []
		self.no = 0
		self.ni = 0
		self.nh = 0
		self.alpha = alpha
		self.mom = mom

	def parseTrain(self):
		input_file = open(self.filePath, 'r')
		input_data = []
		#
		#  raw training data from csv
		#  [id][features...][label]
		#
		for row in csv.reader(input_file, delimiter = ','):
			input_data.append(row)

		self.case_num = len(input_data)
		self.feature_num = len(input_data[0]) - 2  # 57  without bias
		# self.feature_num = len(input_data[0]) - 1  # 58  with bias
		# print self.feature_num
		self.label = np.empty([self.case_num])
		self.feature = np.ones([self.case_num, self.feature_num])

		#
		# init training data
		#____________________________________________________________
		# label   [0 ~ case_num]  : ground truth of training data
		# feature [bias, 1 ~ feature_num]  : features of training data
		#
		for i in range(0, self.case_num):
			self.label[i] = int(input_data[i][len(input_data[i])-1])
			self.feature[i] = [ float(value) for value in input_data[i][1:self.feature_num+1] ]  # without bias
			# self.feature[i][1:] = [ float(value) for value in input_data[i][1:self.feature_num+1] ]  # with bias
		# self.tf_idf()
		# for i in range(49, self.feature_num):
		# 	self.feature[:, i] = np.log(self.feature[:, i]+1)
		# 	print i, self.feature[:, i]
		# self.feature[:, 55] = np.log10(self.feature[:, 55])

	# def tf_idf(self):
	# 	for i in range(0, 49):
	# 		n = np.count_nonzero(self.feature[:, i])
	# 		self.feature[:, i] *= np.log(self.case_num/n)

	def initNN(self , nh, no):
		self.ni = self.feature_num
		self.nh = nh
		self.no = no
		self.wI = np.matrix( np.random.uniform(-1, 1, (self.ni, self.nh) )) # wI ->  i x h
		self.wO = np.matrix( np.random.uniform(-0.25, 0.25, (self.nh, self.no) )) # wO ->  h x o
		self.cI = np.zeros( shape = (self.ni, self.nh))
		self.cO = np.zeros( shape = (self.nh, self.no))
		self.aH = np.zeros(self.nh)
		self.aO = np.zeros(self.no)
		return

	def update(self, input):
		input = np.tanh(input)
		# for index in range(self.nh):
		# 	self.aH = np.tanh(input * self.wI + np.random.randint(2))
		self.aH = np.tanh(input * self.wI )#+ np.random.randint(2))
		# self.aH /= 2
		# self.aH += 0.5
		# print self.aH

		# for index in range(self.no):
		# 	self.aO = np.tanh(self.aH * self.wO + np.random.randint(2))
		self.aO = np.tanh(self.aH * self.wO )#+ np.random.randint(2))
		# remap to 0~1
		self.aO /= 2
		self.aO += 0.5
		# print self.aO

		self.aH = np.array(self.aH).reshape(self.nh)
		self.aO = np.array(self.aO).reshape(self.no)

		return self.aO

	def backPropagate(self, y, input):
		# y     : label[c]
		# input : feature[c]
		# label   [0 ~ case_num]  : ground truth of training data
		# feature [0 ~ case_num]  : features of training data
		# o_deltas 1 x o
		# h_deltas 1 x h
		o_deltas = np.zeros(self.no)
		for o_index in range(self.no):
			o_error	= y - self.aO[o_index]
			# print y, o_error
			o_deltas[o_index] = (1.0 -  np.square(self.aO[o_index])) * o_error

		h_deltas = np.zeros(self.nh)
		# for h_index in range(self.nh):
		# 	h_error = o_deltas * self.wO[h_index]  # like transpose wO^t = o x h  but only cal 1 cow
		# 	# print h_error
		# 	h_deltas[h_index] = (1.0 -  np.square(self.aH[h_index])) * h_error
		h_error = (o_deltas * self.wO.T).reshape(self.nh)
		h_deltas[:] = (1.0 -  np.square(self.aH[:])) * h_error[:]

		# self.wO = np.array(self.wO).reshape(self.nh,self.no)
		# self.wI = np.array(self.wI).reshape(self.ni,self.nh)

		N = self.alpha
		M = self.mom
        # update output weights
		for j in range(self.nh):
			for k in range(self.no):
				change = o_deltas[k] * self.aH[j]
				self.wO[j,k] += N * change + M * self.cO[j][k]
				self.cO[j][k] = change
		# update input weights
		for i in range(self.ni):
			for j in range(self.nh):
				change = h_deltas[j] * input[i]
				self.wI[i,j] += N * change + M * self.cI[i][j]
				self.cI[i][j] = change

		return (np.square(y-self.aO) / 2)

	def training(self):
		iter = 10000

		#
		# iter  // do update & backpropagate
		#
		for i in range(iter+1):
			error = 0.0
			for c in range(self.case_num):
				# print self.update(self.feature[c]) ,
				self.update(self.feature[c])
				error += self.backPropagate([self.label[c]], self.feature[c])
				# self.backPropagate([self.label[c]], self.feature[c])
			print("iter : %d, error :%f"%(i, error))

			if iter % 10 == 0:
				#
			    # write out the model
				#
				# modelname = "nnModel"+str(iter)+".txt"
				modelname = "nnModel.txt"
				f = open(modelname, 'w')
				f.write("nh: " + str(self.nh) + "\n")
				for i in range(self.ni):
					for j in range(self.nh):
						f.write(str(self.wI[i,j]))
						f.write(' ')
					f.write('\n')
				for i in range(self.nh):
					for j in range(self.no):
						f.write(str(self.wO[i,j]))
						f.write(' ')
					f.write('\n')
				f.close()

def main():
	train = spam("./spam_train.csv", 0.001, 0.00001)
	train.parseTrain()
	train.initNN(5, 1)
	train.training()

if __name__ == '__main__':
	main()
