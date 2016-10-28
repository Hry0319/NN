import csv
from io import StringIO
import numpy as np
import math


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
		
		self.likelihood = 0
		self.sum_table = []
		self.alpha = alpha
		self.mom = mom
		
	
	def parseTrain(self):
		input_file = open(self.filePath, 'r')
		
		input_data = []
		
		#
		#  raw training data from csv   
		#  [id][bias][features...][label]
		# 
		for row in csv.reader(input_file, delimiter = ','):
			input_data.append(row)
		
		self.case_num = len(input_data)	
		self.feature_num = len(input_data[0]) - 1  # with bias is 58
		print self.feature_num
		self.label = np.empty([self.case_num])
		self.feature = np.ones([self.case_num, self.feature_num])	
		
		#
		# init training data
		#____________________________________________________________
		# label   [0 ~ case_num]  : ground truth of training data
		# feature [1 ~ case_num]  : features of training data
		#      					  : [0] is bias 
		#
		for i in range(0, self.case_num):
			self.label[i] = int(input_data[i][len(input_data[i])-1])
			self.feature[i][1:self.feature_num] = [ float(value) for value in input_data[i][2:self.feature_num+1] ]

		self.tf_idf()
		
	
	def tf_idf(self):
		for i in range(1, 49):
			n = np.count_nonzero(self.feature[:, i])
			self.feature[:, i] *= np.log(self.case_num/n)
	


	def initNN(self , nh, no):
		self.ni = self.feature_num
		self.nh = nh
		self.no = no
	
		self.wI = np.matrix( np.random.rand(self.ni, self.nh) ) # wI ->  i x h
		self.wI -= 0.5
		self.wI *= 2
		self.wO = np.matrix( np.random.rand(self.nh, self.no) ) # wO ->  h x o
		self.wO -= 0.5

		self.cI = np.zeros( shape = (self.ni, self.nh))
		self.cO = np.zeros( shape = (self.nh, self.no))
		
		self.aH = np.zeros(self.nh)
		self.aO = np.zeros(self.no)
		
	def update(self, input):
	
		for index in range(self.nh):
			#sigmoid = tanh
			self.aH = np.tanh(input * self.wI)
		self.aH = np.array(self.aH).reshape(self.nh)
		
		for index in range(self.no):
    		#sigmoid = tanh
			self.aO = np.tanh(self.aH * self.wO)
		self.aO = np.array(self.aO).reshape(self.no)

		return
		
	def backPropagate(self, y, input):
		# y     : label[c]
		# input : feature[c]
		# label   [0 ~ case_num]  : ground truth of training data
		# feature [0 ~ case_num]  : features of training data

		o_deltas = np.zeros(self.no)
		for o_index in range(self.no):
			o_error	= y - self.aO[o_index]
			o_deltas[o_index] = np.square(1 - self.aO[o_index]) * o_error

		h_deltas = np.zeros(self.nh)
		for h_index in range(self.nh):
			h_error = o_deltas * self.wO[h_index]  # like transpose wO^t = o x h  but only cal 1 cow
			# print h_error, o_deltas , self.wO[h_index]
			# print self.aH[0] ,  h_error
			h_deltas[h_index] = np.square (1 - self.aH[h_index]) * h_error

		#
		# o_deltas 1 x o
		# h_deltas 1 x h
		#
		N = self.alpha
		M = self.mom

		# self.wO = np.array(self.wO).reshape(self.nh,self.no)
		# self.wI = np.array(self.wI).reshape(self.ni,self.nh)

        # update output weights
		for j in range(self.nh):
			for k in range(self.no):
				change = o_deltas[k] * self.aH[j]
				# print self.wO
				self.wO[j,k] += N * change + M * self.cO[j][k]
				self.cO[j][k] = change

		# update input weights
		for i in range(self.ni):
			for j in range(self.nh):
				change = h_deltas[j] * input[i]
				# print "_____ \n", self.wI
				self.wI[i,j] += N * change + M * self.cI[i][j]
				self.cI[i][j] = change
				
		error = np.square(y-self.aO) / 2
		return error
		
	def training(self):
		iter = 10

		#
		# iter  // do update & backpropagate
		#
		for i in range(iter):
			error = 0.0
			for c in range(self.case_num):
				self.update(self.feature[c])
				error += self.backPropagate([self.label[c]], self.feature[c])
				# self.backPropagate([self.label[c]], self.feature[c])
			print("iter : %d, error :%f"%(i, error))

		# 
	    # write out the model 
		#	
		f = open("nnModel.txt", 'w')
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
	train = spam("./spam_train.csv", 0.6, 0.1)
	train.parseTrain()
	train.initNN(3, 1)
	train.training()
	
if __name__ == '__main__':
	main()
