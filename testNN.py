import csv
from io import StringIO
import numpy as np
from itertools import islice

# def update(input):
	# aH = []
	# aO = []
	# for index in range(nh):
	# 	aH = np.tanh(input * wI)
	# aH = np.array(aH).reshape(nh)
	# for index in range(no):
	# 	aO = np.tanh(aH * wO)
	# aO = np.array(aO).reshape(no)


def test(testFile, weightFile):
	hidden = 0
	feature_num = 57
	# with open('ansNN.csv', 'w', newline='') as out:
	with open('ansNN.csv', 'wb') as out:
		reader = csv.reader(open(testFile,'r'))
		writer = csv.writer(out, delimiter = ',')
		f = open("nnZ.txt", "w")
		m = open(weightFile, 'r')

		line = m.readline()
		hidden = int((line.strip().split(' '))[1])


		wI = np.matrix(np.zeros(shape = (feature_num, hidden)))
		wO = []


		i = 0
		for line in m.readlines():
			line = line.strip().split(' ')
			if line[0] == 'nh:':
				continue
			if i < feature_num:
				wI[i] = [float(val) for val in line]
				i += 1
			else:
				wO.append( [float(val) for val in line] )

		# wO = np.matrix(wO)
		# print wO

		writer.writerow(["id", "label"])
		row = [0, 0]
		data = np.ones(feature_num)
		aH = np.zeros(hidden)
		aO = np.zeros(1)
		for case in reader:
			row[0] = str(case[0])
			# data[1:] = np.asarray([float(val) for val in case[1:]])
			data[:] = np.asarray([float(val) for val in case[1:]])

			# break
			# aH = 1/ (1+ np.exp(-1*(data*wI)))
			# aO = 1/ (1+ np.exp(-1*np.sum(wO*aH)))
			# print data * wI
			# print aH*wO
			# break

			aH = np.tanh((data * wI))
			# aH /= 2
			# aH += 0.5

			# print aH
			# break
			aO = np.tanh(np.sum(aH*wO))
			# aO /= 2
			# aO += 0.5
			# print aO
			# break

			f.write(str(aO)+'\n')
			if aO > 0:
				row[1] = "1"
			else:
				row[1] = "0"
			writer.writerow(row)

		f.close()

def main():
	test("./spam_data/spam_test.csv", "nnModel.txt")

if __name__ == "__main__":
	main()
