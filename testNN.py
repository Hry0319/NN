import csv
from io import StringIO
import numpy as np
from itertools import islice

def test(testFile, weightFile):
	hidden = 3
	# with open('ansNN.csv', 'w', newline='') as out:
	with open('ansNN.csv', 'wb') as out:
		reader = csv.reader(open(testFile,'r'))
		writer = csv.writer(out, delimiter = ',')
		
		wI = np.matrix(np.zeros(shape = (58, hidden)))
		wO = np.matrix(np.zeros(shape = (hidden, 1)))


		f = open("nnZ.txt", "w")
		m = open(weightFile, 'r')

		i = 0
		for line in m.readlines():
			line = line.strip().split(' ')
			if i < 58: 
				wI[i] = [float(val) for val in line]
				i += 1
			else:
				wO[0] = [float(val) for val in line]
		
		
		writer.writerow(["id", "label"])
		row = [0, 0]
		data = np.ones(58)
		aH = np.zeros(hidden)
		aO = np.zeros(1)
		for case in reader:
			row[0] = str(case[0])
			data[1:] = np.asarray([float(val) for val in case[1:]])
			
			# aH = 1/ (1+ np.exp(-1*np.sum(wI*data)))
			# aO = 1/ (1+ np.exp(-1*np.sum(wO*aH)))
			# print data * wI
			

			aH = np.tanh(data * wI)
			# print aH
			aO = np.tanh(np.sum(wO*aH))
			
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