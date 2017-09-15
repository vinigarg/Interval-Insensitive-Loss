import binning
import sys
import csv
import numpy as np
from cvxopt import solvers
from cvxopt import matrix

filename = "machine"
C=1
lam = 1000
reg = -1 * float(1/lam)
map_range = {}

def calculate_b(map_range):
	b = {}
	for key in map_range.keys():
		range = map_range[key]
		b[key] = [range[0]+0.5, range[1]+0.5]
	return b

def classLabel(value, map_range):
	value = float(value)
	key_set = map_range.keys()
	result=0
	key_set.sort()
	for key in key_set:
		if map_range[key][0]< value and map_range[key][1]>=value:
			result=key
			break	
	return result

def ordinal_data(data, map_range):
	new_data = {}
	for x in xrange(len(data)):
		label = classLabel(data[x][-1], map_range)
		if label not in new_data.keys():
			new_data[label] = []
		new_data[label].append(data[x])
	return new_data

def read_data(filename, map_range):
	data = []
	count = 0
	with open(filename,'r') as f:
		reader = csv.reader(f)
		for row in reader:
			data.append([float(x) for x in row])
			count+=1
	return count, data

def generate (index, data) :
	data_t = []
	for i in xrange(len(index)):
		data_t.append(data[index[i]])
	return data_t

def form_ranges(Y, k, miny):
	x=miny
	classLabel=1
	map_range={}
	while x < 1000:
		yl = x
		x+=int(k)
		yr = x
		map_range[classLabel]=[yl,yr]
		classLabel+=1
	return  map_range

def normalize(full_data):
	from sklearn import preprocessing
	a = np.array(full_data)
	A_scaled = preprocessing.scale(a)
	m = np.amin(A_scaled, axis=0)
	return len(A_scaled), A_scaled, m[-1]


def compute(arg):
	data_length, full_data = read_data(filename, map_range)
	data_length, full_data, miny = normalize(full_data)
	Y = np.ceil(1.0 * len(data) / int(k))
	map_range = form_ranges(Y, arg, miny)


	from sklearn import model_selection
	kf = model_selection.KFold(n_splits=5, shuffle=True)

	acc1=[]
	acc2=[]
	MAE=[]
	print map_range
	for train_index, test_index in kf.split(full_data):
		# training the sample
		data = generate(train_index, full_data)
		data_length = len(data)

		b_map = calculate_b(map_range)	# contains dict of byl and byr per class
		A = [1.0]*2*data_length
		b = [0.0]
		G = np.identity(2*data_length)
		h = [float(C/data_length)]*2*data_length
		P_left = []
		P_right= []

		#i,j loop
		for d in data:
			label = classLabel(d[-1], map_range)
			rang = map_range[label]
			b_rang = b_map[label]
			list_left = [0]
			list_right = [0]
			
			#y loop
			for t in data:
				if rang[0] >= t[-1]:
					list_left.append(rang[0]-t[-1]-b_rang[0])
				if rang[1] <= t[-1]:
					list_right.append(rang[1]-t[-1]-b_rang[1])

			P_left.append(max(list_left))
			P_right.append(max(list_right))

		P = P_left  + P_right
		Q=[[0]*2*data_length]*2*data_length
		Q = np.array(Q)
		#part 1
		for i, data_i in enumerate(data):
			label_i = classLabel(data_i[-1], map_range)
			rang_i = map_range[label_i]
		
			for j, data_j in enumerate(data):
				label_j = classLabel(data_j[-1], map_range)
				rang_j = map_range[label_j]
				list_left = [0]
				list_right = [0]
				
				#y loop
				for t in data:
					if rang_i[0] >= t[-1]:
						list_left.append(t[-1]-rang_i[0])
					if rang_j[0] >= t[-1]:
						list_right.append(t[-1]-rang_j[0])

				x_i = np.array(data_i[:-1])
				x_j = np.array(data_j[:-1])
				Q[i][j] = x_i.dot(x_j) * max(list_left) * max(list_right) * reg

		#part 2
		for i, data_i in enumerate(data):
			label_i = classLabel(data_i[-1], map_range)
			rang_i = map_range[label_i]
			for j, data_j in enumerate(data):
				label_j = classLabel(data_j[-1], map_range)
				rang_j = map_range[label_j]
				list_right = [0]
				list_left = [0]
				
				#y loop
				for t in data:
					if rang_i[1] <= t[-1]:
						list_left.append(t[-1]-rang_i[1])
					if rang_j[1] <= t[-1]:
						list_right.append(t[-1]-rang_j[1])

				x_i = np.array(data_i[:-1])
				x_j = np.array(data_j[:-1])
				Q[i][data_length+j] = max(list_left) * max(list_right) * x_i.dot(x_j) * reg

		#part 3
		for i, data_i in enumerate(data):
			label_i = classLabel(data_i[-1], map_range)
			rang_i = map_range[label_i]
			
			for j, data_j in enumerate(data):
				label_j = classLabel(data_j[-1], map_range)
				rang_j = map_range[label_j]
				list_left = [0]
				list_right = [0]
				#y loop
				for t in data:
					if rang_i[0] >= t[-1]:
						list_left.append(t[-1]-rang_i[0])
					if rang_j[1] <= t[-1]:
						list_right.append(t[-1]-rang_j[1])

				x_i = np.array(data_i[:-1])
				x_j = np.array(data_j[:-1])
				Q[i+data_length][j] = max(list_left) * max(list_right) * x_i.dot(x_j) *reg


		#part 4
		for i, data_i in enumerate(data):
			label_i = classLabel(data_i[-1], map_range)
			rang_i = map_range[label_i]
			for j, data_j in enumerate(data):

				label_j = classLabel(data_j[-1], map_range)
				rang_j = map_range[label_j]
			
				list_left = [0]
				list_right = [0]
			
				#y loop
				for t in data:
					if rang_i[1] <= t[-1]:
						list_left.append(t[-1]-rang_i[1])
					if rang_j[0] >= t[-1]:
						list_right.append(t[-1]-rang_j[0])

				x_i = np.array(data_i[:-1])
				x_j = np.array(data_j[:-1])
				Q[i+data_length][j+data_length] = max(list_left) * max(list_right) * x_i.dot(x_j) * reg

		sol = solvers.qp(matrix(np.array(Q), tc='d'),
							matrix(np.array(P), tc='d'),
						 	matrix(np.array(G), tc='d'),
						 	matrix(np.array(h), tc='d'),
						 	matrix(np.array(A),(1,len(A)), tc='d'),
						 	matrix(np.array(b), tc='d')	
						)

		W=[]
		if sol['status'] == 'optimal':	
			# w generation
			Alpha = sol['x']

			for i, data_i in enumerate(data):
				label = classLabel(data_i[-1], map_range)
				rang = map_range[label]

				alpha_i = Alpha[i]
				alpha_i_star = Alpha[i+data_length]

				list_left = [0]
				list_right = [0]
				x_i = np.array(data_i[:-1])
			
				for t in data:
					if rang[0] >= t[-1]:
						list_left.append(t[-1]-rang[0])
					if rang[1] <= t[-1]:
						list_right.append(t[-1]-rang[1])

				temp1 = max(list_left)*alpha_i
				temp1 = x_i.dot(temp1)
				temp2 = max(list_right)*alpha_i_star
				temp2 = x_i.dot(temp2)

				if len(W)==0:
					W = np.add(temp1, temp2)
				else:
					temp1 = np.add(temp1, temp2)
					W = np.add(W, temp1)	

			W = W * reg
			# print(W)
			# testing for accuracy
			data_test = generate(test_index, full_data)
			accuracy1, accuracy2, mae = test(data_test, W, map_range)
			acc1.append(accuracy1)
			acc2.append(accuracy2)
			MAE.append(mae)
			# print accuracy

	print np.array(acc1).mean(), np.array(acc2).mean(), np.array(MAE).mean()

import random
def partial1 ( pre_y, ex_y):
	a = int(random.random() * 3)
	# print a , a%3
	if a%3 == 0:
		return pre_y >= ex_y-1 and pre_y <= ex_y+1
	elif a%3 ==1:
		return pre_y >= ex_y and pre_y <= ex_y+1
	elif a%3 == 2: 
		return pre_y >= ex_y-1 and pre_y <= ex_y
	return False 

def partial2 ( pre_y, ex_y):
	a = int(random.random() * 8)
	if a%8 == 0:
		return pre_y >= ex_y-1 and pre_y <= ex_y
	elif a%8 == 1:
		return pre_y >= ex_y and pre_y <= ex_y+1
	elif a%8 == 2: 
		return pre_y >= ex_y-1 and pre_y <= ex_y+1
	elif a%8 == 3:
		return pre_y >= ex_y-2 and pre_y <= ex_y
	elif a%8 == 4: 
		return pre_y >= ex_y and pre_y <= ex_y+2
	elif a%8 == 5:
		return pre_y >= ex_y-2 and pre_y <= ex_y+2
	elif a%8 == 6: 
		return pre_y >= ex_y-2 and pre_y <= ex_y+1
	elif a%8 == 7:
		return pre_y >= ex_y-1 and pre_y <= ex_y+2
	else :
		return 

def loss(y, rang):
	if y<= rang[1] and y >= rang[0]:
		return 0
	elif y > rang[1]:
		return y - rang[1]
	elif y < rang[0] : 
		return  - y + rang[0]
	return 0


def test(data, W, map_range):
	accuracy1=0
	accuracy2=0
	MAE=0
	for d in data:
		expected_label = classLabel(d[-1],map_range)
		x = np.array(d[:-1])
		observed_label = classLabel(x.dot(W), map_range)

		MAE += loss(x.dot(W), map_range[expected_label])

		if partial1(observed_label, expected_label):
			accuracy1+=1

		if partial2(observed_label, expected_label):
			accuracy2+=1

	return float(accuracy1)/len(data) , float(accuracy2)/len(data), float(MAE)/len(data)


if __name__ == '__main__':
	C=10
	compute(sys.argv[1])