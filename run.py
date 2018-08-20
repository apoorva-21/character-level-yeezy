import numpy as np
import pickle
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import load_model
trainData= []
trainLabels = []
INPUT_LENGTH = 10
listDataIdx = []
str = ""

def generateRap(model, inputStr, lengthToGenerate):
	global dictStr
	listInputIdx = []
	for c in list(inputStr):
		listInputIdx.append(dictStr[c]) #generate a list of all character indices in order of the lyrics
	for i in range(0, lengthToGenerate):
		npListInputIdx = np.array(listInputIdx)
		inputOneHot = to_categorical(npListInputIdx, num_classes = N_UNIQUE_CHARS)
		inputOneHot = np.array(inputOneHot)
		inputOneHot = np.reshape(inputOneHot,(1, inputOneHot.shape[0], inputOneHot.shape[1]))
		outputOneHot = model.predict(inputOneHot)
		outputIdx = np.argmax(outputOneHot)
		listInputIdx.append(outputIdx)
		listInputIdx = listInputIdx[1:]#update the input for the next iteration
		outputChar = ''
		for key, val in dictStr.items():
			if val == outputIdx:
				outputChar = key
				break
		inputStr += outputChar
	return inputStr


with open('./data/kanye_verses.txt', 'rb') as f:
	allData = f.read()
listStr = list(allData)
dictStr = dict()
for i , n in enumerate(set(listStr)):
	dictStr[n] = i
N_UNIQUE_CHARS = len(dictStr)

# with open('./data/dictPickle.pkl', 'wb') as f:
# 	pickle.dump(dictStr, f, pickle.HIGHEST_PROTOCOL)
# print "Dictionary Pickled!"

# print generateRap(load_model('model-iValue-5.h5'), 'yeezy taug', 20)
# exit()
for c in list(allData):
	listDataIdx.append(dictStr[c]) #generate a list of all character indices in order of the lyrics
for i in range(len(listDataIdx) - INPUT_LENGTH):
	#take every n characters as a training featureset, with label as n+1th char
	trainData.append(listDataIdx[i:i+INPUT_LENGTH])
	trainLabels.append(listDataIdx[i+INPUT_LENGTH]) #label is the n+1th character
trainData = np.array(trainData)
print trainData.shape
trainLabels = np.array(trainLabels)
trainLabels = np.reshape(trainLabels, (trainLabels.shape[0], 1))
#converting the data and inputs to one-hot vectors:
trainDataOneHot = to_categorical(trainData, num_classes = N_UNIQUE_CHARS)
trainLabelsOneHot = to_categorical(trainLabels, num_classes = N_UNIQUE_CHARS)
print trainLabelsOneHot.shape
print trainDataOneHot.shape
model = Sequential()
model.add(LSTM(75, input_shape = (trainDataOneHot.shape[1], trainDataOneHot.shape[2])))
model.add(Dense(N_UNIQUE_CHARS, activation = 'softmax'))
print model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
for i in range(1,6):
	model.fit(trainDataOneHot, trainLabelsOneHot, epochs = 10, verbose = 1, batch_size = 500)
	model.save('model-iValue-{}.h5'.format(i))
	print generateRap(model,'yeezy taug', 20)
