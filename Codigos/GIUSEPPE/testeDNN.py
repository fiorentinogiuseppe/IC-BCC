from keras.models import *
from keras.layers import *
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import progressbar
import sys
from platypus import NSGAII, Problem, Real
from sklearn.model_selection import train_test_split
import csv
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score


from random import shuffle

from platypus import *

print(sys.version)

j=3 #ultimo elemento do vetor
model = Sequential()
md=Sequential()

model.add(Conv2D(filters=3, #se filters 0 da erro
                        kernel_size=3,
                        input_shape=(128, 128, 3)))#era 3 ))
model.add(Conv2D(filters=3, #se filters 0 da erro
                        kernel_size=3))#era 3 ))
model.add(MaxPooling2D())#era 1
model.add(Conv2D(filters=3, #se filters 0 da erro
                        kernel_size=3))#era 3 ))
#model.add(Flatten())
model.add(Dense(units=5#numero de classes 
                        ,activation='sigmoid'))

print(model.summary())

for i in range(1,6,1):
	if(i<j):
		md.add(model.get_layer(None,i))
	else:
		if(i==1): md.add(Conv2D(filters=2, #se filters 0 da erro
                        kernel_size=2,
                        input_shape=(128, 128, 3)))#era 3 ))
		elif(i==2 or i==4):
			md.add(Conv2D(filters=2, #se filters 0 da erro
                        kernel_size=2))#era 3 ))
		elif(i==3):
			md.add(MaxPooling2D(pool_size=(3, 3)))#era 1
		elif(i==5):
			md.add(Dense(units=4#numero de classes 
                        ,activation='sigmoid'))



print(md.summary())

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])