import random
import pandas as pd
import keras
from sklearn.preprocessing import LabelEncoder
from keras.models import *
from keras.layers import *
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import SGD
import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import mnist


configs={} #dicionario de resultados config  "results.update({'config1': 'resultado'})
melhoracc=None

A=[range(1,10), range(0,2), range(1,3), range(1,125), #Conv2D remover
  range(1,10), range(0,2), range(1,3), range(1,125), #Conv2D remover 
  range(2,4), range(1,3), #MaxPooling2D
  range(1,10), range(0,2), range(1,3), range(1,125), #Conv2D
  range(1,3), #Dense
  range(1,7)] #Numero de camadas Modificadas

def RandomDNN(gera,input_shape):
    

    model = Sequential()
    input1=random.randint(1,9)
    input2=random.randint(1,124)
    input3=random.randint(1,2)
    input4=random.randint(1,2)
    input5=random.randint(1,9)
    input6=random.randint(1,124)
    input7=random.randint(1,2)
    
    model.add(Conv2D(filters=input1, kernel_size=input2, strides=input3 ,padding="same", input_shape=input_shape))
    model.add(Conv2D(filters=input1, kernel_size=input2, strides=input3 ,padding="same"))

    pool=random.randint(2,3)
    
    model.add(MaxPooling2D(pool_size=(pool,pool), strides=input4))
    model.add(Conv2D(filters=input5, kernel_size=input6,strides=input7, padding="same"))
    model.add(Flatten())
    model.add(Dense(units=10, activation='softmax'))

    configs.update({gera: model})
    
    return model
    
def ModifiedDNN(numCamadasModf, model, input_shape):
    
    input1=random.randint(1,9)
    input2=random.randint(1,124)
    input3=random.randint(1,2)
    input4=random.randint(1,2)
    input5=random.randint(1,9)
    input6=random.randint(1,124)
    input7=random.randint(1,2)
    pool=random.randint(2,3)

    md=Sequential()
    for i in range(1,7,1):
        if(i<numCamadasModf or i==5): md.add(model.get_layer(None,i))
        else:
            if(i==1): md.add(Conv2D(filters=input1, kernel_size=input2, strides=input3, padding="same",input_shape=input_shape))
            if(i==4): md.add(Conv2D(filters=input4, kernel_size=input5, strides=input6, padding="same")) 
            elif(i==3): md.add(MaxPooling2D(pool_size=(pool,pool), strides=input7))
            elif(i==6): md.add(Dense(units=10,activation='softmax'))
    return md    


batch_size = 128
num_classes = 10
epochs = 12
# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


for i in range(10):
    print(i)
    model= RandomDNN(i,input_shape)
    md= ModifiedDNN(3, model, input_shape)
    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    earlyStopping = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10, verbose=1, mode='auto')
    history=model.fit(x_test,y_test,batch_size=50,
                        epochs=1,
                        callbacks=[earlyStopping],
                        validation_split=0.33,
                        shuffle=True,
                        verbose=2)

    train_acc = history.history['acc'][-1]
    val_acc = history.history['val_acc'][-1]
    if(melhoracc != None):
        if(val_acc<melhoracc):
            melhoracc=val_acc
    else: melhoracc=val_acc
    
print("Melhor acuracia", melhoracc)
print("Dicionarios de config", config)
