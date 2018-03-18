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
import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import *
from keras.layers import *
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import os
import progressbar
import sys
from platypus import NSGAII, Problem, Real
from sklearn.model_selection import train_test_split
import csv
from random import *
from random import shuffle
from platypus import *
from keras.datasets import cifar10
import keras
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from scipy.misc import imread,imresize
from sklearn.model_selection import train_test_split
from glob import glob


configs={} #dicionario de resultados config  "results.update({'config1': 'resultado'})
melhoracc=None

A=[range(1,10), range(0,2), range(1,3), range(1,125), #Conv2D remover
  range(1,10), range(0,2), range(1,3), range(1,125), #Conv2D remover 
  range(2,4), range(1,3), #MaxPooling2D
  range(1,10), range(0,2), range(1,3), range(1,125), #Conv2D
  range(1,3), #Dense
  range(1,7)] #Numero de camadas Modificadas


def load_notmnist(path='./notMNIST_small',letters='ABCDEFGHIJ',
                  img_shape=(28,28),test_size=0.25,one_hot=False):
    
    # download data if it's missing. If you have any problems, go to the urls and load it manually.
    if not os.path.exists(path):
        print("Downloading data...")
        assert os.system('curl http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz > notMNIST_small.tar.gz') == 0
        print("Extracting ...")
        assert os.system('tar -zxvf notMNIST_small.tar.gz > untar_notmnist.log') == 0
    
    data,labels = [],[]
    print("Parsing...")
    for img_path in glob(os.path.join(path,'*/*')):
        class_i = img_path.split('/')[-2]
        if class_i not in letters: continue
        try:
            data.append(imresize(imread(img_path), img_shape))
            labels.append(class_i,)
        except:
            print("found broken img: %s [it's ok if <10 images are broken]" % img_path)
        
    data = np.stack(data)[:,None].astype('float32')
    data = (data - np.mean(data)) / np.std(data)

    #convert classes to ints
    letter_to_i = {l:i for i,l in enumerate(letters)}
    labels = np.array(list(map(letter_to_i.get, labels)))
    
    if one_hot:
        labels = (np.arange(np.max(labels) + 1)[None,:] == labels[:, None]).astype('float32')
    
    #split into train/test
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[3], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[3], 1)
    
    print("Done")
    return X_train, y_train, X_test, y_test


def configBase(x_train, y_train, x_test, y_test, img_rows, img_cols):
      tamanho=x_train.shape
      print(tamanho)
      if(len(tamanho)==4):
            # convert class vectors to binary class matrices
            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)
            
      elif(len(tamanho)==3):
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)
            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            x_train /= 255
            x_test /= 255
            
            # convert class vectors to binary class matrices
            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)
      else:
            print("Tamanho estranho. Por favor coloque bases com shape de tamanha 3 ou 4")
            print("Retornando None")
            return None      
      return x_train, y_train, x_test, y_test

#Salvar em arquivos
def writeCSV(nameFile, row):
      fileCSV = csv.writer(open(nameFile, "a"))
      fileCSV.writerow(row)

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

print("Configuração da Base")
# input image dimensions
img_rows, img_cols = 28, 28 #configuração manual a respeito da base que sera carregada (tamanho de cada imagem)

#output dimension
num_classes=10 #configuração manual a respeito da base que sera carregada (quantas classes tem a base)

print("Carregando Base")
#Load DB
x_train, y_train, x_test, y_test=load_notmnist() #Caso queira carregar bases que nao sao do keras o processo eh o mesmo apenas obtenha o dados de treino
                                                            # e teste e envie para a configuração de base. A bases de Gestos do prof sergio ja ta configurada int nao
                                                            #precisa ir pra proxima fase. So ir direto pro problema
      
      
print("Configurando Base")
x_train, y_train, x_test, y_test= configBase(x_train, y_train, x_test, y_test, img_rows, img_cols)




for i in range(10):
    print(i)
    model= RandomDNN(i,x_train.shape[1:])
    md= ModifiedDNN(3, model, x_train.shape[1:])
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
