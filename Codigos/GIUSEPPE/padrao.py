import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import MProblem as MP
from platypus import NSGAII, Problem, Real,SMPSO
import pandas as pd
from sklearn.preprocessing import LabelEncoder
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
import copy
from random import *
from random import shuffle
from platypus import *
from keras.datasets import cifar10
import keras
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import os
import numpy as np
from scipy.misc import imread,imresize
from sklearn.model_selection import train_test_split
from glob import glob

# Retorna quantidade de imagens em diretorio
def getNumSamples(src):
    sum = 0
    for cl in os.listdir(src):
        class_dir = os.path.join(src, cl)
        files = os.listdir(class_dir)
        l = len(files)
        sum += l

    return sum, len(os.listdir(src))

# Carrega base de dados e converte em array numpy
def array_from_dir(data_dir, nb_samples, nb_classes, width, height):
    images = np.zeros((nb_samples, width, height, 1))
    y = np.zeros((nb_samples, nb_classes))

    bar = progressbar.ProgressBar(max_value=len(os.listdir(data_dir)))

    cont = 0
    cl_index = 0
    for cl in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, cl)
        files = os.listdir(class_dir)
        shuffle(files)
        files_per_class = 0
        for file in files:
            files_per_class += 1

            # lê em escala de cinza
            img = cv2.imread(os.path.join(class_dir, file), flags=0)

            # muda escala
            img = cv2.resize(img, (width, height))

            # deixa entre 0 e 1
            img = np.array(img) / 255.0

            # coloca no array
            images[cont, :, :, 0] = img

            # guarda a classe
            y[cont] = to_categorical(cl_index, nb_classes)

            cont += 1
        cl_index += 1
        bar.update(cl_index)

    # retorna array com imagens em classes
    return images, y


# divide base aleatoriamente em treinamento, teste e validação
def split_dataset_random(b_x, b_y, p_train, p_val):
    total_samples = b_x.shape[0]
    train_size = int(total_samples * p_train)
    valid_size = int(total_samples * p_val)
    test_size = total_samples - train_size - valid_size

    print()
    print('train set:', train_size, ' images')
    print('validation set:', valid_size, ' images')
    print('test set:', test_size, ' images')

    x_tr = np.zeros((train_size, b_x[0].shape[0], b_x[0].shape[1], b_x[0].shape[2]))
    y_tr = np.zeros((train_size, b_y[0].shape[0]))

    x_val = np.zeros((valid_size, b_x[0].shape[0], b_x[0].shape[1], b_x[0].shape[2]))
    y_val = np.zeros((valid_size, b_y[0].shape[0]))

    x_te = np.zeros((test_size, b_x[0].shape[0], b_x[0].shape[1], b_x[0].shape[2]))
    y_te = np.zeros((test_size, b_y[0].shape[0]))

    from random import shuffle
    index = list(range(total_samples))
    shuffle(index)

    for i in range(total_samples):
        if i < train_size:
            x_tr[i] = b_x[index[i]]
            y_tr[i] = b_y[index[i]]
        elif i < (train_size + valid_size):
            x_val[i - train_size] = b_x[index[i]]
            y_val[i - train_size] = b_y[index[i]]
        else:
            x_te[i - train_size - valid_size] = b_x[index[i]]
            y_te[i - train_size - valid_size] = b_y[index[i]]

    return x_tr, y_tr, x_te, y_te, x_val, y_val

print("Carregando Base")
    
#Base Gestos

# indica pasta da base de dados, uma pasta com imagens para cada classe
pasta_base = 'DB/Gestures/'

# tamanho do batch de treinamento
batch_size = 200

nb_db_samples, num_classes = getNumSamples(pasta_base)

print('Base tem ', nb_db_samples, ' imagens e ', num_classes, ' classes')

# tamanho da imagem
options_im_size = [16, 32, 48, 64]
im_sz=random.randint(0,3)
img_width, img_height = options_im_size[im_sz], options_im_size[im_sz]
print('Rescaling database to', img_width, 'x', img_height, ' pixels')
base_x, base_y = array_from_dir(data_dir=pasta_base, nb_samples=nb_db_samples,
                                        nb_classes=num_classes, width=img_width, height=img_height)

X_train, X_test, y_train, y_test = train_test_split(base_x,base_y,test_size=0.2,random_state=0) 
input_shape=X_train.shape[1:]
print("Criando o modelo")
model = Sequential()
model.add(Conv2D(filters=5, kernel_size=4,padding="same", input_shape=input_shape))
model.add(Conv2D(filters=5, kernel_size=4, padding="same"))
model.add(MaxPooling2D())
model.add(Conv2D(filters=5, kernel_size=4, padding="same"))
model.add(Flatten())
model.add(Dense(units=5))

print("Compilando")
learning_rate = 0.1
epochs=50
decay_rate = learning_rate / epochs
momentum = 0.8
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])#,'precision','recall', 'f1'])

print("Iniciando treinamento")
earlyStopping = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10, verbose=1, mode='auto')
history=model.fit(X_train,y_train,batch_size=200,
                        epochs=epochs,
                        callbacks=[earlyStopping],
                        validation_split=0.33,
                        shuffle=True,
                        verbose=2)

train_acc = history.history['acc'][-1]
val_acc = history.history['val_acc'][-1]
print("acuracia",val_acc)

