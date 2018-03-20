
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
from keras.datasets import mnist
from scipy.misc import imread, imresize
from sklearn.model_selection import train_test_split
from glob import glob
from sklearn.metrics import  precision_recall_fscore_support

configs = {}  # dicionario de resultados config  "results.update({'config1': 'resultado'})
melhoracc = None

A = [range(1, 10), range(0, 2), range(1, 3), range(1, 125),  # Conv2D remover
     range(1, 10), range(0, 2), range(1, 3), range(1, 125),  # Conv2D remover
     range(2, 4), range(1, 3),  # MaxPooling2D
     range(1, 10), range(0, 2), range(1, 3), range(1, 125),  # Conv2D
     range(1, 3),  # Dense
     range(1, 7)]  # Numero de camadas Modificadas



# Salvar em arquivos
def writeCSV(nameFile, row):
    fileCSV = csv.writer(open(nameFile, "a"))
    fileCSV.writerow(row)


def DefaultDNN(gera, input_shape):
    model = Sequential()
    model.add(Conv2D(filters=5, kernel_size=4, padding="same", input_shape=input_shape))
    model.add(Conv2D(filters=5, kernel_size=4, padding="same"))
    model.add(MaxPooling2D(pool_size=[2, 2], strides=2))
    model.add(Conv2D(filters=5, kernel_size=4, padding="same"))
    model.add(Flatten())
    model.add(Dense(units=5, activation='softmax'))

    configs.update({gera: model})

    return model


def ModifiedDNN(numCamadasModf, model, input_shape):

    md = Sequential()
    for i in range(1, 7, 1):
        if (i < numCamadasModf or i == 5):
            md.add(model.get_layer(None, i))
        else:
            if (i == 4):
                md.add(Conv2D(filters=l, kernel_size=o, strides=n, padding="same"))
            elif (i == 3):
                md.add(MaxPooling2D(pool_size=(j, j), strides=k))
            elif (i == 6):
                md.add(Dense(units=5, activation='softmax'))
    return md


batch_size = 128
num_classes = 10
epochs = 12

# Retorna quantidade de imagens em diretorio
def getNumSamples(src):
    sum = 0
    for cl in os.listdir(src):
        class_dir = os.path.join(src, cl)
        files = os.listdir(class_dir)
        l = len(files)
        sum += l

    return sum, len(os.listdir(src))

import cv2
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


# indica pasta da base de dados, uma pasta com imagens para cada classe
pasta_base = 'Gestures/'

# tamanho do batch de treinamento
batch_size = 200

nb_db_samples, num_classes = getNumSamples(pasta_base)

print('Base tem ', nb_db_samples, ' imagens e ', num_classes, ' classes')

# tamanho da imagem
        ##im_sz = sum(2 ** i * b for i, b in enumerate(bina[random.randint(0,4)], bina[random.randint(0,4)]))
options_im_size = [16, 32, 48, 64]
im_sz=random.randint(0,3)
img_width, img_height = options_im_size[im_sz], options_im_size[im_sz]

        # 2) Carrega a base de dados
print('Rescaling database to', img_width, 'x', img_height, ' pixels')
base_x, base_y = array_from_dir(data_dir=pasta_base, nb_samples=nb_db_samples,
                                        nb_classes=num_classes, width=img_width, height=img_height)

        # 3) Separa aleatoriamente em treinamento (60%), validação (20%) e teste (20%)
x_train, y_train, x_test, y_test, X_val, Y_val = split_dataset_random(base_x, base_y, 0.6, 0.2)

# inicia o algoritmo

val=0
fm = 0
mdia=0
start = time.time()
for j in range(2, 3):
    for k in range(1, 2):
        for l in range(1, 9):
            for m in range(1, 2):
                for n in range(1, 2):
                    for o in range(1, 1024):
                        for p in range(1, 2):
                            for s in range(3, 6):
                                print("Compilando")
                                model = DefaultDNN(1, x_train.shape[1:])

                                md = ModifiedDNN(s, model, x_train.shape[1:])

                                md.compile(optimizer='rmsprop',
                                              loss='categorical_crossentropy',
                                              metrics=['accuracy'])
                                earlyStopping = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10,
                                                              verbose=1, mode='auto')
                                history = md.fit(x_test, y_test, batch_size=200,
                                                    epochs=50,
                                                    callbacks=[earlyStopping],
                                                    validation_data=[X_val, Y_val],
                                                    shuffle=True,
                                                    verbose=2)

                                train_acc = history.history['acc'][-1]
                                val_acc = history.history['val_acc'][-1]

                                predictions = model.predict(x_test, batch_size=batch_size)

                                report_lr = precision_recall_fscore_support(y_test.argmax(axis=1),
                                                                            predictions.argmax(axis=1),
                                                                            average='macro')

                                #print("val ", (val_acc+ report_lr[2])/2)
                                if(mdia < (val_acc+ report_lr[2])/2):
                                    val= val_acc
                                    fm = report_lr[2]
                                    mdia = (val_acc + report_lr[2])/2



end = time.time()
tmp = round((end - start), 2)
print("mlho val : ", val)
print("mlho fm : ", fm)

print("Tmp : ", tmp)


