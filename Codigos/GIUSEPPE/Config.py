from platypus import Problem, Solution, EPSILON
from platypus import Real, Binary
from keras.models import *
from keras.layers import *
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import progressbar
import sys
from random import shuffle
import keras.backend as K


from platypus import *

class config(Problem):
    def __init__(self):
        # indica pasta da base de dados, uma pasta com imagens para cada classe
        self.pasta_base = 'DB/Gestures/'

        # tamanho do batch de treinamento
        self.batch_size = 200

        self.nb_db_samples, self.num_classes = self.getNumSamples(self.pasta_base)

        print('Base tem ', self.nb_db_samples, ' imagens e ', self.num_classes, ' classes')

    def getPastaBase(self):
        return self.pasta_base
    def getNbDbSamples(self):
        return self.nb_db_samples
    def getNumClasses(self):
        return self.num_classes
    def getBatchSize(self):
        return self.batch_size

	# Retorna quantidade de imagens em diretorio
    def getNumSamples(self, src):
        sum = 0
        for cl in os.listdir(src):
            class_dir = os.path.join(src, cl)
            files = os.listdir(class_dir)
            l = len(files)
            sum += l

        return sum, len(os.listdir(src))


    # Carrega base de dados e converte em array numpy
    def array_from_dir(self, data_dir, nb_samples, nb_classes, width, height):
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

                # le em escala de cinza
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


    # divide base aleatoriamente em treinamento, teste e validacao
    def split_dataset_random(self, b_x, b_y, p_train, p_val):
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

