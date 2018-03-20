
from keras.models import *
from keras.layers import *
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import SGD

import cv2
import progressbar
import csv
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from random import shuffle
from platypus import *
import keras
import os
import numpy as np
from scipy.misc import imread, imresize
from sklearn.model_selection import train_test_split
from glob import glob

print(sys.version)


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


# Salvar em arquivos
def writeCSV(nameFile, row):
    fileCSV = csv.writer(open(nameFile, "a"))
    fileCSV.writerow(row)


# Retorna quantidade de imagens em diretorio
def getNumSamples(src):
    sum = 0
    for cl in os.listdir(src):
        class_dir = os.path.join(src, cl)
        files = os.listdir(class_dir)
        l = len(files)
        sum += l

    return sum, len(os.listdir(src))


# Script
class script(Problem):
    def __init__(self, name, base_x, base_y, lencategories):
        # 0	        1
        encoding = [Integer(2, 3), Integer(1, 2),
                    # 2	        3	    4		      5
                    Integer(1, 9), Integer(0, 1), Integer(1, 2), Integer(1, 124),
                    # 6
                    Integer(1, 2),
                    # 7
                    Integer(3, 6)]

        variables = len(encoding)
        objectives = 2
        super(script, self).__init__(variables, objectives)
        self.types[:] = encoding
        self.class_name = name
        self.base_x = base_x
        self.base_y = base_y
        self.nb_db_samples, self.num_classes = getNumSamples(pasta_base)

        self.batch_size = 200  # de quanto em quanto vai caminhar
        self.epochs = 50  # repeticoes
        self.solucoes = {}  # dicionario de solucoes
        self.id = 0
        self.lencategories = lencategories

    def DefaulttDNN(self, input_shape, lencategories):
        model = Sequential()
        model.add(Conv2D(filters=5, kernel_size=4, padding="same", input_shape=input_shape))
        model.add(Conv2D(filters=5, kernel_size=4, padding="same"))
        model.add(MaxPooling2D(pool_size=[2, 2], strides=2))
        model.add(Conv2D(filters=5, kernel_size=4, padding="same"))
        model.add(Flatten())
        model.add(Dense(units=5, activation='softmax'))
        return model

    def ModifieddDNN(self, numCamadasModf, model, solution, lencategories, input_shape):
        md = Sequential()
        for i in range(1, 7, 1):
            if (i < numCamadasModf or i == 5 or i == 1 or i == 2):
                md.add(model.get_layer(None, i))
            else:
                if (i == 3):
                    md.add(MaxPooling2D(pool_size=(solution.variables[0], solution.variables[0]),
                                        strides=solution.variables[1]))
                elif (i == 4):
                    md.add(Conv2D(filters=solution.variables[2], kernel_size=solution.variables[5],
                                  strides=solution.variables[4], padding="same"))
                elif (i == 6):
                    md.add(Dense(units=self.lencategories, activation='softmax'))
        return md

    def getSolucaoFinal(self, info):
        return self.solucoes.get(info)

    def evaluate(self, solution):

        options_im_size = [16, 32, 48, 64]
        im_sz = random.randint(0, 3)
        img_width, img_height = options_im_size[im_sz], options_im_size[im_sz]

        # 2) Carrega a base de dados
        print('Rescaling database to', img_width, 'x', img_height, ' pixels')
        base_x, base_y = array_from_dir(data_dir=pasta_base, nb_samples=nb_db_samples,
                                        nb_classes=num_classes, width=img_width, height=img_height)

        # 3) Separa aleatoriamente em treinamento (60%), validação (20%) e teste (20%)
        X_train, y_train, X_test, y_test, X_val, Y_val = split_dataset_random(base_x, base_y, 0.6, 0.2)



        print("Criando o modelo")

        self.X_train = X_train

        modelPadrao = self.DefaulttDNN(input_shape=self.X_train.shape[1:],
                                       lencategories=5)

        print(modelPadrao.summary())

        model = self.ModifieddDNN(numCamadasModf=solution.variables[7], model=modelPadrao, solution=solution,
                                  lencategories=5, input_shape=self.X_train.shape[1:])

        print(model.summary())

        print("Compilando")
        learning_rate = 0.1
        decay_rate = learning_rate / self.epochs
        momentum = 0.8
        sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

        print("Iniciando treinamento")
        earlyStopping = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10, verbose=1, mode='auto')
        print(X_train.shape)
        history = model.fit(X_train, y_train, batch_size=self.batch_size,
                            epochs=self.epochs,
                            callbacks=[earlyStopping],
                            validation_data=[X_val, Y_val],
                            shuffle=True,
                            verbose=2)

        train_acc = history.history['acc'][-1]
        val_acc = history.history['val_acc'][-1]


        # metrics################################################################
        print('\n acc', val_acc)
        predictions = model.predict(X_test, batch_size=batch_size)
        print(classification_report(y_test.argmax(axis=1),
                                    predictions.argmax(axis=1)))
        report_lr = precision_recall_fscore_support(y_test.argmax(axis=1),
                                                    predictions.argmax(axis=1),
                                                    average='macro')

        print("\nprecision = %0.2f, recall = %0.2f, F1 = %0.2f, accuracy = %0.2f\n" % \
              (report_lr[0], report_lr[1], report_lr[2], accuracy_score(y_test.argmax(axis=1),
                                                                        predictions.argmax(axis=1))))
        train_epochs = len(history.history['acc'])
        ##########################################################################

        from keras import backend as K
        if K.backend() == 'tensorflow':
            K.clear_session()

        self.id += 1

        solution.objectives[:] = [-val_acc, - report_lr[2]]

        variaveis = []
        print(solution.objectives)
        for i in solution.variables:
            variaveis.append(i)
        self.solucoes.update({solution.objectives: variaveis})

        # Salvar dados
        print("Salvando dados...")
        row = solution.objectives
        writeCSV('files.csv', row)


# MAIN

if __name__ == '__main__':
    print("Carregando Base")

    # Base Gestos

    # indica pasta da base de dados, uma pasta com imagens para cada classe
    pasta_base = 'Gestures/'

    # tamanho do batch de treinamento
    batch_size = 200

    nb_db_samples, num_classes = getNumSamples(pasta_base)

    print('Base tem ', nb_db_samples, ' imagens e ', num_classes, ' classes')

    # tamanho da imagem
    options_im_size = [16, 32, 48, 64]
    im_sz = random.randint(0, 3)
    img_width, img_height = options_im_size[im_sz], options_im_size[im_sz]
    print('Rescaling database to', img_width, 'x', img_height, ' pixels')
    base_x, base_y = array_from_dir(data_dir=pasta_base, nb_samples=nb_db_samples,
                                    nb_classes=num_classes, width=img_width, height=img_height)

    print("Configurando Problem")
    problem2 = script(name='Problem', base_x=base_x, base_y=base_y,
                      lencategories=num_classes)  # Qualquer outra base usar esse

    print("Configurando Otimizador")
    optimizer = SMPSO(problem2,
                       swarm_size=30,
                       leader_size=5,
                       generator=RandomGenerator(),
                       mutation_probability=0.1,
                       mutation_perturbation=0.5,
                       max_iterations=100,
                       mutate=None)

    print("Rodando o codigo")
    num_repet = 10
    repeticao = 0

    soma = 0
    soma_fm = 0
    soma_t = 0

    for i in range(num_repet):
        # executa por uma geracao
        repeticao += 1
        print("::::Repeticao: ", repeticao, "::::")
        start = time.time()
        optimizer.run(1)
        end = time.time()
        print(">>>>>>>>>>>>>RESULT<<<<<<<<<<<<<<<\n")
        val = (optimizer.result)[0].objectives
        print("Objective", -val[0], "fm ", -val[1])



        tmp = round((end - start), 2)

        soma_t = soma_t + tmp
        soma = soma + (-val[0])
        soma_fm = soma_fm + (-val[1])

    print("Media acuracia ", soma / num_repet, "i ", num_repet)
    print("Media tmpo ", soma_t / num_repet, "i ", num_repet)
    print("Media fm ", soma_fm / num_repet, "i ", num_repet)
