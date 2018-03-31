
from keras.models import *
from keras.layers import *
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import SGD

import cv2
import progressbar
import csv
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from sklearn.datasets import load_breast_cancer
from random import shuffle
from platypus import *
import keras
import os
import numpy as np
from scipy.misc import imread, imresize
from sklearn.model_selection import train_test_split
from glob import glob

print(sys.version)


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

# Salvar em arquivos
def writeCSV(nameFile, row):
    fileCSV = csv.writer(open(nameFile, "a"))
    fileCSV.writerow(row)



# Script
class script(Problem):
    def __init__(self, name, base):
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
        self.base = base
        self.lencategories=len(cancer.target_names)

        self.batch_size = 200  # de quanto em quanto vai caminhar
        self.epochs = 50  # repeticoes
        self.solucoes = {}  # dicionario de solucoes
        self.id = 0


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
        
        # 3) Separa aleatoriamente em treinamento (60%), validação (20%) e teste (20%)
        x_train, y_train, x_test, y_test= train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66) 
        X_train, Y_train, X_test, Y_test= configBase(x_train, y_train, x_test, y_test, img_rows, img_cols)


        print("Criando o modelo")

        

        modelPadrao = self.DefaulttDNN(input_shape=self.X_train.shape[1:],
                                       lencategories=self.lencategories)

        print(modelPadrao.summary())

        model = self.ModifieddDNN(numCamadasModf=solution.variables[7], model=modelPadrao, solution=solution,
                                  lencategories=self.lencategories, input_shape=self.X_train.shape[1:])

        print(model.summary())

        print("Compilando")
        learning_rate = 0.1
        decay_rate = learning_rate / self.epochs
        momentum = 0.8
        sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

        print("Iniciando treinamento")
        earlyStopping = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10, verbose=1, mode='auto')

        history = model.fit(X_train, Y_train, batch_size=self.batch_size,
                            epochs=self.epochs,
                            callbacks=[earlyStopping],
                            validation_split=0.33,
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
    #sklearn.datasets.load_breast_cancer -> Wisconsin Diagnostic Breast Cancer https://github.com/patrickmlong/Breast-Cancer-Wisconsin-Diagnostic-DataSet/blob/master/Breast%20Cancer%20Wisconsin%20(Diagnostic)%20DataSet_in_progress.ipynb
#https://github.com/Elhamkesh/Breast-Cancer-Scikitlearn/blob/master/CancerML.ipynb
    cancer=load_breast_cancer()
    
    #No longer exist -> IRMA
    #Archive -> ISIC
    #Archive -> Wisconsis Breast Cancer


    

    # tamanho da imagem
    
    print(len(cancer.feature_names))
    print("Configurando Problem")
    problem2 = script(name='Problem', base= cancer)  

    print("Configurando Otimizador")
    optimizer = SMPSO(problem2,
                       swarm_size=5,
                       leader_size=5,
                       generator=RandomGenerator(),
                       mutation_probability=0.1,
                       mutation_perturbation=0.5,
                       max_iterations=5,
                       mutate=None)

    print("Rodando o codigo")
    num_repet = 1
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

    print("Salvando as médias de dados...")
    rowAcc = soma_fm / num_repet
    rowFm = soma_fm / num_repet
    writeCSV('MediaACC.csv', rowAcc)
    writeCSV('MediaFM.csv', rowFm)
