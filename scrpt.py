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
from scipy.io import arff
import pandas as pd

print(sys.version)


def load_base(name):
        data= arff.loadarff(name)    
        df = pd.DataFrame(data[0])
        df=df.as_matrix(columns=df.columns[0:])
        base_y=[]
        base_x=[]
        for i in df:
            for j in i:
                if(j==b'1' or j==b'2' or j==b'3'):
                            base_y.append(j)
                else:
                        base_x.append(j)
        base_x=np.reshape(base_x,(-1,13))
        base_y=np.reshape(base_y,(-1,1))
        return base_x, base_y	

def configBase(x_train, y_train, x_test, y_test):
     
            x_train = np.expand_dims(x_train, axis=2)
            x_test = np.expand_dims(x_test, axis=2)
            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            x_train /= 255
            x_test /= 255
            
            y_train = keras.utils.to_categorical(y_train) #talvez o erro esteja aqui e eu nao to sabendo interpretar
            y_test = keras.utils.to_categorical(y_test)   #talvez o erro esteja aqui e eu nao to sabendo interpretar
            writeCSV('train.csv', y_train)
            writeCSV('teste.csv', y_test)
            return x_train, y_train, x_test, y_test

# Salvar em arquivos
def writeCSV(nameFile, row):
        outfile = open(nameFile, "a")
        fileCSV = csv.writer(outfile)
        fileCSV.writerow(row)
        outfile.close()

# Script
class script(Problem):
    def __init__(self, name, base_x, base_y,lencategories):
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
        self.lencategories=lencategories

        self.batch_size = 200  # de quanto em quanto vai caminhar
        self.epochs = 50  # repeticoes
        self.solucoes = {}  # dicionario de solucoes
        self.id = 0


    def DefaulttDNN(self, input_shape, lencategories):
        print(input_shape)
        model = Sequential()
        model.add(Conv1D(filters=5, kernel_size=4, padding="same", input_shape=input_shape))
        model.add(Conv1D(filters=5, kernel_size=4, padding="same"))
        model.add(MaxPooling1D(pool_size= 2, strides=2))
        model.add(Conv1D(filters=5, kernel_size=4, padding="same"))
        model.add(Flatten())
        model.add(Dense(units=self.lencategories, activation='softmax'))
        return model

    def ModifieddDNN(self, numCamadasModf, model, solution, lencategories, input_shape):
        md = Sequential()
        for i in range(1, 7, 1):
            if (i < numCamadasModf or i == 5 or i == 1 or i == 2 ):
                md.add(model.get_layer(None, i))
            else:
                if (i == 3):
                    md.add(MaxPooling1D(pool_size=solution.variables[0],
                                        strides=solution.variables[1]))
                elif (i == 4):
                    md.add(Conv1D(filters=solution.variables[2], kernel_size=solution.variables[5],
                                  strides=solution.variables[4], padding="same"))
                elif (i == 6):
                    md.add(Dense(units=self.lencategories, activation='softmax'))
        return md

    def getSolucaoFinal(self, info):
        return self.solucoes.get(info)

    def evaluate(self, solution):
        
        # 3) Separa aleatoriamente em treinamento (60%), validação (20%) e teste (20%)
        X_train, X_test, Y_train, Y_test = train_test_split(self.base_x,self.base_y,test_size=0.2,random_state=0)
        X_train, Y_train, X_test, Y_test = configBase(X_train, Y_train, X_test, Y_test)
        
        print("Criando o modelo")

        #https://stackoverflow.com/questions/43396572/dimension-of-shape-in-conv1d/43399308#43399308

        modelPadrao = self.DefaulttDNN(input_shape=X_train.shape[1:],
                                       lencategories=self.lencategories)

        print(modelPadrao.summary())

        model = self.ModifieddDNN(numCamadasModf=solution.variables[7], model=modelPadrao, solution=solution,
                                  lencategories=self.lencategories, input_shape=X_train.shape[1:])

        print(model.summary())

        print("Compilando")
        learning_rate = 0.1
        decay_rate = learning_rate / self.epochs
        momentum = 0.8
        sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        print("Iniciando treinamento")
        print(X_train.shape)
        print(Y_train.shape)
        earlyStopping = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10, verbose=1, mode='auto')

        history = model.fit(x=X_train, y=Y_train, batch_size=self.batch_size,
                            epochs=self.epochs,
                            callbacks=[earlyStopping],
                            validation_split=0.33,
                            shuffle=True,
                            verbose=2)

        train_acc = history.history['acc'][-1]
        val_acc = history.history['val_acc'][-1]


        # metrics################################################################
        print('\n acc', val_acc)
        predictions = model.predict(X_test, batch_size=self.batch_size)
        print(classification_report(Y_test.argmax(axis=1),
                                    predictions.argmax(axis=1)))
        report_lr = precision_recall_fscore_support(Y_test.argmax(axis=1),
                                                    predictions.argmax(axis=1),
                                                    average='macro')

        print("\nprecision = %0.2f, recall = %0.2f, F1 = %0.2f, accuracy = %0.2f\n" % \
              (report_lr[0], report_lr[1], report_lr[2], accuracy_score(Y_test.argmax(axis=1),
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
    base1="mamografia_Haralick.arff"
    base2="mamografia_LBP.arff"
    base3="mamografia_Seg-Zernike.arff"
    base4="melanoma_Haralick.arff"
    base5="melanoma_LBP.arff"
    base_x,base_y= load_base(base2)

    print("Configurando Problem")

    problem2 = script(name='Problem', base_x=base_x, base_y=base_y, lencategories=4) 
    
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
    
    rowAcc = []
    rowAcc.append(soma / num_repet)
    rowFm = []
    rowFm.append(soma_fm / num_repet)
    print(rowAcc)
    print(rowFm)
    writeCSV('MediaACC.csv', rowAcc)
    writeCSV('MediaFM.csv', rowFm)
