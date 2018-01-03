from __future__ import print_function

from platypus import Problem, Solution, EPSILON
from platypus import Real, Binary

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from keras.models import *
from keras.layers import *
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD,Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report,confusion_matrix
import Config as configuracao
import matplotlib.pyplot as plt




from platypus import *

class SProblem(Problem):
        def __init__(self, name, base_x, base_y):
                #nao vai de 0 a 9
                encoding = [Integer(1,9), Integer(0,1), Integer(1,2), Integer(1,124),
                            Integer(2,3), Integer(1,2), 
                            Integer(1,2)]
                variables = len(encoding)
                objectives = 1
                super(SProblem, self).__init__(variables, objectives)
                self.types[:] = encoding
                self.class_name = name
                self.cnfg=configuracao.config()
                self.base_x=base_x
                self.base_y=base_y

                self.batch_size = 28#de quanto em quanto vai caminhar
                self.num_classes = 3
                self.epochs = 50#repeticoes
                self.solucoes={}#dicionario de solucoes
                self.id = 0
        def getSolucoes(self):
                return self.solucoes
        def getSolucaoFinal(self,info):
                return self.solucoes.get(info)
        def evaluate(self, solution):
                print(solution)
                print("Rescaling database")
                X_train,X_test, y_train,y_test = train_test_split(self.base_x,self.base_y,test_size=0.2,random_state=0) 
                
                print("Train Shape",X_train.shape,"\nTeste Shap:",X_test.shape)
                #(119, 3, 1) - 119- batch_size
                #            - 3- input_dim
                #            - 1-
                print("Criando o modelo")
                if(solution.variables[1]==0):
                        pad="valid"
                else:
                        pad="same"
                #erro se usar valid 
                model = Sequential()
                print("fase 1")
                model.add(Conv2D(filters=solution.variables[0], #se filters 0 da erro
                        kernel_size=solution.variables[3],#era 3
                        strides=solution.variables[2], #era 1
                        padding="same",
                        input_shape=X_train.shape[1:]
                        ))
                print("fase 2")
                print("fase 2")
                print(model.summary())
                model.add(MaxPooling2D(pool_size=(solution.variables[4],solution.variables[4]), 
                        strides=solution.variables[5]))#era 1
                print("fase 3")
                model.add(Flatten())
                model.add(Dense(units=5#numero de classes 
                        ,activation='sigmoid'))
                print("fase 4")
                print(model.summary())
                print("Compilando")
                learning_rate = 0.1
                decay_rate = learning_rate / self.epochs
                momentum = 0.8
                sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
                model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

                print("\nSalvando o diagrama do modelo")

                print("Iniciando treinamento")
                earlyStopping = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10, verbose=1, mode='auto')
                history=model.fit(X_train,y_train,batch_size=self.batch_size,
                        epochs=self.epochs,
                        callbacks=[earlyStopping],
                        validation_split=0.33,
                        shuffle=True,
                        verbose=2)
                '''
                y_pred = model.predict(X_test)

                y_test_class = np.argmax(y_test,axis=1)
                y_pred_class = np.argmax(y_pred,axis=1)

                print(classification_report(y_test_class,y_pred_class))
                print(confusion_matrix(y_test_class,y_pred_class))
                
                score = model.evaluate(X_test, y_test, verbose=0)
                print('Test loss:', score[0])
                print('Test accuracy:', score[1])
                solution.objectives=score
                print("solutionnnn: ",solution)
                '''
                plt.clf()
                plt.plot(history.history['acc'])
                plt.plot(history.history['val_acc'])
                plt.title('model accuracy')
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.legend(['train', 'validation'], loc='upper left')
                #plt.savefig(str(self.id)+'_accuracy')
                plt.clf()

                # summarize history for loss
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'validation'], loc='upper left')
                #plt.savefig(str(self.id)+'_loss')

                train_acc = history.history['acc'][-1]
                val_acc = history.history['val_acc'][-1]

                train_epochs = len(history.history['acc'])

                start = time.time()

                # Testa
                test_scores = model.evaluate(X_test, y_test, verbose=0)

                end = time.time()
                test_ms = round((end - start) * 1000, 1)
                print("Test time: ", test_ms, " ms")

                test_acc = test_scores[1]
                print('Test accuracy:', test_acc)

                from keras import backend as K
                if K.backend() == 'tensorflow':
                    K.clear_session()

                self.id += 1

                # retorna os objetivos (acuracia e tempo)
                #solution.objectives[:] = [-val_acc, test_ms]#teim um sinal negativo aq
                solution.objectives[:] = [-val_acc]#como o objectives la em cima ta 1 nao precisa do tempo
                
                #Salvando as variaveis junto com os objectives num dicionario, 
                #pois no main as variaveis estavam
                #binarias
                variaveis=[]
                print(solution.objectives[0])
                for i in solution.variables:
                        variaveis.append(i)
                self.solucoes.update({solution.objectives[0]:variaveis})
#[1, 0, 2, 64, 3, 2, 1]
#1 - filter
#same #problema com o valid -padding
#2- stride
#64- kernel size
#(3,3)- poolsize
#2- stride
#1-#problema nao sei onde usar
