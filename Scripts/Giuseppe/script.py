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

print(sys.version)

#Salvar em arquivos
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

#Script
class script(Problem):
        def __init__(self, name, base_x, base_y):
        						#0			1			2				3
                encoding = [Integer(1,9), Integer(0,1), Integer(1,2), Integer(1,124),
                				#4			5			6				7
                			Integer(1,9), Integer(0,1), Integer(1,2), Integer(1,124),
                				#8			9		
                            Integer(2,3), Integer(1,2), 
                            	#10			11			12				13
                            Integer(1,9), Integer(0,1), Integer(1,2), Integer(1,124),
                            	#14
                            Integer(1,2),
                            	#15
                            Integer(1,6)]

                variables = len(encoding)
                objectives = 1
                super(script, self).__init__(variables, objectives)
                self.types[:] = encoding
                self.class_name = name
                self.base_x=base_x
                self.base_y=base_y

                self.batch_size = 28#de quanto em quanto vai caminhar
                self.num_classes = 3
                self.epochs = 50#repeticoes
                self.solucoes={}#dicionario de solucoes
                self.id = 0
                self.X_train=None;

        def DefaulttDNN(self):
              print("Train Shape",self.X_train.shape[1:])
              model = Sequential()
              input1=random.randint(1,9)
              input2=random.randint(1,124)
              input3=random.randint(1,2)
              input4=random.randint(1,2)
              input5=random.randint(1,9)
              input6=random.randint(1,124)
              input7=random.randint(1,2)
              model.add(Conv2D(filters=input1, kernel_size=input2, strides=input3 ,padding="same", input_shape=self.X_train.shape[1:]))
              model.add(Conv2D(filters=input1, kernel_size=input2, strides=input3 ,padding="same"))
              pool=random.randint(2,3)

              model.add(MaxPooling2D(pool_size=(pool,pool), strides=input4))
              model.add(Conv2D(filters=input5, kernel_size=input6,strides=input7, padding="same"))
              model.add(Flatten())
              model.add(Dense(units=5, activation='sigmoid'))
              return model

        def ModifieddDNN(self,numCamadasModf, model, solution):
            print(solution)
            md=Sequential()
            for i in range(1,7,1):
                  if(i<numCamadasModf or i==5):
                        md.add(model.get_layer(None,i))
                  else:
                        if(i==1): md.add(Conv2D(filters=solution.variables[0], kernel_size=solution.variables[3], strides=solution.variables[2], padding="same",input_shape=self.X_train.shape[1:]))
                        elif(i==2 or i==4): md.add(Conv2D(filters=solution.variables[10], kernel_size=solution.variables[13], strides=solution.variables[12], padding="same"))
                        elif(i==3): md.add(MaxPooling2D(pool_size=(solution.variables[8],solution.variables[8]), strides=solution.variables[9]))
                        elif(i==6): md.add(Dense(units=5,activation='sigmoid'))
            return md
        def getSolucaoFinal(self,info):
                return self.solucoes.get(info)


        def evaluate(self, solution):

                print("Rescaling database")

                X_train,X_test, y_train,y_test = train_test_split(self.base_x,self.base_y,test_size=0.2,random_state=0) 
                                
                print("Criando o modelo")

                self.X_train=X_train

                modelPadrao= self.DefaulttDNN()

                print(modelPadrao.summary())

                model=self.ModifieddDNN(numCamadasModf=solution.variables[15], model=modelPadrao,solution=solution)

                print(model.summary())

                print("Compilando")
                learning_rate = 0.1
                decay_rate = learning_rate / self.epochs
                momentum = 0.8
                sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
                model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])#,'precision','recall', 'f1'])

                print("Iniciando treinamento")
                earlyStopping = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10, verbose=1, mode='auto')
                history=model.fit(X_train,y_train,batch_size=self.batch_size,
                        epochs=self.epochs,
                        callbacks=[earlyStopping],
                        validation_split=0.33,
                        shuffle=True,
                        verbose=2)

               	train_acc = history.history['acc'][-1]
                val_acc = history.history['val_acc'][-1]
		
                #train_recall = history.history['recall'][-1]
                #val_recall = history.history['val_recall'][-1]


                ##train_precision = history.history['precision'][-1]
                #val_precision = history.history['val_precision'][-1]

                #train_f1 = history.history['f1'][-1]
                #val_f1 = history.history['val_f1'][-1]

		#metrics################################################################
                #print('\n acc' , val_acc)
                #predictions = model.predict(X_test, batch_size=self.batch_size)

                #print(classification_report(y_test.argmax(axis=1),
                #                    predictions.argmax(axis=1)))
                #report_lr = precision_recall_fscore_support(y_test.argmax(axis=1),
                #                    predictions.argmax(axis=1),
                #                            average='macro')
                #print("\nprecision = %0.2f, recall = %0.2f, F1 = %0.2f, accuracy = %0.2f\n" % \
                #      (report_lr[0], report_lr[1], report_lr[2], accuracy_score(y_test.argmax(axis=1),
                #                    predictions.argmax(axis=1))))
                #train_epochs = len(history.history['acc'])
            	##########################################################################

                from keras import backend as K
                if K.backend() == 'tensorflow':
                    K.clear_session()

                self.id += 1

                solution.objectives[:] = [-val_acc] # -val_precision ,-val_recall , -val_f1]como o objectives la em cima ta 1 nao precisa do tempo

                variaveis=[]
                print(solution.objectives)
                for i in solution.variables:
                        variaveis.append(i)
                self.solucoes.update({solution.objectives:variaveis})
                
                #Salvar dados
                print("Salvando dados...")
                row = solution.objectives
                writeCSV('files.csv', row)

#MAIN

if __name__ == '__main__':
    problem=MP.MProblem(name='Problem')
    optimizer = NSGAII(problem, population_size=10)
    num_generations = 1
    # inicia o algoritmo
    geracao=0
    for i in range(num_generations):
            # executa por uma geracao
            geracao+=1
            print("::::Geracao: ",geracao,"::::")
            optimizer.run(1)

    for solution in optimizer.result:
            print(solution.objectives)

    print("MEU PROBLEMA")
    
    problem2=script(name='Problem',base_x=problem.getbaseX(),base_y=problem.getbaseY())
    print("optimizer")
    optimizer2= SMPSO(problem2,
                 swarm_size = 5,
                 leader_size = 5,
                 generator = RandomGenerator(),
                 mutation_probability = 0.1,
                 mutation_perturbation = 0.5,
                 max_iterations = 5,
                 mutate = None)
    num_generations=1
    geracao=0
    for i in range(num_generations):
            # executa por uma geracao
            geracao+=1
            print("::::Geracao: ",geracao,"::::")
            optimizer2.run(1)

    print(">>>>>>>>>>>>>RESULT<<<<<<<<<<<<<<<")
    val=(optimizer2.result)[0].objectives
    print("Objective",val[0])
    resp=problem2.getSolucaoFinal(val[0])
    print("Config",resp)

                
