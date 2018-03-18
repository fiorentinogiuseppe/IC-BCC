#!/home/giuseppe/.virtualenvs/dl4cv/bin/python3.5
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



print(sys.version)


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
        		      #0	        1	    2		      3
                encoding = [Integer(1,9), Integer(0,1), Integer(1,2), Integer(1,124),
                	      #4		5	    6		      7
                	    Integer(1,9), Integer(0,1), Integer(1,2), Integer(1,124),
                	      #8		9		
                            Integer(2,3), Integer(1,2), 
                              #10	        11	    12		      13
                            Integer(1,9), Integer(0,1), Integer(1,2), Integer(1,124),
                              #14
                            Integer(1,2),
                              #15
                            Integer(3,6)]

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
                self.X_train=None

        def DefaulttDNN(self,lencategories):
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
              model.add(Dense(units=lencategories, activation='softmax'))
              return model

        def ModifieddDNN(self,numCamadasModf, model, solution, lencategories):
            print(solution)
            md=Sequential()
            for i in range(1,7,1):
                  if(i<numCamadasModf or i==5):
                        md.add(model.get_layer(None,i))
                  else:
                        if(i==1): md.add(Conv2D(filters=solution.variables[0], kernel_size=solution.variables[3], strides=solution.variables[2], padding="same",input_shape=self.X_train.shape[1:]))
                        elif(i==2 or i==4): md.add(Conv2D(filters=solution.variables[10], kernel_size=solution.variables[13], strides=solution.variables[12], padding="same"))
                        elif(i==3): md.add(MaxPooling2D(pool_size=(solution.variables[8],solution.variables[8]), strides=solution.variables[9]))
                        elif(i==6): md.add(Dense(units=lencategories,activation='softmax'))
            return md
        def getSolucaoFinal(self,info):
                return self.solucoes.get(info)


        def evaluate(self, solution):

                print("Rescaling database")

                X_train,X_test, y_train,y_test = train_test_split(self.base_x,self.base_y,test_size=0.2,random_state=0) 
                                
                print("Criando o modelo")

                self.X_train=X_train

                modelPadrao= self.DefaulttDNN(lencategories=10)#pensar num jeito de automatizar

                print(modelPadrao.summary())

                model=self.ModifieddDNN(numCamadasModf=solution.variables[15], model=modelPadrao,solution=solution, lencategories=10)

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
    '''
    #Base Gestos
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
   
    #Base Mnist
    # input image dimensions
    img_rows, img_cols = 28, 28 #configuração manual a respeito da base

    #output dimension
    num_classes=10 #configuração manual a respeito da base

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test)= mnist.load_data()
    x_train, y_train, x_test, y_test= carregarBase(x_train, y_train, x_test, y_test, img_rows, img_cols)
    
    #Shape (num_samples,28,28)

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
    
    #Base Fashion_Mnist 
    # input image dimensions
    img_rows, img_cols = 28, 28 #configuração manual a respeito da base

    #output dimension
    num_classes=10 #configuração manual a respeito da base

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    #Shape (num_samples,28,28)

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

    
    #Base Cifar10
    
    # input image dimensions
    img_rows, img_cols = 32, 32

    #output dimension
    num_classes=10

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    #Shape (num_samples,3,32,32)


    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    '''
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

    print("Configurando Problem")
    #problem2=script(name='Problem',base_x=problem.getbaseX(),base_y=problem.getbaseY()) #Usar qndo for testar a base de gestos
    problem2=script(name='Problem',base_x=x_train,base_y=y_train) #Qualquer outra base usar esse
    
    print("Configurando Otimizador")
    optimizer2= SMPSO(problem2,
                 swarm_size = 5,
                 leader_size = 5,
                 generator = RandomGenerator(),
                 mutation_probability = 0.1,
                 mutation_perturbation = 0.5,
                 max_iterations = 5,
                 mutate = None)

    print("Rodando o codigo")
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
