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
        def __init__(self, name, base_x, base_y, lencategories):
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
                self.nb_db_samples, self.num_classes = getNumSamples(pasta_base)

                self.batch_size = 200#de quanto em quanto vai caminhar
                self.epochs = 50#repeticoes
                self.solucoes={}#dicionario de solucoes
                self.id = 0
                self.lencategories=lencategories

        def DefaulttDNN(self,input_shape,lencategories):
              model = Sequential()
	      model.add(Conv2D(filters=5, kernel_size=4, input_shape=input_shape))
	      model.add(Conv2D(filters=5, kernel_size=4))
	      model.add(MaxPooling2D())
	      model.add(Conv2D(filters=5, kernel_size=4))
	      model.add(Flatten())
	      model.add(Dense(units=5))
              return model

        def ModifieddDNN(self,numCamadasModf, model, solution, lencategories, input_shape):
            print(solution)
            md=Sequential()
            for i in range(1,7,1):
                  if(i<numCamadasModf or i==5):
                        md.add(model.get_layer(None,i))
                  else:
                        if(i==1): md.add(Conv2D(filters=solution.variables[0], kernel_size=solution.variables[3], strides=solution.variables[2], padding="same",input_shape=input_shape))
                        elif(i==2): md.add(Conv2D(filters=solution.variables[4], kernel_size=solution.variables[7], strides=solution.variables[6], padding="same"))
                        elif(i==3): md.add(MaxPooling2D(pool_size=(solution.variables[8],solution.variables[8]), strides=solution.variables[9]))
                        elif(i==4): md.add(Conv2D(filters=solution.variables[10], kernel_size=solution.variables[13], strides=solution.variables[12], padding="same"))
                        elif(i==6): md.add(Dense(units=self.lencategories,activation='softmax'))
            return md
        def getSolucaoFinal(self,info):
                return self.solucoes.get(info)


        def evaluate(self, solution):

                print("Rescaling database")

                X_train, X_test, y_train, y_test = train_test_split(self.base_x,self.base_y,test_size=0.2,random_state=0) 
                                
                print("Criando o modelo")

                self.X_train=X_train

                modelPadrao= self.DefaulttDNN(input_shape=self.X_train.shape[1:],lencategories=5)#pensar num jeito de automatizar

                print(modelPadrao.summary())

                model=self.ModifieddDNN(numCamadasModf=solution.variables[15], model=modelPadrao,solution=solution, lencategories=5, input_shape=self.X_train.shape[1:])

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
                print('\n acc' , val_acc)
                predictions = model.predict(X_test, batch_size=self.batch_size)

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
    '''
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
    
    print("Configuração da Base")
    # input image dimensions
    img_rows, img_cols = 28, 28 #configuração manual a respeito da base que sera carregada (tamanho de cada imagem)

    #output dimension
    num_classes=10 #configuração manual a respeito da base que sera carregada (quantas classes tem a base)
    '''
    #print("Carregando Base")
    #Load DB
    #x_train, y_train, x_test, y_test=load_notmnist() #Caso queira carregar bases que nao sao do keras o processo eh o mesmo apenas obtenha o dados de treino
                                                            # e teste e envie para a configuração de base. A bases de Gestos do prof sergio ja ta configurada int nao
                                                            #precisa ir pra proxima fase. So ir direto pro problema    
    #print("Configurando Base")
    #x_train, y_train, x_test, y_test= configBase(x_train, y_train, x_test, y_test, img_rows, img_cols)#Usar para outras bases

    print("Configurando Problem")
    problem2=script(name='Problem',base_x=base_x, base_y=base_y, lencategories=num_classes) #Qualquer outra base usar esse
    
    print("Configurando Otimizador")
    optimizer2= SMPSO(problem2,
                 swarm_size = 30,
                 leader_size = 5,
                 generator = RandomGenerator(),
                 mutation_probability = 0.1,
                 mutation_perturbation = 0.5,
                 max_iterations = 100,
                 mutate = None)

    print("Rodando o codigo")
    num_repet=10
    repeticao=0
    for i in range(num_repet):
            # executa por uma geracao
            repeticao+=1
            print("::::Repeticao: ",repeticao,"::::")
            optimizer2.run(1)

    print(">>>>>>>>>>>>>RESULT<<<<<<<<<<<<<<<")
    val=(optimizer2.result)[0].objectives
    print("Objective",val[0])
    resp=problem2.getSolucaoFinal(val[0])
    print("Config",resp)

