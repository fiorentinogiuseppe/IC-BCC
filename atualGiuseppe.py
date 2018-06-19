import pandas as pd
from keras.models import *
from keras.layers import *
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.datasets import cifar10, mnist
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

# Salvar em arquivos
def writeCSV(nameFile, row):
    fileCSV = csv.writer(open(nameFile, "a"))
    fileCSV.writerow(row)

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
        #self.nb_db_samples, self.num_classes = getNumSamples(pasta_base)

        self.batch_size = 200  # de quanto em quanto vai caminhar
        self.epochs = 50  # repeticoes
        self.solucoes = {}  # dicionario de solucoes
        self.melhores = np.zeros((1,2))
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
    def getTodosValores(self):
        return self.melhores


    def evaluate(self, solution):

        #options_im_size = [16, 32, 48, 64]
        #im_sz = random.randint(0, 3)
        #img_width, img_height = options_im_size[im_sz], options_im_size[im_sz]

        # 2) Carrega a base de dados
        #print('Rescaling database to', img_width, 'x', img_height, ' pixels')
        #base_x, base_y = array_from_dir(data_dir=pasta_base, nb_samples=nb_db_samples,
        #                                nb_classes=num_classes, width=img_width, height=img_height)

        # 3) Separa aleatoriamente em treinamento (60%), validacaoo (20%) e teste (20%)
        X_train, X_test, y_train, y_test = train_test_split(self.base_x,self.base_y,test_size=0.01,random_state=0)#''0.01'''
        X_train, X_test, y_train, y_test = train_test_split(X_test,y_test,test_size=0.1,random_state=0)
        #X_train, X_test, y_train, y_test = train_test_split(X_test,y_test,test_size=0.1,random_state=0)

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
                            validation_split=0.33,
                            shuffle=True,
                            verbose=2)

        train_acc = history.history['acc'][-1]
        val_acc = history.history['val_acc'][-1]


        # metrics################################################################
        print('\n acc', val_acc)
        predictions = model.predict(X_test, batch_size=self.batch_size)
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

    print("Carregando Base")
    #Load DB
    #x_train, y_train, x_test, y_test=load_notmnist()  
    #print("Configurando Base")
    #x_train, y_train, x_test, y_test= configBase(x_train, y_train, x_test, y_test, img_rows, img_cols)
    '''
    print("Carregando Base")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # input image dimensions
    img_rows, img_cols = 32, 32
    #output dimension
    num_classes=10

    print("Configurando Base")
    x_train, y_train, x_test, y_test = configBase(x_train, y_train, x_test, y_test, img_rows, img_cols)
   
    print("Configurando Problem")
    problem2=script(name='Problem',base_x=x_train, base_y=y_train,  lencategories=num_classes) 


    print("Rodando o codigo")
    num_repet = 10
    repeticao = 0

    soma = 0
    soma_fm = 0
    soma_t = 0

    for i in range(num_repet):
        # executa por uma geracao
        repeticao += 1
        print("Configurando Otimizador")
        optimizer = SMPSO(problem2,
                          swarm_size=30,
                          leader_size=5,
                          generator=RandomGenerator(),
                          mutation_probability=0.1,
                          mutation_perturbation=0.5,
                          max_iterations=100,
                          mutate=None)

        print("::::Repeticao: ", repeticao, "::::")
        start = time.time()
        optimizer.run(1)
        end = time.time()
        print(">>>>>>>>>>>>>RESULT<<<<<<<<<<<<<<<\n")
        val = (optimizer.result)[0].objectives
        print("Objective", -val[0], "fm ", -val[1])


        tmp = round((end - start), 2)
        
        soma_t = soma_t + tmp
        #soma = soma + (-val[0])
        #soma_fm = soma_fm + (-val[1])

    #DESVIO PADRAO, MEDIA, DISPERSAO
    train_df = pd.read_csv('files.csv')
    print("acc")
    print(train_df['acc'].mean())
    print(train_df['acc'].std())

    print("fm")
    print(train_df['fm'].mean())
    print(train_df['fm'].std())
    

    #print("Media acuracia ", soma / num_repet, "i ", num_repet)
    print("Media tmpo ", soma_t / num_repet, "i ", num_repet)
    #print("Media fm ", soma_fm / num_repet, "i ", num_repet)
