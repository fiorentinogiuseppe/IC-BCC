#!/home/giuseppe/.virtualenvs/dl4cv/bin/python3.5
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


from random import shuffle

from platypus import *

print(sys.version)

#Salvar em arquivos
def writeCSV(self, nameFile, row):
      self.fileCSV = csv.writer(open(nameFile, "a"))
      self.fileCSV.writerow(row)

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
pasta_base = 'DB/Gestures/'

# tamanho do batch de treinamento
batch_size = 200

nb_db_samples, num_classes = getNumSamples(pasta_base)

print('Base tem ', nb_db_samples, ' imagens e ', num_classes, ' classes')

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
                            Integer(0,4)]

                variables = len(encoding)
                objectives = 4
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

        def evaluate(self, solution):
                print(solution)
                print("Rescaling database")
                X_train,X_test, y_train,y_test = split_dataset_random(self.base_x,self.base_y,test_size=0.2,random_state=0) 
                
                print("Train Shape",X_train.shape,"\nTeste Shap:",X_test.shape)
                
                print("Criando o modelo")
                if(solution.variables[1]==0):
                        pad1="valid"
                else:
                        pad1="same"

                model = Sequential()

                model.add(Conv2D(filters=solution.variables[0], #se filters 0 da erro
                        kernel_size=solution.variables[3],#era 3
                        strides=solution.variables[2], #era 1
                        padding=pad1,
                        input_shape=X_train.shape[1:]
                        ))

                if(solution.variables[5]==0):
                        pad2="valid"
                else:
                        pad2="same"

                model.add(Conv2D(filters=solution.variables[4], #se filters 0 da erro
                        kernel_size=solution.variables[7],#era 3
                        strides=solution.variables[6], #era 1
                        padding=pad2,
                        input_shape=X_train.shape[1:]
                        ))

                model.add(MaxPooling2D(pool_size=(solution.variables[8],solution.variables[8]), 
                        strides=solution.variables[9]))#era 1

                if(solution.variables[11]==0):
                        pad3="valid"
                else:
                        pad3="same"
                model.add(Conv2D(filters=solution.variables[10], #se filters 0 da erro
                        kernel_size=solution.variables[13],#era 3
                        strides=solution.variables[12], #era 1
                        padding=pad3,
                        input_shape=X_train.shape[1:]
                        ))
                model.add(Flatten())
                model.add(Dense(units=5#numero de classes 
                        ,activation='sigmoid'))

                print(model.summary())

                print("Compilando")
                learning_rate = 0.1
                decay_rate = learning_rate / self.epochs
                momentum = 0.8
                sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
                model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy','precision','recall', 'f1'])


                print("\nSalvando o diagrama do modelo")

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

                train_recall = history.history['recall'][-1]
                val_recall = history.history['val_recall'][-1]


                train_precision = history.history['precision'][-1]
                val_precision = history.history['val_precision'][-1]

                train_f1 = history.history['f1'][-1]
                val_f1 = history.history['val_f1'][-1]


                from keras import backend as K
                if K.backend() == 'tensorflow':
                    K.clear_session()

                self.id += 1

                solution.objectives[:] = [-val_acc,-val_precision ,-val_recall , -val_f1]#como o objectives la em cima ta 1 nao precisa do tempo

                variaveis=[]
                print(solution.objectives)
                for i in solution.variables:
                        variaveis.append(i)
                self.solucoes.update({solution.objectives:variaveis})
                
                #Salvar dados
                print("Salvando dados...")
                row = solution.objectives
                writeCSV('files.csv', row)

                