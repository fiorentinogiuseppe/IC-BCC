
from keras.models import *
from keras.layers import *
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import os
import progressbar
import sys
from platypus import NSGAII, Problem, Real
from sklearn.model_selection import train_test_split
import csv
from random import *
from random import shuffle
from platypus import *
from keras.datasets import cifar10
import keras
from keras.datasets import mnist
from scipy.misc import imread, imresize
from sklearn.model_selection import train_test_split
from glob import glob

configs = {}  # dicionario de resultados config  "results.update({'config1': 'resultado'})
melhoracc = None

A = [range(1, 10), range(0, 2), range(1, 3), range(1, 125),  # Conv2D remover
     range(1, 10), range(0, 2), range(1, 3), range(1, 125),  # Conv2D remover
     range(2, 4), range(1, 3),  # MaxPooling2D
     range(1, 10), range(0, 2), range(1, 3), range(1, 125),  # Conv2D
     range(1, 3),  # Dense
     range(1, 7)]  # Numero de camadas Modificadas


def load_notmnist(path='./notMNIST_small', letters='ABCDEFGHIJ',
                  img_shape=(28, 28), test_size=0.25, one_hot=False):
    # download data if it's missing. If you have any problems, go to the urls and load it manually.
    if not os.path.exists(path):
        print("Downloading data...")
        assert os.system(
            'curl http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz > notMNIST_small.tar.gz') == 0
        print("Extracting ...")
        assert os.system('tar -zxvf notMNIST_small.tar.gz > untar_notmnist.log') == 0

    data, labels = [], []
    print("Parsing...")
    for img_path in glob(os.path.join(path, '*/*')):
        class_i = img_path.split('/')[-2]
        if class_i not in letters: continue
        try:
            data.append(imresize(imread(img_path), img_shape))
            labels.append(class_i, )
        except:
            print("found broken img: %s [it's ok if <10 images are broken]" % img_path)

    data = np.stack(data)[:, None].astype('float32')
    data = (data - np.mean(data)) / np.std(data)

    # convert classes to ints
    letter_to_i = {l: i for i, l in enumerate(letters)}
    labels = np.array(list(map(letter_to_i.get, labels)))

    if one_hot:
        labels = (np.arange(np.max(labels) + 1)[None, :] == labels[:, None]).astype('float32')

    # split into train/test
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[3], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[3], 1)

    print("Done")
    return X_train, y_train, X_test, y_test


def configBase(x_train, y_train, x_test, y_test, img_rows, img_cols):
    tamanho = x_train.shape
    print(tamanho)
    if (len(tamanho) == 4):
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

    elif (len(tamanho) == 3):
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


def DefaultDNN(gera, input_shape):
    model = Sequential()
    model.add(Conv2D(filters=5, kernel_size=4, padding="same", input_shape=input_shape))
    model.add(Conv2D(filters=5, kernel_size=4, padding="same"))
    model.add(MaxPooling2D(pool_size=[2, 2], strides=2))
    model.add(Conv2D(filters=5, kernel_size=4, padding="same"))
    model.add(Flatten())
    model.add(Dense(units=5, activation='softmax'))

    configs.update({gera: model})

    return model


def ModifiedDNN(numCamadasModf, model, input_shape):
    # CONV2D
    input0 = random.randint(1, 9)
    input1 = random.randint(0, 1)
    input2 = random.randint(1, 2)
    input3 = random.randint(1, 1024)
    # MoxPooling2D
    input4 = random.randint(1, 9)
    input5 = random.randint(0, 1)
    # CONV2D
    input6 = random.randint(1, 2)
    input7 = random.randint(1, 1024)
    input8 = random.randint(2, 3)
    input9 = random.randint(1, 2)
    # CONV2D
    input10 = random.randint(1, 9)
    input11 = random.randint(0, 1)
    input12 = random.randint(1, 2)
    input13 = random.randint(1, 1024)
    # DENSE
    input14 = random.randint(1, 2)

    md = Sequential()
    for i in range(1, 7, 1):
        if (i < numCamadasModf or i == 5):
            md.add(model.get_layer(None, i))
        else:
            if (i == 1): md.add(
                Conv2D(filters=input0, kernel_size=input3, strides=input2, padding="same", input_shape=input_shape))
            elif (i == 2): md.add(Conv2D(filters=input4, kernel_size=input7, strides=input6, padding="same"))
            elif (i == 3): md.add(MaxPooling2D(pool_size=(input8, input8), strides=input9))
            elif (i == 4): md.add(Conv2D(filters=input10, kernel_size=input13, strides=input12, padding="same"))
            elif (i == 6): md.add(Dense(units=5, activation='softmax'))


    return md


batch_size = 128
num_classes = 10
epochs = 12

print("Configuração da Base")
# input image dimensions
img_rows, img_cols = 28, 28  # configuração manual a respeito da base que sera carregada (tamanho de cada imagem)

# output dimension
num_classes = 10  # configuração manual a respeito da base que sera carregada (quantas classes tem a base)

print("Carregando Base")
# Load DB
#x_train, y_train, x_test, y_test = load_notmnist()  # Caso queira carregar bases que nao sao do keras o processo eh o mesmo apenas obtenha o dados de treino
# e teste e envie para a configuração de base. A bases de Gestos do prof sergio ja ta configurada int nao
# precisa ir pra proxima fase. So ir direto pro problema


print("Configurando Base")
#x_train, y_train, x_test, y_test = configBase(x_train, y_train, x_test, y_test, img_rows, img_cols)

from keras.datasets import cifar10

#(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Retorna quantidade de imagens em diretorio
def getNumSamples(src):
    sum = 0
    for cl in os.listdir(src):
        class_dir = os.path.join(src, cl)
        files = os.listdir(class_dir)
        l = len(files)
        sum += l

    return sum, len(os.listdir(src))

import cv2
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


# indica pasta da base de dados, uma pasta com imagens para cada classe
pasta_base = 'Gestures/'

# tamanho do batch de treinamento
batch_size = 200

nb_db_samples, num_classes = getNumSamples(pasta_base)

print('Base tem ', nb_db_samples, ' imagens e ', num_classes, ' classes')

# tamanho da imagem
        ##im_sz = sum(2 ** i * b for i, b in enumerate(bina[random.randint(0,4)], bina[random.randint(0,4)]))
options_im_size = [16, 32, 48, 64]
im_sz=random.randint(0,3)
img_width, img_height = options_im_size[im_sz], options_im_size[im_sz]

        # 2) Carrega a base de dados
print('Rescaling database to', img_width, 'x', img_height, ' pixels')
base_x, base_y = array_from_dir(data_dir=pasta_base, nb_samples=nb_db_samples,
                                        nb_classes=num_classes, width=img_width, height=img_height)

        # 3) Separa aleatoriamente em treinamento (60%), validação (20%) e teste (20%)
x_train, y_train, x_test, y_test, X_val, Y_val = split_dataset_random(base_x, base_y, 0.6, 0.2)
from sklearn.metrics import precision_recall_fscore_support, classification_report
soma =0
soma_fm = 0
soma_t = 0
num= 30
for i in range(num):
    print(i)
    start = time.time()
    model = DefaultDNN(i, x_train.shape[1:])



    md = ModifiedDNN(random.randint(3, 6), model, x_train.shape[1:])

    learning_rate = 0.1
    epochs = 50
    decay_rate = learning_rate / epochs
    momentum = 0.8
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    md.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    earlyStopping = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10, verbose=1, mode='auto')
    history = md.fit(x_test, y_test, batch_size=200,
                        epochs=50,
                        callbacks=[earlyStopping],
                        validation_data=[X_val, Y_val],
                        shuffle=True,
                        verbose=2)

    train_acc = history.history['acc'][-1]
    val_acc = history.history['val_acc'][-1]

    end = time.time()
    tmp = round((end - start), 2)

    soma_t = soma_t + tmp
    print(i,val_acc)
    print("tmp ",  tmp)
    soma = soma + val_acc
##metrics


    predictions = model.predict(x_test, batch_size=batch_size)
    print(classification_report(y_test.argmax(axis=1),
            predictions.argmax(axis=1)))
    report_lr = precision_recall_fscore_support(y_test.argmax(axis=1),
                                        predictions.argmax(axis=1),
                                                average='macro')



    print("\nprecision = %0.2f, recall = %0.2f, F1 = %0.2f\n" % \
              (report_lr[0], report_lr[1], report_lr[2]))

    soma_fm = soma_fm + report_lr[2]
    print("fm", report_lr[2])

print("Media acuracia ", soma/num, "i ", i)
print("Media tmpo ",  soma_t/num , "i ", i)
print("Media fm ", soma_fm/num , "i ", i)

#print("Dicionarios de config", config)
