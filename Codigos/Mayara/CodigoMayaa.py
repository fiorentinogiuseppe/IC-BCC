from __future__ import print_function


from keras.layers import Dense, Activation

from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

from keras.models import *

from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import progressbar

from random import shuffle
from Platypus.platypus import *

print(sys.version)


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

from random import shuffle
from Platypus.platypus import *

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


# indica pasta da base de dados, uma pasta com imagens para cada classe
pasta_base = 'Gestures/'

# tamanho do batch de treinamento
batch_size = 200

nb_db_samples, num_classes = getNumSamples(pasta_base)

print('Base tem ', nb_db_samples, ' imagens e ', num_classes, ' classes')


class DnnProblem(Problem):
    def __init__(self, name):
        # nao vai de 0 a 9
        encoding = [
                    Integer(1, 9), Integer(1, 2), Integer(1, 2), Integer(1, 1024),
                    Integer(1, 9), Integer(1, 2), Integer(1, 2), Integer(1, 1024),
                    Integer(2, 3), Integer(1, 2),
                    Integer(1, 9), Integer(1, 2), Integer(1, 2), Integer(1, 1024),
                    Integer(1, 2),
                    Integer(3, 5)]
        variables = len(encoding)
        objectives = 1
        super(DnnProblem, self).__init__(variables, objectives)
        self.types[:] = encoding
        self.class_name = name

        self.batch_size = 28  # de quanto em quanto vai caminhar
        self.num_classes = 3
        self.epochs = 2  # repeticoes
        self.solucoes = {}  # dicionario de solucoes
        self.id = 0



    def getSolucoes(self):
        return self.solucoes

    def getSolucaoFinal(self, info):
        return self.solucoes.get(info)

    def evaluate(self, solution):
        n_camadas_mudar = solution.variables[15]
        print(n_camadas_mudar," foram modificadas")
        print(solution)

        # 1) Decodifica o cromossomo

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
        X_tr, Y_tr, X_te, Y_te, X_val, Y_val = split_dataset_random(base_x, base_y, 0.6, 0.2)

        # 4) Cria e testa o modelo
        print("Criando o modelo")

        image1_input = Input(shape=(X_tr[0].shape[0], X_tr[0].shape[1], 1))
        # conv11 = Conv2D(name="CONV_1", filters=solution.variables[1],
        #                 kernel_size=3,
        #                 strides=(2,2),
        #                 activation='relu')(image1_input)
        #
        # conv12 = Conv2D(name="CONV_2",  filters=solution.variables[1],
        #                 kernel_size=3,
        #                 strides=(2,2),
        #                 activation='relu')(conv11)
        #
        # conv12 = MaxPooling2D(pool_size=(2,2))(conv12)
        #
        # conv12 = Conv2D(name="CONV_3", filters=solution.variables[1],
        #                 kernel_size=3,
        #                 strides=(2,2),
        #                 activation='relu')(conv12)
        # dense1 = Flatten()(conv12)
        #
        # dense1 = Dense(name="DENSE_1", units=50,
        #                activation='sigmoid')(dense1)
        #
        # out = Dense(name="DENSE_OUT", units=num_classes, activation='sigmoid')(dense1)
        #
        # model = Model(image1_input, outputs=out)

        model = Sequential()
        print("fase 1")
        model.add(Conv2D(filters=solution.variables[0],  # se filters 0 da erro
                         kernel_size=solution.variables[3],  # era 3
                         strides=solution.variables[2],  # era 1
                         padding="same",
                         input_shape=X_tr.shape[1:]
                         ))
        print("fase 2")
        print("fase 2")
        print(model.summary())
        model.add(MaxPooling2D(pool_size=(solution.variables[4], solution.variables[4]),
                               strides=solution.variables[5]))  # era 1
        print("fase 3")
        model.add(Flatten())
        model.add(Dense(units=5  # numero de classes
                        , activation='sigmoid'))

        print("Compilando")

        learning_rate = 0.1
        decay_rate = learning_rate / self.epochs
        momentum = 0.8
        sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])



        print("\nSalvando o diagrama do modelo")

        # salva o diagrama do modelo com o id do cromossomo
        # plot_model(model, to_file=str(self.id)+'_model.png', show_shapes=True)

        print("Iniciando treinamento")

        earlyStopping = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10, verbose=1, mode='auto')
        history = model.fit(x=X_tr, y=Y_tr, batch_size=batch_size,
                            epochs=1,#50
                            callbacks=[earlyStopping],
                            validation_data=[X_val, Y_val],
                            shuffle=True,
                            verbose=2)

        # Salva o resultado de treinamento
        plt.clf()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        # plt.savefig(str(self.id)+'_accuracy')
        plt.clf()

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        # plt.savefig(str(self.id)+'_loss')

        train_acc = history.history['acc'][-1]
        val_acc = history.history['val_acc'][-1]


        print('\n acc' , val_acc)
        ##metrics
        predictions = model.predic(X_te, batch_size=batch_size)
        print(classification_report(Y_te.argmax(axis=1),
                                    predictions.argmax(axis=1)))
        report_lr = precision_recall_fscore_support(Y_te.argmax(axis=1),
                                    predictions.argmax(axis=1),
                                            average='macro')

        print("\nprecision = %0.2f, recall = %0.2f, F1 = %0.2f, accuracy = %0.2f\n" % \
              (report_lr[0], report_lr[1], report_lr[2], accuracy_score(Y_te.argmax(axis=1),
                                    predictions.argmax(axis=1))))
        
        train_epochs = len(history.history['acc'])

        start = time.time()

        # Testa
        test_scores = model.evaluate(X_te, Y_te, verbose=0)

        end = time.time()
        test_ms = round((end - start) * 1000, 1)
        print("Test time: ", test_ms, " ms")

        test_acc = test_scores[1]
        print('Test accuracy:', test_acc)

        from keras import backend as K
        if K.backend() == 'tensorflow':
            K.clear_session()

        self.id += 1

        # retorna os objetivos (acuracia e tempo
        solution.objectives[:] = [-val_acc, test_ms]


optimizer = SMPSO(problem=DnnProblem(name="pso"),
                           swarm_size=5,
                           leader_size=5,
                           generator=RandomGenerator(),
                           mutation_probability=0.1,
                           mutation_perturbation=0.5,
                           max_iterations=5,
                           mutate=None)

# define quantidade de gerações
num_generations = 5

# inicia o algoritmo
for i in range(num_generations):
    # executa por uma geração
    optimizer.run(1)

    # salva gráficos de acurácia de validação e tempo para cada geração
    plt.clf()
    plt.xlim([0, 1])
    plt.ylim([0, 10])

    plt.scatter([-s.objectives[0] for s in optimizer.result],
                [s.objectives[1] for s in optimizer.result], c='b', marker='o')
    plt.xlabel("$accuracy$")
    plt.ylabel("$recognition_time$")
    plt.savefig('NSGAIII_gen' + '_' + str(i))
