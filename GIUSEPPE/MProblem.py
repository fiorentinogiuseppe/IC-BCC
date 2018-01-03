from platypus import Problem, Solution, EPSILON
from platypus import Real, Binary
from keras.models import *
from keras.layers import *
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import progressbar
import sys
from random import shuffle
import keras.backend as K
import Config as configuracao


from platypus import *

class MProblem(Problem):
        def __init__(self, name):
                # Estrutura do cromossomo:
                # tamanho da imagem (2 bits),
                # filtros convolutivos na primeira camada (2 bits),
                # filtros convolutivos na segunda camada (2 bits),
                # learnin_rate(2 bits),
                # momentum (2 bits)
                # TOTAL: 10 bits
                encoding = [Binary(2), Binary(2), Binary(2), Binary(2), Binary(2)]
                variables = len(encoding)
                # dois objetivos: acuracia e tempo
                objectives = 2
                super(MProblem, self).__init__(variables, objectives)
                # indica pasta da base de dados, uma pasta com imagens para cada classe    
                self.types[:] = encoding
                self.class_name = name
                self.id = 0
                self.cnfg=configuracao.config()
                self.base_x=None
                self.base_y=None

        def getbaseX(self):
            return self.base_x

        def getbaseY(self):
            return self.base_y

        # Funcao de teste do cromossomo
        def evaluate(self, solution):
                #1) Decodifica o cromossomo

                # tamanho da imagem

                im_sz = sum(2 ** i * b for i, b in enumerate(solution.variables[0]))
                options_im_size = [16, 32, 48, 64]

                # filters

                filters1 = sum(2 ** i * b for i, b in enumerate(solution.variables[1])) ##solution.variables [v,F]
                options_filters1 = [3, 6, 12, 24]

                filters2 = sum(2 ** i * b for i, b in enumerate(solution.variables[2]))
                options_filters2 = [3, 6, 12, 24]

                learnin_rate = sum(2 ** i * b for i, b in enumerate(solution.variables[3]))
                options_learnin_rate = [0.001, 0.01, 0.1, 0.5]

                momentum = sum(2 ** i * b for i, b in enumerate(solution.variables[4]))
                options_momentum = [0.0001, 0.001, 0.01, 0.1]

                img_width, img_height = options_im_size[im_sz], options_im_size[im_sz]

                # 2) Carrega a base de dados
                print('Rescaling database to', img_width, 'x', img_height, ' pixels')
                self.base_x, self.base_y = self.cnfg.array_from_dir(data_dir=self.cnfg.getPastaBase(), nb_samples=self.cnfg.getNbDbSamples(),
                                                nb_classes=self.cnfg.getNumClasses(), width=img_width, height=img_height)

                # 3) Separa aleatoriamente em treinamento (60%), validacao (20%) e teste (20%)
                X_tr, Y_tr, X_te, Y_te, X_val, Y_val = self.cnfg.split_dataset_random(self.base_x, self.base_y, 0.6, 0.2)

                # 4) Cria e testa o modelo
                print("Criando o modelo")
                print(X_tr.shape)
                print(X_te.shape)
                image1_input = Input(shape=(X_tr[0].shape[0], X_tr[0].shape[1], 1))

                conv11 = Conv2D(name="CONV_1", 
                                filters=options_filters1[filters1],
                                kernel_size=3,
                                activation='relu') (image1_input)
                conv11 = MaxPooling2D(pool_size=(2, 2))(conv11)

                conv12 = Conv2D(name="CONV_2", filters=options_filters2[filters2],
                                kernel_size=3,
                                activation='relu')(conv11)

                conv12 = MaxPooling2D(pool_size=(2, 2))(conv12)

                dense1 = Flatten()(conv12)

                dense1 = Dense(name="DENSE_1", units=50,
                                          activation='sigmoid')(dense1)

                out = Dense(name="DENSE_OUT", units=self.cnfg.getNumClasses(), activation='sigmoid')(dense1)

                model = Model(image1_input, outputs=out)

                print("Compilando")

                model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr=options_learnin_rate[learnin_rate], momentum=options_momentum[momentum]),
                      metrics=['accuracy'])

                print("\nSalvando o diagrama do modelo")

                # salva o diagrama do modelo com o id do cromossomo
                # plot_model(model, to_file=str(self.id)+'_model.png', show_shapes=True)

                print("Iniciando treinamento")

                earlyStopping = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10, verbose=1, mode='auto')
                history = model.fit(x=X_tr, y=Y_tr, batch_size=self.cnfg.getBatchSize(),
                                    epochs=50,
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
