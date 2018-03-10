#!/home/giuseppe/.virtualenvs/dl4cv/bin/python3.5
from platypus import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import MProblem as MP
import SProblem as SP
import os
from platypus import NSGAII, Problem, Real,SMPSO
import pandas as pd
from sklearn.preprocessing import LabelEncoder


if __name__ == '__main__':
    # Estrutura do cromossomo:
    # 4 primeiras partes eh convulutiva
    # 2 segundas partes eh max_pooling
    # 1 ultima parte eh Dense
    # TOTAL: 10 bits
    print ("DATABASE")
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
    #os.system('spd-say "The first part has finished"')
    '''
    #IRIS PROBLEM
    dataset = pd.read_csv('iris.csv')
    X = dataset.iloc[:,1:4].values
    y = dataset.iloc[:,4].values

    encoder =  LabelEncoder()
    y1 = encoder.fit_transform(y)

    Y = pd.get_dummies(y1).values
    '''
    print("MEU PROBLEMA")
    
    problem2=SP.SProblem(name='Problem',base_x=problem.getbaseX(),base_y=problem.getbaseY())
    #problem2=SP.SProblem(name='Problem',base_x=X,base_y=Y)
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

    os.system('cls' if os.name == 'nt' else 'clear')
    print(">>>>>>>>>>>>>RESULT<<<<<<<<<<<<<<<")
    val=(optimizer2.result)[0].objectives
    print("Objective",val[0])
    resp=problem2.getSolucaoFinal(val[0])
    print("Config",resp)
    #os.system('spd-say "The second part has finished"')