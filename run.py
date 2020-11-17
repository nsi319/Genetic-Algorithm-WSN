import configparser
import numpy as np
import pandas as pd

from cluster import Clustering
from genetic import Genetic
from generation import Generation

def readVars(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    generation = int(config.get("vars", "generation"))
    population = int(config.get("vars", "population")) 
    pSelection = float(config.get("vars", "selection"))
    pMutation = float(config.get("vars", "mutation"))
    pCrossover = float(config.get("vars", "crossover"))
    heads = int(config.get("vars","heads"))
    return generation, pSelection, pMutation, pCrossover, population, heads


if __name__ == '__main__':
    config_file = "config.txt"
    data = pd.read_csv('data/points.csv', header=None)
    dim = data.shape[1]

    generationCount = 0
    generation_total, pSelection, pMutation, pCrossover, population, heads = readVars(config_file)

    print("-------------Genetic Algo Parameters-------------------")
    print("Total Generations: ", generation_total)
    print("Population size: ", population)
    print("Minimum number of clusters: ", heads)
    print("Probability of Selection: ", pSelection)
    print("Probability of Crossover: ", pCrossover)
    print("Probability of Mutation: ", pMutation)
    print("-------------------------------------------------------")

    chromosome_length = data.shape[0]  

    initial = Generation(population, 0)

    initial.randomGenerateChromosomes(chromosome_length) 

    clustering = Clustering(initial, data)  

    generation = clustering.calculateGenerationFitness()
    while generationCount <= generation_total:
        GA = Genetic(population, pSelection, pMutation, pCrossover, generation, data, generationCount)
        generation, generationCount = GA.geneticProcess(generation)
        clustering.printChromoData(generation.chromosomes[0])
    clustering.output_result()
