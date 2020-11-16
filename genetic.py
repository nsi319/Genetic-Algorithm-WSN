import random
import numpy as np
from generation import Generation
from chromosome import Chromosome
from cluster import Clustering

class Genetic:
    def __init__(self, population, pSelection, pMutation, pCrossover, generation, data, generationCount):
        self.population = population
        self.pSelection = pSelection
        self.pMutation = pMutation
        self.pCrossover = pCrossover
        self.generation = generation
        self.data = data
        self.generationCount = generationCount

    def geneticProcess(self, generation):
        generation_total = self.generation
        pSelection = self.pSelection
        pMutation = self.pMutation
        pCrossover = self.pCrossover
        population = self.population
        print("\n")
        print("----------------------Generation Number: ",self.generationCount, "----------------------------")
        generation.sortChromosomes()

        generation = self.selection(generation)
        generation = self.crossover(generation)
        generation = self.mutation(generation)
        self.generationCount += 1

        return generation, self.generationCount

    def selection(self, generation):
        population = self.population
        pSelection = self.pSelection

        # replace the worst pSelection*population individual with the best pSelection*population individual
        for i in range(0, int(pSelection * population)):
            generation.chromosomes[population -
                                   1 - i] = generation.chromosomes[i]

        # sort chromosomes after ranking selection
        generation.sortChromosomes()
        return generation

    def crossover(self, generation):
        population = self.population
        pCrossover = self.pCrossover

        index = random.sample(
            range(0, population - 1), int(pCrossover * population))

        for i in range(int(len(index) / 2),+2): 
            generation = self.doCrossover(
                generation, i, index)

        generation.sortChromosomes()

        return generation

    def doCrossover(self, generation, i, index):

        chromo = generation.chromosomes
        length = chromo[0].length
        cut = random.randint(1, length - 1)
        parent1 = chromo[index[i]]
        parent2 = chromo[index[i + 1]]
        genesChild1 = parent1.genes[0:cut] + parent2.genes[cut:length]
        genesChild2 = parent1.genes[cut:length] + parent2.genes[0:cut]
        child1 = Chromosome(genesChild1, len(genesChild1))
        child2 = Chromosome(genesChild2, len(genesChild2))

        clustering = Clustering(generation, self.data)
        child1 = clustering.calculateFitnessChild(child1)
        child2 = clustering.calculateFitnessChild(child2)

        temp = []
        temp.append(parent1)
        temp.append(parent2)
        temp.append(child1)
        temp.append(child2)

        temp = sorted(temp, reverse=True,
                       key=lambda elem: elem.fitness)

        generation.chromosomes[index[i]] = temp[0]
        generation.chromosomes[index[i + 1]] = temp[1]

        return generation

    def mutation(self, generation):
        population = self.population
        fitnessList = []
        generationAfterM = Generation(population, generation.generationCount)
        flagMutation = (np.zeros(population)).tolist()

        for i in range(population):
            temp = generation.chromosomes[i]
            fitnessList.append(temp.fitness)

        for i in range(population):
            if i == 0:  # Ibest doesn't need mutation
                generationAfterM.chromosomes.append(generation.chromosomes[0])
                flagMutation[0] = 0
            else:
                generationAfterM = self.doMutation(
                    generation.chromosomes[i],	generationAfterM, flagMutation, fitnessList, i)

        generationAfterM.sortChromosomes()
        return generationAfterM

    def doMutation(self, chromosomeBeforeM, generationAfterM, flagMutation, fitnessList, i):
        pMutation = self.pMutation
        dice = []
        length = len(chromosomeBeforeM.genes)
        chromosome = Chromosome([], length)
        geneFlag = []

        for j in range(length):
            dice.append(float('%.2f' % random.uniform(0.0, 1.0)))
            if dice[j] > pMutation:
                chromosome.genes.append(chromosomeBeforeM.genes[j])
                geneFlag.append(0)

            if dice[j] <= pMutation:
                chromosome.genes.append(
                    float('%.2f' % random.uniform(0.0, 1.0)))
                geneFlag.append(1)

        check = sum(geneFlag)

        if check == 0:
            flagMutation[i] = 0
            chromosome.fitness = fitnessList[i]
        else:
            flagMutation[i] = 1

            clustering = Clustering(chromosomeBeforeM, self.data)
            chromosome = clustering.calculateChildFitness(chromosome)

        generationAfterM.chromosomes.append(chromosome)
        return generationAfterM
