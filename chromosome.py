import random
import configparser

class Chromosome:
    def __init__(self, genes, length):
        self.genes = genes
        self.length = length
        self.fitness = 0

    def randomGenerateChromosome(self):
        self.genes = [0]*self.length
        number_ones = 0
        config = configparser.ConfigParser()
        config.read("config.txt")
        heads = int(config.get("vars", "heads"))    # minimum number of heads

        for i in range(100000):
            if number_ones==heads:
                break
            index = int(random.uniform(0,self.length))
            if self.genes[index]==0:
                self.genes[index]=1
                number_ones = number_ones + 1
        return self
