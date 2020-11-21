import pandas as pd
import math
import json
import numpy as np
import configparser

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
from scipy.spatial import distance

from kmeans import kmeans_cluster


pd.options.mode.chained_assignment = None
class Clustering:
    def __init__(self, generation, data):
        self.generation = generation
        self.data = data
        self.max_acc = 0
        self.max_fitness = 0
        self.best_cluster = []
        self.final_cluster_heads = []
    
    def customFitnessScore(self, genes):
        N = len(genes)
        config = configparser.ConfigParser()
        config.read("config.txt")
        w = float(config.get("vars", "w"))      # weight parameter (ensure balance between distance and heads measure)
        
        heads = []
        head_to_sink = []
        cluster_index = [None]*N
        distance_to_head_list = [None]*N
        c_index = 0
        D = 0
        N = len(genes)
        for i in range(N):
            if genes[i]>=0.5:
                heads.append(i)
                head_to_sink.append(math.sqrt(pow(self.data.iloc[i][0],2) + pow(self.data.iloc[i][1],2)))
                distance_to_head_list[i] = 0
                cluster_index[i] = c_index
                c_index = c_index + 1
            D = D + math.sqrt(pow(self.data.iloc[i][0],2) + pow(self.data.iloc[i][1],2))

        H = len(heads)
    
        for i in range(N):
            if genes[i]<0.5:
                x,y = self.data.iloc[i]
                dmin = 10000000
                chosen_cluster = 0
                distance_to_head = 0
                for j in range(H):
                    dist = math.sqrt(pow(self.data.iloc[i][0]-self.data.iloc[heads[j]][0],2) + pow(self.data.iloc[i][1]-self.data.iloc[heads[j]][1],2))
                    dist = dist + head_to_sink[j]
                    if dist < dmin:
                        dmin = dist
                        chosen_cluster = j
                        distance_to_head = dist - head_to_sink[j]
                distance_to_head_list[i] = distance_to_head
                cluster_index[i] = chosen_cluster

        # print("Inside child chromo", genes, distance_to_head_list, head_to_sink)

        distance_p = sum(distance_to_head_list) + sum(head_to_sink)
        fitness = w * (D - distance_p) + (1-w)*(N-H)
        
        return fitness

    def calculateChildFitness(self, childChromosome):
        genes = childChromosome.genes
        childChromosome.fitness = self.customFitnessScore(genes)
        return childChromosome


    def calculateGenerationFitness(self):
        generation = self.generation
        numOfInd = generation.numberOfIndividual  
        data = self.data         
        chromo = generation.chromosomes

        for p in range(0, numOfInd):
            # print(chromo[i].genes)
            genes = chromo[p].genes
            generation.chromosomes[p].fitness = self.customFitnessScore(genes)

        return generation

    def printChromoData(self, chromo):

        genes = chromo.genes
        N = len(genes)

        config = configparser.ConfigParser()
        config.read("config.txt")
        heads_no = int(config.get("vars", "heads"))    # number of heads
        w = float(config.get("vars", "w"))      # weight param
        
        heads = []
        head_to_sink = []
        cluster_index = [None]*N
        distance_to_head_list = [None]*N
        c_index = 0
        D = 0
        N = len(genes)
        for i in range(N):
            if genes[i]>=0.5:
                heads.append(i)
                head_to_sink.append(math.sqrt(pow(self.data.iloc[i][0],2) + pow(self.data.iloc[i][1],2)))
                distance_to_head_list[i] = 0
                cluster_index[i] = c_index
                c_index = c_index + 1
            D = D + math.sqrt(pow(self.data.iloc[i][0],2) + pow(self.data.iloc[i][1],2))

        H = len(heads)

        for i in range(N):
            if genes[i]<0.5:
                x,y = self.data.iloc[i]
                dmin = 10000000
                chosen_cluster = 0
                distance_to_head = 0
                for j in range(H):
                    dist = math.sqrt(pow(self.data.iloc[i][0]-self.data.iloc[heads[j]][0],2) + pow(self.data.iloc[i][1]-self.data.iloc[heads[j]][1],2))
                    dist = dist + head_to_sink[j]
                    if dist < dmin:
                        dmin = dist
                        chosen_cluster = j
                        distance_to_head = dist - head_to_sink[j]
                distance_to_head_list[i] = distance_to_head
                cluster_index[i] = chosen_cluster

        # print("Inside child chromo", genes, distance_to_head_list, head_to_sink)

        distance_p = sum(distance_to_head_list) + sum(head_to_sink)

        fitness = w * (D - distance_p) + (1-w)*(N-H)

        if distance_p==0:
            accuracy = float('inf')
        else:
            accuracy = pow(D,2) / pow(distance_p,2)
            
        if self.max_acc < accuracy:
            self.final_cluster_heads = []
            for i in range(H):
                self.final_cluster_heads.append([self.data.iloc[heads[i]][0],self.data.iloc[heads[i]][1]])
            self.best_cluster = cluster_index
        
        self.max_fitness = max(self.max_fitness, fitness)
        self.max_acc = max(self.max_acc, accuracy)
        print("Reduction in energy: {0:.2f}%".format(100-(1/accuracy)*100))
        print("Fitness Score: {0:.4f}".format(fitness))
        print("Alloted Clusters:", cluster_index)


    def output_result(self):
        print("--------------------------------------------------------------------")
        print("Max achieved Reduction in Energy {0:.2f}%".format(100-(1/self.max_acc)*100))
        print("Maximum fitness score achieved: {0:.4f}".format(self.max_fitness))
        print("Best Cluster Prediction: ", self.best_cluster)
        print("Cluster Heads: ", self.final_cluster_heads)

        print("----------------------------------------------------------")

        kmeans_cluster(len(self.final_cluster_heads),0)
        
        print("---------------------------------------------------------")

        fig, ax = plt.subplots()
        for i in range(len(self.final_cluster_heads)):
            ax.annotate("head", (self.final_cluster_heads[i][0], self.final_cluster_heads[i][1]))
            plt.plot(self.final_cluster_heads[i][0],self.final_cluster_heads[i][1],marker='^', color='r', markersize=12)
            plt.plot([0,self.final_cluster_heads[i][0],], [0,self.final_cluster_heads[i][1],],color='k',linestyle='dashed',linewidth=0.8)
            for j in range(len(self.best_cluster)):
                if self.best_cluster[j]==i:
                    plt.plot([self.data.iloc[j][0],self.final_cluster_heads[i][0],], [self.data.iloc[j][1],self.final_cluster_heads[i][1],],color='b',linewidth=0.5)


        ax.annotate("sink",(0,0))
        plt.scatter(self.data.iloc[:][0],self.data.iloc[:][1], c=self.best_cluster,cmap='rainbow')
        plt.scatter([0],[0],c=['black'])
        plt.show()