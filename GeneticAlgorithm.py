import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings

class GeneticAlgorithm:
    def __init__(self, size, n_feat, n_parents, mutation_rate, n_gen, X_train, X_test, y_train, y_test):
        warnings.filterwarnings("ignore")
        self.size = size
        self.n_feat = n_feat
        self.n_parents = n_parents
        self.mutation_rate = mutation_rate
        self.n_gen = n_gen
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_chromo = None
        self.best_score = 0

    def initialization_of_population(self):
        population = []
        for i in range(self.size):
            chromosome = np.ones(self.n_feat, dtype=bool)
            chromosome[:int(0.45 * self.n_feat)] = False
            np.random.shuffle(chromosome)
            population.append(chromosome)
        return population

    def fitness_score(self, population):
        scores = []
        for chromosome in population:
            clf = LogisticRegression()
            clf.fit(self.X_train.iloc[:, chromosome], self.y_train)
            predictions = clf.predict(self.X_test.iloc[:, chromosome])
            score = accuracy_score(self.y_test, predictions)
            scores.append([score, chromosome])
        scores.sort(key=lambda x: x[0], reverse=True)
        sorted_scores, sorted_population = zip(*scores)
        return sorted_scores, sorted_population

    def selection(self, pop_after_fit1):
        population_nextgen = []
        total_fitness = sum(individual[0] for individual in pop_after_fit1)
        distance = total_fitness / self.n_parents
        start_offset = random.uniform(0, distance)
        pointer = start_offset
        index = 0
        accumulated_fitness = 0

        for _ in range(self.n_parents):
            while accumulated_fitness < pointer:
                accumulated_fitness += pop_after_fit1[index][0]
                index = (index + 1) % len(pop_after_fit1)

            population_nextgen.append(pop_after_fit1[index][1])
            pointer += distance

        return population_nextgen

    def crossover(self, pop_after_sel):
        population_nextgen = []
        for i in range(len(pop_after_sel)):
            child = pop_after_sel[i]
            child_copy = child.copy()
            child_copy[3:12] = pop_after_sel[(i + 1) % len(pop_after_sel)][3:12]
            population_nextgen.append(child_copy)
            population_nextgen.append(child)
        return population_nextgen

    def mutation(self, population):
        mutated_population = []
        for chromosome in population:
            mutated_chromosome = chromosome.copy()
            for i in range(len(mutated_chromosome)):
                if random.random() < self.mutation_rate:
                    mutated_chromosome[i] = not mutated_chromosome[i]
            mutated_population.append(mutated_chromosome)
        return mutated_population

    def evolve(self):
        population_nextgen = self.initialization_of_population()
        for i in range(self.n_gen):
            print("----------------------------------")
            scores, pop_after_fit = self.fitness_score(population_nextgen)
            print(i, "'s FITTEST SPECIMEN ", scores[0])
            if scores[0] > self.best_score and scores[0] != 1.0:
                self.best_chromo = pop_after_fit[0]
                self.best_score = scores[0]
            combined_list = list(zip(scores, pop_after_fit))
            pop_after_sel = self.selection(combined_list)
            pop_after_cross = self.crossover(pop_after_sel)
            population_nextgen = self.mutation(pop_after_cross)
            print("BEST_SPECIMEN OVERALL = ", self.best_score)
        return [self.best_chromo, self.best_score]

    def get_best_individual(self):
        return self.best_chromo

    def select_features(self):
        best_individual = self.get_best_individual()
        genetically_selected_train = self.X_train.iloc[:, best_individual]
        genetically_selected_test = self.X_test.iloc[:, best_individual]
        return genetically_selected_train, genetically_selected_test

    def train_and_predict(self):
        genetically_selected_train, genetically_selected_test = self.select_features()
        logmodel1 = LogisticRegression()
        logmodel1.fit(genetically_selected_train, self.y_train)
        predictions = logmodel1.predict(genetically_selected_test)
        accuracy = accuracy_score(self.y_test, predictions)
        return accuracy


#SAMPLE USAGE

size = 6600  #population size
n_feat = 30  # number of features
n_parents = 40  # number of parents
mutation_rate = 0.75  # mutation rate
n_gen = 200  # the number of generations

cancer = load_breast_cancer()
df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
labels = pd.DataFrame(cancer['target'])
df.shape

labels=cancer["target"]
label_names=cancer["target_names"]

#splitting the model into training and testing set
X_train, X_test, y_train, y_test = train_test_split(df,
                                                    labels, test_size=0.2,
                                                    random_state=20)

# Example usage:
# Initialize the GeneticAlgorithm object
gen_algo = GeneticAlgorithm(size=size, n_feat=n_feat, n_parents=40, mutation_rate=0.75, n_gen=200,
                            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

# Evolve the population
best_chromo, best_score = gen_algo.evolve()

# Get accuracy after genetic algorithm
accuracy_after_genetic_algo = gen_algo.train_and_predict()
print("Accuracy score after genetic algorithm is = " + str(accuracy_after_genetic_algo))
