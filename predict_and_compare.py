from GeneticAlgorithm import GeneticAlgorithm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA

#Load DATASET
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



#WITHOUT FEATURE SELECTION
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print("Accuracy without Feature Selection = "+ str(accuracy_score(y_test,predictions)))


#PCA
n_components = 12
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)

X_test_pca = pca.transform(X_test)

logmodel_PCA = LogisticRegression()

logmodel_PCA.fit(X_train_pca, y_train)

predictions = logmodel_PCA.predict(X_test_pca)

PCA_accuracy = accuracy_score(y_test, predictions)

print("Accuracy score after PCA is = " + str(PCA_accuracy))



#GENETIC ALGORITHM
size = 6600  #population size
n_feat = 30  # number of features
n_parents = 40  # number of parents
mutation_rate = 0.75  # mutation rate
n_gen = 200  # the number of generations

gen_algo = GeneticAlgorithm(size=size, n_feat=n_feat, n_parents=40, mutation_rate=0.75, n_gen=200,
                            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

# Evolve the population
best_chromo, best_score = gen_algo.evolve()

# Get accuracy after genetic algorithm
accuracy_after_genetic_algo = gen_algo.train_and_predict()
print("Accuracy score after genetic algorithm is = " + str(accuracy_after_genetic_algo))

