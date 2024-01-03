# Feature Selection Using A Genetic Algorithm


This repository explores the possibility of treating feature selection as a NP-hard travelling salesman problem (TSP). Instead of cities, imagine features as destinations. The goal is to find the most optimal route (set of features) that maximizes performance while minimizing computational cost, just as the TSP seeks the shortest route visiting cities. Both involve exploring numerous combinations to find the best sequence (or set) that satisfies specific criteria.

I used the Breast Cancer Wisconsin (Diagnostic) Data Set to explore this hypothesis. The dataset contains clinical measurements extracted from fine needle aspirates (FNA) of breast masses, aiming to predict whethera tumor is **malignant** or **benign**. The dataset consists of 569 instances with 30 features.

Treating feature selection as a TSP, I implemented a custom genetic algorithm tailored for this task. <br>

Additionally, the study included the use of Principal Component Analysis (PCA) as an alternative feature selection method, serving as a comparative benchmark against the Genetic Algorithm. <br>

Post feature selection, Sklearn's Logistic Regression was employed for evaluation and comparative analysis.

## Results
<div align="center">

| Method                      | Accuracy Score   |
|-----------------------------|------------------|
| Genetic Algorithm           | 0.9912           |
| PCA                         | 0.9474           |
| Without Feature Selection   | 0.9298           |

</div>

## How to run it?
You may install the required packages by

```
pip3 install -r requirements.txt
```

Run `predict_and_compare.py` to reproduce the results of this study.

`GeneticAlgorithm.py` contains the entirety of the Genetic Algorithm (GA).

**Note**: The existing GA parameters in the predict_and_compare.py file are chosen after rigorous finetuning.


I implemented a genetic algorithm-based feature selection technique. It initializes a population of chromosomes representing feature selections, assigns fitness scores using logistic regression, and selects the best individuals for reproduction. The algorithm applies one-point crossover and mutation operations to evolve the population over multiple generations.

We also performed PCA for feature selection to compare our Genetic Algorithm(GA) with. The best score achieved with what seemed the optimal number of components was 94.7%

Anyways after evolving the population with our GA, we selected the best individual and extracted the corresponding features from the training and testing sets. And then trained another logistic regression model using the genetically selected features and achieved an accuracy of **99.1%** with the same basic sklearn LR model.
