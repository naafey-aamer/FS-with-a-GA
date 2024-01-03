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


## Overview of the Genetic Algorithm (GA)

  **Initialization of Population**:
        The algorithm begins by creating an initial population of potential feature subsets. Each subset is represented as a binary array, where each bit corresponds to the inclusion or exclusion of a feature.

  **Fitness Calculation**:
        The fitness of each subset is evaluated using logistic regression as a classifier. The subsets act as feature selectors, and the model's accuracy on a held-out test set determines their fitness. Higher accuracy indicates a better-performing subset.

  **Selection**:
        Stochastic Universal Sampling (SUS) method is used for selection. Individuals (subsets) with higher fitness have a higher chance of being selected for reproduction, mimicking the concept of "survival of the fittest."

  **Crossover**:
        Selected subsets undergo crossover, where parts of one subset are exchanged with another to create new subsets. This process introduces diversity into the population by combining characteristics of successful subsets.

  **Mutation**:
        To maintain genetic diversity, some bits in the subsets are randomly flipped (mutated) with a certain probability. This operation helps in exploring new solutions.

  **Evolution**:
        Through multiple generations (defined by n_gen), the algorithm iterates over selection, crossover, and mutation, gradually improving the subsets' fitness..

Initialized with a population of 6600, and a mutation rate of 0.6 for high exploration, the population was set to evolve for 200 generations.
<img src="https://github.com/naafey-aamer/Breast_Cancer_Classifier/blob/main/images/end_GA.png" alt="image" width="400"> <br>

## Contact

This was just a personal project to explore my developing interest in heuristic algorithms and optimization in general.

To discuss further or to contribute, reach out at: naafey.aamer@gmail.com
