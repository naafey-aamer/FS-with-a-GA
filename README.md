# Breast Cancer Diagnosis with Logistic Regression and Genetic Algorithm Feature Selection

#Optimizing Feature Selection for Breast Cancer Classification Using An Evolutionary Algorithm

This notebook explores the diagnosis of breast cancer using the Breast Cancer Wisconsin (Diagnostic) Data Set. The dataset contains clinical measurements extracted from fine needle aspirates (FNA) of breast masses, aiming to predict the presence of malignant or benign tumors. The dataset consists of 569 instances with 30 features.

Using sklearn's basic Logistic Regression method on all 30 features results in a classification accuracy of 92.9%.

Since the dataset has 30 features, The curse of dimensionality becomes significant

To enhance the logistic regression model's performance, I implemented a genetic algorithm-based feature selection technique. It initializes a population of chromosomes representing feature selections, assigns fitness scores using logistic regression, and selects the best individuals for reproduction. The algorithm applies one-point crossover and mutation operations to evolve the population over multiple generations.

We also performed PCA for feature selection to compare our Genetic Algorithm(GA) with. The best score achieved with what seemed the optimal number of components was 94.7%

Anyways after evolving the population with our GA, we selected the best individual and extracted the corresponding features from the training and testing sets. And then trained another logistic regression model using the genetically selected features and achieved an accuracy of **99.1%** with the same basic sklearn LR model.
