# Breast_Cancer_Classifier
#Optimizing Feature Selection for Breast Cancer Classification Using An Evolutionary Algorithm

Using sklearn's simple Logistic Regression method results in a classification accuracy of 92.9%.

The dataset has 30 features. The curse of dimensionality really plays a part when we have this many features.

So I devised and fine tuned an evolutionary algorithm for feature selection. 

After the algorithm returned its carefully selected features, I ran sklearn's simple Logistic Regression method again on the selected features, and it achieved an accuracy of 99.1%!
