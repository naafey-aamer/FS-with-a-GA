import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


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

n_components = 12
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)

X_test_pca = pca.transform(X_test)

logmodel_PCA = LogisticRegression()

logmodel_PCA.fit(X_train_pca, y_train)

predictions = logmodel_PCA.predict(X_test_pca)

PCA_accuracy = accuracy_score(y_test, predictions)

print("Accuracy =", PCA_accuracy)