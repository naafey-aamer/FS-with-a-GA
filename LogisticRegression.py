import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score


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

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print("Accuracy = "+ str(accuracy_score(y_test,predictions)))
