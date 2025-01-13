import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix

breast_cancer_df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)

breast_cancer_df['target'] = dataset.target

x = breast_cancer_df.iloc[: , :-1]
y = breast_cancer_df.iloc[: , -1]

x_train , x_test , y_train ,y_test = train_test_split(x,y,test_size = 0.3)

mlp = MLPClassifier(hidden_layer_sizes =(5,3),activation='relu',solver='lbfgs' )

mlp.fit(x_train,y_train)

y_pred = mlp.predict(x_test)
print(y_pred)

accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.2f}")

class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

conf_matrix = confusion_matrix(y_test,y_pred)
print(conf_matrix)
