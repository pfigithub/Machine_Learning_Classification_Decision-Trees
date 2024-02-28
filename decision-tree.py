# import packages
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# reading data
my_data = pd.read_csv("drug200.csv", delimiter=",")
my_data[0:5]

# seprating x and y(base on our data)
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
y = my_data["Drug"]