# import packages
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# reading data
my_data = pd.read_csv("drug200.csv", delimiter=",")
my_data[0:5]