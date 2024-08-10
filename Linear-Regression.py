
# Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Loading the dataset
data = pd.read_csv('Salary_Data.csv')
X = data.iloc[: , :-1].values # X is predictor
y = data.loc[: , -1].values # Y is response

# Split dataset into test and traing set


# Traing Linear Regression Model


# Predicting test set results


# Visualizing the Training set results

# Visualizing the test set results






                