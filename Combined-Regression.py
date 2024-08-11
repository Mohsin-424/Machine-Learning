# Importing Libraries

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Loading the dataset
data = pd.read_csv('Salary_Data.csv')
X = data.iloc[:, :-1].values  # X is predictor
y = data.iloc[:, -1].values  # Y is response

# Split dataset into test and training set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# Training Linear Regression Model

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting test set results

y_predicted = regressor.predict(X_test)

# Visualizing both Training and Test set results
plt.figure(figsize=(10, 6))

# Training set results
plt.scatter(X_train, y_train, color='red', label='Training set')
plt.plot(X_train, regressor.predict(X_train), color='green', label='Regression line (Training set)')

# Test set results
plt.scatter(X_test, y_test, color='blue', label='Test set')
plt.plot(X_test, y_predicted, color='black', linestyle='dashed', label='Regression line (Test set)')

plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()
