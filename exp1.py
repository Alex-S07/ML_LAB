import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error , r2_score
#loading dataset
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns = housing.feature_names)
data["price"] = housing.target
#select one feature
x = data[['AveRooms']].values
y = data['price'].values
#split data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
#feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
w = 0
b = 0
learning_rate = 0.01
epochs = 1000
n = len(X_train_scaled)
for i in range(epochs):
    y_pred = w * X_train_scaled.flatten() + b
    dw = (1/n) * np.sum((y_pred - y_train)* X_train_scaled.flatten())
    db = (1/n) * np.sum(y_pred - y_train)
    w = w - learning_rate * dw
    b = b - learning_rate * db
    if i % 100 == 0:
        cost = (1 / (2*n)) * np.sum((y_pred - y_train) ** 2)
        print(f"Epoch {i}, Cost = {cost:.4f}")
y_pred_gd = w * X_test_scaled.flatten() + b
print("Gradient Descent")
print("-----------------")
print( "Weights: ", w)
print( "Bias: ", b)
print("MSE: ", mean_squared_error(y_test, y_pred_gd))
print("R2 Score: ", r2_score(y_test, y_pred_gd))
#normal Equation
X_train_ne = np.c_[np.ones((len(X_train),1)),X_train]
X_test_ne = np.c_[np.ones((len(X_test),1)),X_test]
theta = np.linalg.inv(X_train_ne.T @ X_train_ne) @ X_train_ne.T @ y_train
y_pred_ne = X_test_ne @ theta
print("Normal Equation")
print("---------------")
print("---------------")
print("Intercept :", theta[0])
print("slope : " , theta[1])
print("MSE :", mean_squared_error(y_test, y_pred_ne))
print("r2 Score : ", r2_score(y_test,y_pred_ne))
# Plot
plt.figure(figsize=(8,6))
plt.scatter(X_test_scaled, y_test, color='blue', alpha=0.5, label='Actual Data')
# Sort the x-values so the regression line is smooth
sorted_idx = np.argsort(X_test_scaled.flatten())
plt.plot(
    X_test_scaled.flatten()[sorted_idx],
    y_pred_gd[sorted_idx],
    color='red',
    linewidth=2,
    label='Gradient Descent'
)
plt.plot(
    X_test_scaled.flatten()[sorted_idx],
    y_pred_ne[sorted_idx],
    color='green',
    linewidth=2,
    linestyle='--',
    label='Normal Equation'
)
plt.xlabel("Average Rooms (Scaled)")
plt.ylabel("House Price")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()
