import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data = pd.read_csv("AAPL.csv") 
data.head()

# Strip leading and trailing spaces from column names
data.columns = data.columns.str.strip()

# Remove the dollar sign and convert 'Close/Last' to float
data['Close/Last'] = data['Close/Last'].replace({'\$': ''}, regex=True).astype(float)

# Remove commas and convert 'Volume' to integer
data['Volume'] = data['Volume'].astype(str).str.replace(',', '').astype(int)

# Extract the features and target variables
close_prices = data['Close/Last']
volume = data['Volume']

# Correcting the target column name
X = data[["Volume"]]  # Feature: Volume
y = data["Close/Last"]  # Target: Closing Price
# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# 7. Get model parameters
intercept = model.intercept_
coefficient = model.coef_[0] 
print(f"Intercept: {intercept}")
print(f"Coefficient: {coefficient}")