# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# 1. Load the data from CSV
data = pd.read_csv('AAPL.csv')

# 2. Strip leading and trailing spaces from column names
data.columns = data.columns.str.strip()

# 3. Remove the dollar sign and convert 'Close/Last' to float
data['Close/Last'] = data['Close/Last'].replace({'\$': ''}, regex=True).astype(float)

# 4. Remove commas and convert 'Volume' to integer
data['Volume'] = data['Volume'].astype(str).str.replace(',', '').astype(int)

# 5. Extract the features and target variables
X = data[["Volume"]]  # Feature: Volume
y = data["Close/Last"]  # Target: Closing Price

# 6. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Create and train the Decision Tree Regressor
model = DecisionTreeRegressor(max_depth=2, random_state=42)
model.fit(X_train, y_train)

# 8. Make predictions
y_pred = model.predict(X_test)

# 9. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# 10. Get feature importance (optional)
feature_importances = model.feature_importances_
print(f"Feature Importances: {feature_importances}")
