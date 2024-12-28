import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("AAPL.csv")

# Preprocess the dataset
# Strip leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Convert 'Close/Last' to numeric (remove '$' and cast to float)
data['Close/Last'] = data['Close/Last'].replace({'\$': ''}, regex=True).astype(float)

# Ensure 'Volume' is numeric (remove commas if necessary)
if data['Volume'].dtype != 'int64' and data['Volume'].dtype != 'float64':
    data['Volume'] = data['Volume'].astype(str).str.replace(',', '').astype(int)

# Use only the 'Close/Last' column for simplicity
close_prices = data['Close/Last']

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = scaler.fit_transform(close_prices.values.reshape(-1, 1))

# Define sequence length (how many past steps to use as input)
seq_length = 10

# Prepare the data for LSTM (sliding window approach)
X = []
y = []

for i in range(seq_length, len(close_prices_scaled)):
    X.append(close_prices_scaled[i-seq_length:i])  # Last `seq_length` values as features
    y.append(close_prices_scaled[i])  # The next value as the target

X = np.array(X)
y = np.array(y)

# Reshape X to be 3D for LSTM: (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(96, return_sequences=True),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dense(16, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Make predictions
y_pred = model.predict(X_test)

# Rescale predictions and actual values back to original scale
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate Mean Squared Error
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
print(f"Mean Squared Error: {mse}")
