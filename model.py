# Install Libararies
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers

# House data (X-- Area in Marlas, Portions, Age(Years),   Y-- Price(Millions))
X = np.array([
    [5,2,0],
    [5,1,0],
    [8,2,5],
    [10,1,8],
    [5,3,3],
    [6,2,2],
    [10,2,0],
    [8,2,0],
    [20,2,0],
    [20,2,5]
], dtype=np.float32)

# Labels are the house prices in $1000s
y = np.array([
    25,
    20,
    25,
    20,
    25,
    24,
    40,
    35,
    70,
    50
], dtype=np.float32)

# Normalize 
X_mean = np.mean(X, axis=0)  
X_std = np.std(X, axis=0)    
X_norm = (X - X_mean) / X_std

y_mean = np.mean(y)
y_std = np.std(y)
y_norm = (y - y_mean) / y_std

# Split Dataset
num_samples = X.shape[0]
train_size = int(0.8 * num_samples)  # 80% for training
val_size = int(0.1 * num_samples)    # 10% for validation
      # Remaining 10% for test set

      # Training data
X_train = X_norm[:train_size]
y_train = y_norm[:train_size]

      # Validation data (used to tune hyperparameters, prevent overfitting)
X_val = X_norm[train_size:train_size + val_size]
y_val = y_norm[train_size:train_size + val_size]

      # Test data (unseen during training and validation — final model evaluation)
X_test = X_norm[train_size + val_size:]
y_test = y_norm[train_size + val_size:]

# Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],),
                          kernel_regularizer=regularizers.l2(0.01)),

    Dense(32, activation='relu',
                          kernel_regularizer=regularizers.l2(0.01)),

    Dense(1)
])

# Compile the Model
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])

# Train the Model
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=2,
                    validation_data=(X_val, y_val),
                    verbose=0)

# Evaluate Model on Test Data
test_loss, test_mae = model.evaluate(X_test, y_test)
test_mae_denorm = test_mae * y_std
print(f"Test MAE (denormalized): Rs {test_mae_denorm:.2f}M")

# Plotting Graph
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()

# Predict New House Price ( Marlas, Portion, Age)
new_house = np.array([[3, 2, 0]], dtype=np.float32)
new_house_norm = (new_house - X_mean) / X_std
predicted_price_norm = model.predict(new_house_norm)
predicted_price = predicted_price_norm[0][0] * y_std + y_mean
print(f"Predicted price for new house: Rs {predicted_price:.2f}M")


# Evaluate Model on Training Data
train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
print(f"Train Loss: {train_loss:.4f}")
print(f"Train MAE (denormalized): Rs {train_mae * y_std:.2f}M")

# Error Analysis
y_pred_norm = model.predict(X_test)
y_pred = y_pred_norm.flatten() * y_std + y_mean
y_true = y_test * y_std + y_mean
errors = y_true - y_pred
for i, (true_price, pred_price, error) in enumerate(zip(y_true, y_pred, errors)):
    print(f"Test sample {i}: True price = Rs {true_price:.2f}M, Predicted = Rs {pred_price:.2f}M, Error = Rs {error:.2f}M")

import joblib

# Save model
model.save('model/house_price_model.h5')

# Save normalization data
joblib.dump((X_mean, X_std, y_mean, y_std), 'model/preprocessing.pkl')
