import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from evaluate import data_loader
import tensorflow as tf

# Generate synthetic data (replace with your actual data loading)
X, y = data_loader("MD")

# Sort sequences based on their lengths
sequence_lengths = np.array([len(seq) for seq in X])
sorted_indices = np.argsort(sequence_lengths)
X_sorted = []
for i in sorted_indices:
    X_sorted.append(X[i])
y_np = np.array(y)
y_sorted = y_np[sorted_indices].tolist()

# Define the LSTM model
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(64, input_shape=(None, 1)))  # LSTM layer with variable input length
    model.add(Dense(1))                          # Output layer with 1 neuron
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

def split_train_test(X, y, train_ids, test_ids):
    y_np = np.array(y)
    X_train = []
    y_train = y_np[train_ids].tolist()
    for i in train_ids:
        X_train.append(X[i])
    
    X_test = []
    y_test = y_np[test_ids].tolist()
    for i in test_ids:
        X_test.append(X[i])

    return X_train, y_train, X_test, y_test
    

# Perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True)

fold = 0
for train_idx, test_idx in kf.split(X_sorted):
    fold += 1
    print(f"Fold {fold}")

    # Split data into train and validation sets
    X_train, y_train, X_val, y_val = split_train_test(X_sorted, y_sorted, train_idx, test_idx)

    # Create and compile a new LSTM model for each fold
    model = create_lstm_model()

    # Train the model on batches of sequences with the same length
    prev_length = len(X_train[0]) if len(X_train) > 0 else 0
    batch_start_idx = 0

    for i in range(1, len(X_train)):
        current_length = len(X_train[i])
        if current_length != prev_length:
            # Train on the previous batch of sequences with the same length
            X_batch = X_train[batch_start_idx:i]
            y_batch = y_train[batch_start_idx:i]
            # X_batch = np.array(X_batch)
            batch_size = len(X_batch)
            X_batch =  tf.convert_to_tensor(X_batch, dtype=tf.float32)
            y_batch = np.array(y_batch)
            # Determine batch size based on the number of sequences with the same length
            if batch_size > 0:
                model.fit(X_batch, y_batch, epochs=10, batch_size=batch_size, verbose=0)
            
            batch_start_idx = i
            prev_length = current_length

    # Train on the last batch of sequences with the same length
    X_batch = X_train[batch_start_idx:]
    y_batch = y_train[batch_start_idx:]
    last_batch_size = len(X_batch)
    # Determine batch size for the last batch
    X_batch =  tf.convert_to_tensor(X_batch, dtype=tf.float32)
    y_batch = np.array(y_batch)
    if last_batch_size > 0:
        model.fit(X_batch, y_batch, epochs=10, batch_size=last_batch_size, verbose=0)

    # X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
    # X_val = np.array(X_val)
    y_val = np.array(y_val)
    # Evaluate the model on the validation data for this fold
    predictions = []
    for x in X_val:
        x = np.array(x)
        x = np.expand_dims(x, axis=-1)
        pred = model.predict(x)
        predictions.append(model.predict(x)[0][0])
    predictions = np.array(predictions)

    mse = np.mean((predictions - y_val) ** 2)  # Calculate Mean Squared Error
    print("Mean Squared Error:", mse)

# After all folds are completed, you can perform additional analysis or aggregate results
