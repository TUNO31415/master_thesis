import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from evaluate import data_loader

# Generate synthetic data (replace with your actual data loading)
X, y = data_loader("MD")

# Sort sequences based on their lengths
sequence_lengths = np.array([len(seq) for seq in X])
sorted_indices = np.argsort(sequence_lengths)
X_sorted = []
for i in sorted_indices:
    X_sorted.append(X[i])
y_sorted = y[sorted_indices]

# Define the LSTM model
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(64, input_shape=(None, 1)))  # LSTM layer with variable input length
    model.add(Dense(1))                          # Output layer with 1 neuron
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

# Perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True)

fold = 0
for train_idx, val_idx in kf.split(X_sorted):
    fold += 1
    print(f"Fold {fold}")

    # Split data into train and validation sets
    X_train, X_val = X_sorted[train_idx], X_sorted[val_idx]
    y_train, y_val = y_sorted[train_idx], y_sorted[val_idx]

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

            # Determine batch size based on the number of sequences with the same length
            batch_size = len(X_batch)
            if batch_size > 0:
                model.fit(X_batch, y_batch, epochs=10, batch_size=batch_size, verbose=0)
            
            batch_start_idx = i
            prev_length = current_length

    # Train on the last batch of sequences with the same length
    X_batch = X_train[batch_start_idx:]
    y_batch = y_train[batch_start_idx:]

    # Determine batch size for the last batch
    last_batch_size = len(X_batch)
    if last_batch_size > 0:
        model.fit(X_batch, y_batch, epochs=10, batch_size=last_batch_size, verbose=0)

    # Evaluate the model on the validation data for this fold
    val_loss = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {val_loss:.4f}")

    # Optionally, you can save the trained model for each fold
    # model.save(f'lstm_model_fold_{fold}.h5')

    print("-" * 40)

# After all folds are completed, you can perform additional analysis or aggregate results
