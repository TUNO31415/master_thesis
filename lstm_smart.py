import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from utils import split_train_test, evaluation_metrics

# Define the LSTM model
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(64, input_shape=(None, 1)))  # LSTM layer with variable input length
    model.add(Dense(1))                          # Output layer with 1 neuron
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model


def lstm_smart_n_times_k_fold(X, y, n=10, k=10, epochs=10, batch_size=32):
    eval_results = []
    
    for num in range(n):
        kf = KFold(n_splits=k, shuffle=True)

        for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            print(f"ITERATION : {num} | FOLD : {fold}")
            
            X_train, y_train, X_val, y_val = split_train_test(X, y, train_idx, test_idx)

            # Sort training data based on sequence lengths
            sorted_indices = np.argsort([len(seq) for seq in X_train])
            X_train = [X_train[i] for i in sorted_indices]
            y_train = [y_train[i] for i in sorted_indices]

            # Create and compile a new LSTM model for each fold
            model = create_lstm_model()

            # Train the model in batches of sequences with the same length
            prev_length = len(X_train[0]) if len(X_train) > 0 else 0
            batch_X, batch_y = [], []

            for i in range(len(X_train)):
                current_length = len(X_train[i])
                
                if current_length != prev_length:
                    # Train on the previous batch of sequences with the same length
                    train_sequences(model, batch_X, batch_y, epochs=epochs)
                    batch_X, batch_y = [], []
                    
                batch_X.append(X_train[i])
                batch_y.append(y_train[i])
                prev_length = current_length

            # Train on the last batch of sequences
            if batch_X:
                train_sequences(model, batch_X, batch_y, epochs=epochs)

            # Evaluate the model on the validation data for this fold
            predictions = [model.predict(np.expand_dims(x, axis=0))[0][0] for x in X_val]
            eval_results.append(evaluation_metrics(y_val, predictions))

    return eval_results

def train_sequences(model, X_batch, y_batch, epochs=10):
    batch_size = len(X_batch)
    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)
    model.fit(X_batch, y_batch, epochs=epochs, batch_size=batch_size, verbose=0)