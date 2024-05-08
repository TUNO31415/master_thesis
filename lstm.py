import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from evaluate import data_loader
from sklearn.model_selection import KFold
import numpy as np
from evaluate import split_train_test, evaluation_metrics

def create_lstm():
    model = Sequential()
    model.add(LSTM(64, input_shape=(None, 1)))  # None for variable sequence length, 1 for dimension of each input vector
    model.add(Dense(1))  # Output layer with 1 neuron for predicting a single value
    model.compile(loss='mean_squared_error', optimizer='adam') # Compile the model
    return model

def lstm_with_padding_n_times_k_fold(X, Y, n=10, k=10):
    kf = KFold(n_splits=k, shuffle=True)
    eval_results = []
    for num in range(n):
        for fold, (train_index, test_index) in enumerate(kf.split(X, Y)):
            print(f"ITERATION : {num} | FOLD : {fold}")
            X_train, y_train, X_test, y_test = split_train_test(X, Y)
            # Define the LSTM model
            model = create_lstm()

            # Reshape X_train for LSTM input (assuming X_train is a list of variable-length sequences)
            X_train_padded = tf.keras.preprocessing.sequence.pad_sequences(X_train, padding='post', dtype='float32')
            X_train_padded = tf.expand_dims(X_train_padded, axis=-1)  # Add a dimension for features
            
            # Convert y_train to numpy array
            y_train = np.array(y_train)

            # Train the model
            model.fit(X_train_padded, y_train, epochs=10, batch_size=32)

            # Assuming X_test is your test data in the same format as X_train
            X_test_padded = tf.keras.preprocessing.sequence.pad_sequences(X_test, padding='post', dtype='float32')
            X_test_padded = tf.expand_dims(X_test_padded, axis=-1)

            # Predict on test data
            predictions = model.predict(X_test_padded)
            eval_results.append(evaluation_metrics(y_test, predictions.squeeze()))
        
    return eval_results
    

