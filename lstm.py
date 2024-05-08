import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from evaluate import data_loader
from sklearn.model_selection import KFold
import numpy as np

def create_lstm():
    model = Sequential()
    model.add(LSTM(64, input_shape=(None, 1)))  # None for variable sequence length, 1 for dimension of each input vector
    model.add(Dense(1))  # Output layer with 1 neuron for predicting a single value

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

X, Y = data_loader("MD")
kf = KFold(n_splits=10, shuffle=True)

for fold, (train_index, test_index) in enumerate(kf.split(X, Y)):
    X_train = []
    y_np = np.array(Y)
    y_train = y_np[train_index].tolist()
    for i in train_index:
        X_train.append(X[i])
    
    X_test = []
    for i in test_index:
        X_test.append(X[i])
    y_test = y_np[test_index].tolist()

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

    mse = np.mean((predictions.squeeze() - y_test) ** 2)  # Calculate Mean Squared Error
    print("Mean Squared Error:", mse)

