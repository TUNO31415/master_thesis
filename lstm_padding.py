import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import KFold
import numpy as np
from utils import split_train_test, evaluation_metrics
import pandas as pd
from  utils import data_loader
import keras_tuner as kt
import os

def tuned_lstm_padding_n_times_k_fold(X, Y, n=10, k=10):
    eval_results = []
    for num in range(n):
        kf = KFold(n_splits=k, shuffle=True)
        for fold, (train_index, test_index) in enumerate(kf.split(X, Y)):
            print(f"ITERATION : {num} | FOLD : {fold}")
            X_train, y_train, X_test, y_test = split_train_test(X, Y, train_index, test_index)
            tuner = kt.RandomSearch(
                tuning_lstm_model,
                objective='val_loss',
                max_trials=5
                )
            stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            
            # Convert y_train to numpy array
            y_train = np.array(y_train)
            # Reshape X_train for LSTM input (assuming X_train is a list of variable-length sequences)
            X_train_padded = tf.keras.preprocessing.sequence.pad_sequences(X_train, padding='post', dtype='float32')
            X_train_padded = tf.expand_dims(X_train_padded, axis=-1)  # Add a dimension for features

            tuner.search(X_train_padded, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
            
            best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

            model = tuner.hypermodel.build(best_hps)
            # Train the model
            history = model.fit(X_train_padded, y_train, epochs=10, batch_size=32)
            val_acc_per_epoch = history.history['loss']
            best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1

            hypermodel = tuner.hypermodel.build(best_hps)
            hypermodel.fit(X_train_padded, y_train, epochs=best_epoch, batch_size=32)
            # Assuming X_test is your test data in the same format as X_train
            X_test_padded = tf.keras.preprocessing.sequence.pad_sequences(X_test, padding='post', dtype='float32')
            X_test_padded = tf.expand_dims(X_test_padded, axis=-1)

            # Predict on test data
            predictions = hypermodel.predict(X_test_padded)
            eval_results.append(evaluation_metrics(y_test, predictions.squeeze()))
        
    return eval_results

def tuning_lstm_model(hp):
    model = Sequential()
    hp_hiddens = hp.Int('hidden_state', min_value=16, max_value=128, step=16)
    model.add(LSTM(hp_hiddens, input_shape=(None, 1)))  # None for variable sequence length, 1 for dimension of each input vector
    model.add(Dense(1))  # Output layer with 1 neuron for predicting a single value
    model.compile(loss='mean_squared_error', optimizer='adam') # Compile the model
    return model

def create_lstm():
    model = Sequential()
    model.add(LSTM(64, input_shape=(None, 1)))  # None for variable sequence length, 1 for dimension of each input vector
    model.add(Dense(1))  # Output layer with 1 neuron for predicting a single value
    model.compile(loss='mean_squared_error', optimizer='adam') # Compile the model
    return model

def lstm_with_padding_n_times_k_fold(X, Y, n=10, k=10):
    eval_results = []
    for num in range(n):
        kf = KFold(n_splits=k, shuffle=True)
        for fold, (train_index, test_index) in enumerate(kf.split(X, Y)):
            print(f"ITERATION : {num} | FOLD : {fold}")
            X_train, y_train, X_test, y_test = split_train_test(X, Y, train_index, test_index)
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
    
def main():
    dimensions = ["MD", "CI", "FI", "IC", "P"]
    # output_folder = "/Users/taichi/Desktop/master_thesis/results/v6/"
    output_folder = "/tudelft.net/staff-umbrella/tunoMSc2023/paco_dataset/result_lstm_full_res/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    entries = []
    for d in dimensions:
        X, Y = data_loader(d, "/tudelft.net/staff-umbrella/tunoMSc2023/paco_dataset/rtsis_new_prompt/", retrospective_sis_file_path="/tudelft.net/staff-umbrella/tunoMSc2023/paco_dataset/retrospective_sis.csv")
        entries.append(tuned_lstm_padding_n_times_k_fold(X, Y))

    df = pd.DataFrame({
        "Dimension" : dimensions,
        "Model" : "lstm_pad",
        "Results" : entries
    })

    df.to_csv(output_folder + "new_prompt_lstm_pad_full_results.csv.csv")

if __name__ == "__main__":
    main()