from main import ProteinAnalyzer
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelBinarizer
from keras.regularizers import L2
from keras.optimizers import Adam
from keras.layers import (
    Input,
    Dense,
    Dropout,
    Conv1D,
    Flatten,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Embedding,
    Concatenate,
)
from keras.models import Model


PA = ProteinAnalyzer(data_file="preprocessed_data.json")

structure = np.load("structure.npy")
sequences = np.array(list(PA.data["sequences"].values)) / 21
numerical = np.array(list(PA.data[PA.col_names].values))
labels = np.array(list(PA.data["label"].values))

PA.data = 0


def enhance_dataset(array, numerical=False):
    v1, v2, v3, v4 = 1.0, 1.0, 1.0, 1.0
    if numerical:
        v1, v2, v3, v4 = 0.5, 1.8, 0.5, 1.3
    array = np.concatenate((array, v1 * array), axis=0)
    array = np.concatenate((array, v2 * array), axis=0)
    array = np.concatenate((array, v3 * array), axis=0)
    #array = np.concatenate((array, v4 * array), axis=0) # Not enough RAM :(
    return array


structure = enhance_dataset(structure, numerical=True)
sequences = enhance_dataset(sequences, numerical=False)
numerical = enhance_dataset(numerical, numerical=True)
labels = enhance_dataset(labels, numerical=False)

split = 0.1

structure, structure_t = train_test_split(structure, test_size=split, random_state=42)
sequences, sequences_t = train_test_split(sequences, test_size=split, random_state=42)
numerical, numerical_t = train_test_split(numerical, test_size=split, random_state=42)
label, label_t = train_test_split(labels, test_size=split, random_state=42)


# Display the sizes of train, test, and validation sets
print("\nTrain set size:", len(label))
print("\nTest set size:", len(label_t))

# Perform label binarization for multi-class classification
label_binarizer = LabelBinarizer()
label = label_binarizer.fit_transform(label)
label_t = label_binarizer.transform(label_t)


# Build the neural network model

structure_input = Input(shape=structure.shape[1:])
conv_layer1 = Conv1D(
    128, kernel_size=2, activation="relu", kernel_regularizer=L2(0.01)
)(structure_input)
conv_layer2 = Conv1D(128, kernel_size=2, activation="relu")(conv_layer1)
flat_layer = Flatten()(conv_layer2)
dropout1 = dropout_layer = Dropout(0.5)(flat_layer)


sequences_input = Input(shape=sequences.shape[1:])
embedding_layer = Embedding(input_dim=21, output_dim=128)(sequences_input)
lstm_layer = Bidirectional(LSTM(64, recurrent_regularizer=L2(0.01), activation="relu"))(
    embedding_layer
)
sequences_dense = Dense(256, activation="relu")(lstm_layer)
dropout2 = dropout_layer = Dropout(0.5)(sequences_dense)


numerical_input = Input(shape=numerical.shape[1:])
numerical_dense = Dense(64, activation="relu")(numerical_input)
dropout3 = dropout_layer = Dropout(0.5)(numerical_dense)


concat_layer = Concatenate()([dropout1, dropout2, dropout3])
dense_layer1 = Dense(1024, activation="relu")(dropout_layer)
dense_layer2 = Dense(512, activation="tanh")(dense_layer1)
output_layer = Dense(10, activation="sigmoid")(dense_layer2)

# Assuming label is your output, and you should use the correct output layer for it
model = Model(
    inputs=[structure_input, sequences_input, numerical_input], outputs=output_layer
)
print(model.summary())


# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

checkpoint_path = "training/all_data_classifier.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1, save_best_only=True
)

# Fit the model
# history = model.fit(
#     [structure, sequences, numerical],
#     label,
#     epochs=1000,
#     batch_size=100,
#     validation_split=0.2,
#     callbacks=[cp_callback],
# )

# # Plot training history
# def plot_history(history):
#     # Plot training & validation accuracy values
#     plt.plot(history.history['accuracy'])
#     plt.plot(history.history['val_accuracy'])
#     plt.title('Model accuracy')
#     plt.grid()
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend(['Train', 'Validation'], loc='upper left')
#     plt.savefig('training/model-accuracy.png')
#     plt.show()

#     # Plot training & validation loss values
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('Model loss')
#     plt.grid()
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend(['Train', 'Validation'], loc='upper left')
#     plt.savefig('training/model-loss.png')
#     plt.show()

# # Call the function to plot training history
# plot_history(history)


model.load_weights(checkpoint_path)

# Re-evaluate the model
loss, acc = model.evaluate([structure_t, sequences_t, numerical_t], label_t, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

model.save("training/all_data_classifier.keras")
