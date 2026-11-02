import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

# Define a simple feed-forward neural network.
model = Sequential(
    [
        Input(shape=(20,), name="input_features"),
        Dense(64, activation="relu", name="hidden_1"),
        Dropout(0.20, name="dropout"),
        Dense(32, activation="relu", name="hidden_2"),
        Dense(1, activation="sigmoid", name="output"),
    ],
    name="example_sequential_model",
)

# Configure the model.
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Print a text summary in the terminal.
model.summary()

# Draw and save the model architecture graph.
# Requires pydot and Graphviz:
#   pip install pydot graphviz
# On many Linux systems also install:
#   sudo apt install graphviz
tf.keras.utils.plot_model(
    model,
    to_file="example_sequential_model.png",
    show_shapes=True,
    show_layer_names=True,
    show_dtype=True,
    expand_nested=True,
    dpi=120,
)

print("Saved model graph to: example_sequential_model.png")