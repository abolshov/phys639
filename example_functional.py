#!/usr/bin/env python3
"""Construct, summarize, and draw the example network using the Keras Functional API."""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model


def build_model(*, n_features: int = 20) -> tf.keras.Model:
    """Build the same binary-classification model using the Functional API."""
    inputs = Input(shape=(n_features,), name="input_features")

    x = Dense(64, activation="relu", name="hidden_1")(inputs)
    x = Dropout(0.20, name="dropout")(x)
    x = Dense(32, activation="relu", name="hidden_2")(x)

    outputs = Dense(1, activation="sigmoid", name="output")(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="example_functional_model",
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main() -> None:
    model = build_model(n_features=20)

    # Print the terminal summary.
    model.summary()

    # Save a graphical representation of the architecture.
    # Requires:
    #   pip install pydot graphviz
    # and usually a system Graphviz installation, e.g.:
    #   sudo apt install graphviz
    tf.keras.utils.plot_model(
        model,
        to_file="example_functional_model.png",
        show_shapes=True,
        show_layer_names=True,
        show_dtype=True,
        expand_nested=True,
        dpi=120,
    )

    print("\nSaved model graph to: example_functional_model.png")


if __name__ == "__main__":
    main()