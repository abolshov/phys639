#!/usr/bin/env python3
"""Build, summarize, and plot a multi-input/multi-output Functional Keras model."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


N_CONTINUOUS_FEATURES = 12
N_JET_FLAVOURS = 6
N_EVENT_CATEGORIES = 4


def build_model() -> keras.Model:
    """Create a multi-input, multi-output model without adapting preprocessing."""

    # =========================================================================
    # Inputs
    # =========================================================================
    continuous_inputs = keras.Input(
        shape=(N_CONTINUOUS_FEATURES,),
        dtype=tf.float32,
        name="continuous_features",
    )

    jet_flavour_inputs = keras.Input(
        shape=(1,),
        dtype=tf.int32,
        name="jet_flavour",
    )

    event_category_inputs = keras.Input(
        shape=(1,),
        dtype=tf.int32,
        name="event_category",
    )

    # =========================================================================
    # Continuous branch
    #
    # The Normalization layer is included in the graph, but is not adapted here.
    # Before training, call:
    #
    #   model.get_layer("continuous_normalization").adapt(training_features)
    #
    # using only the continuous training features.
    # =========================================================================
    x_continuous = layers.Normalization(
        axis=-1,
        name="continuous_normalization",
    )(continuous_inputs)

    x_continuous = layers.Dense(
        64,
        activation="relu",
        name="continuous_dense_1",
    )(x_continuous)

    x_continuous = layers.BatchNormalization(
        name="continuous_batch_norm_1",
    )(x_continuous)

    x_continuous = layers.Dropout(
        0.20,
        name="continuous_dropout_1",
    )(x_continuous)

    x_continuous = layers.Dense(
        32,
        activation="relu",
        name="continuous_dense_2",
    )(x_continuous)

    # =========================================================================
    # Categorical branches
    #
    # Inputs must contain integer category indices:
    #   jet_flavour:    0 ... N_JET_FLAVOURS - 1
    #   event_category: 0 ... N_EVENT_CATEGORIES - 1
    # =========================================================================
    x_jet_flavour = layers.Embedding(
        input_dim=N_JET_FLAVOURS,
        output_dim=4,
        name="jet_flavour_embedding",
    )(jet_flavour_inputs)

    x_jet_flavour = layers.Flatten(
        name="jet_flavour_flatten",
    )(x_jet_flavour)

    x_event_category = layers.Embedding(
        input_dim=N_EVENT_CATEGORIES,
        output_dim=3,
        name="event_category_embedding",
    )(event_category_inputs)

    x_event_category = layers.Flatten(
        name="event_category_flatten",
    )(x_event_category)

    # =========================================================================
    # Merge all branches and build a shared trunk.
    # =========================================================================
    x = layers.Concatenate(name="merge_all_features")(
        [
            x_continuous,
            x_jet_flavour,
            x_event_category,
        ]
    )

    x = layers.Dense(64, activation="relu", name="shared_dense_1")(x)
    x = layers.Dropout(0.25, name="shared_dropout_1")(x)

    # Residual block:
    # output = ReLU(residual + dense_2(ReLU(dense_1(residual))))
    residual = layers.Dense(32, activation="relu", name="residual_input")(x)

    x_residual = layers.Dense(
        32,
        activation="relu",
        name="residual_dense_1",
    )(residual)

    x_residual = layers.Dense(
        32,
        activation=None,
        name="residual_dense_2",
    )(x_residual)

    x = layers.Add(name="residual_add")([residual, x_residual])
    x = layers.Activation("relu", name="residual_activation")(x)

    # =========================================================================
    # Classification output branch
    # =========================================================================
    x_classification = layers.Dense(
        16,
        activation="relu",
        name="classification_dense",
    )(x)

    signal_probability = layers.Dense(
        1,
        activation="sigmoid",
        name="signal_probability",
    )(x_classification)

    # =========================================================================
    # Regression output branch
    # =========================================================================
    x_regression = layers.Dense(
        16,
        activation="relu",
        name="regression_dense",
    )(x)

    target_mass = layers.Dense(
        1,
        activation="linear",
        name="target_mass",
    )(x_regression)

    return keras.Model(
        inputs={
            "continuous_features": continuous_inputs,
            "jet_flavour": jet_flavour_inputs,
            "event_category": event_category_inputs,
        },
        outputs={
            "signal_probability": signal_probability,
            "target_mass": target_mass,
        },
        name="multi_input_multi_output_model",
    )


def main() -> None:
    model = build_model()

    # Print the terminal architecture summary.
    model.summary()

    # Save a PNG diagram of the Functional graph.
    #
    # Python dependencies:
    #   pip install pydot graphviz
    #
    # System dependency, for example on Ubuntu/Debian:
    #   sudo apt install graphviz
    keras.utils.plot_model(
        model,
        to_file="multi_input_multi_output_model.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        expand_nested=True,
        dpi=150,
    )

    print("\nSaved model graph to: multi_input_multi_output_model.png")


if __name__ == "__main__":
    main()