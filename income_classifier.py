#!/usr/bin/env python3
"""
Train a TensorFlow classifier on the UCI Adult Income dataset.

The model predicts whether annual income exceeds $50,000.

Demonstrated features
---------------------
1. Keras Functional API
2. Numerical and categorical preprocessing
3. Optional normalization of numerical features
4. tf.data pipelines
5. Training callbacks
6. Saving and restoring a complete model
7. Binary-classification metrics and diagnostic plots

All 14 predictor columns from the Adult dataset are used.

Examples
--------
Train with numerical normalization:

    python income_classifier.py --normalize-numeric

Train without numerical normalization:

    python income_classifier.py \
        --no-normalize-numeric \
        --output-dir adult_income_unnormalized
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


# ----------------------------------------------------------------------
# Dataset configuration
# ----------------------------------------------------------------------

TRAIN_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "adult/adult.data"
)

TEST_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "adult/adult.test"
)

COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

NUMERIC_FEATURES = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "income"


# ----------------------------------------------------------------------
# Command-line arguments
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train a TensorFlow classifier on the UCI Adult Income dataset."
        )
    )

    parser.add_argument(
        "--train-csv",
        type=Path,
        default=None,
        help=(
            "Optional local adult.data file. If omitted, the official "
            "training file is downloaded."
        ),
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=None,
        help=(
            "Optional local adult.test file. If omitted, the official "
            "test file is downloaded."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("adult_income_output"),
        help="Output directory (default: adult_income_output).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size (default: 256).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.0e-3,
        help="Initial Adam learning rate (default: 0.001).",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help=(
            "Fraction of the official training data used for validation "
            "(default: 0.20)."
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        help="Early-stopping patience in epochs (default: 7).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--normalize-numeric",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Normalize numerical features using training-set statistics. "
            "Use --no-normalize-numeric to disable normalization."
        ),
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be positive")

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")

    if not 0.0 < args.validation_fraction < 1.0:
        parser.error(
            "--validation-fraction must be greater than 0 and less than 1"
        )

    if args.patience <= 0:
        parser.error("--patience must be positive")

    return args


# ----------------------------------------------------------------------
# Data loading and cleaning
# ----------------------------------------------------------------------


def resolve_data_file(
    *,
    local_path: Path | None,
    url: str,
    filename: str,
) -> Path:
    """Return a local data path, downloading the file when necessary."""

    if local_path is not None:
        if not local_path.is_file():
            raise FileNotFoundError(
                f"Input file does not exist: {local_path}"
            )

        return local_path

    downloaded_path = tf.keras.utils.get_file(
        fname=filename,
        origin=url,
    )

    return Path(downloaded_path)


def read_adult_file(
    path: Path,
    *,
    is_test: bool,
) -> pd.DataFrame:
    """Read an official UCI Adult data file."""

    frame = pd.read_csv(
        path,
        names=COLUMN_NAMES,
        header=None,
        skiprows=1 if is_test else 0,
        skipinitialspace=True,
        na_values=["?"],
        comment="|",
    )

    # adult.test labels have a trailing period, for example ">50K.".
    income_strings = (
        frame[TARGET]
        .astype("string")
        .str.strip()
        .str.rstrip(".")
    )

    frame[TARGET] = income_strings.map(
        {
            "<=50K": 0.0,
            ">50K": 1.0,
        }
    )

    # Remove blank or malformed lines for which no target is available.
    frame = frame.dropna(subset=[TARGET]).reset_index(drop=True)

    return frame


def clean_frames(
    training_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean training and test data.

    Numerical missing-value replacements are calculated using only the
    official training sample. Missing categorical values are represented
    by the explicit category "Unknown".
    """

    training_frame = training_frame.copy()
    test_frame = test_frame.copy()

    numeric_medians: dict[str, float] = {}

    for feature in NUMERIC_FEATURES:
        training_values = pd.to_numeric(
            training_frame[feature],
            errors="coerce",
        )

        numeric_medians[feature] = float(training_values.median())

    for frame in [training_frame, test_frame]:
        for feature in NUMERIC_FEATURES:
            frame[feature] = (
                pd.to_numeric(frame[feature], errors="coerce")
                .fillna(numeric_medians[feature])
                .astype(np.float32)
            )

        for feature in CATEGORICAL_FEATURES:
            frame[feature] = (
                frame[feature]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
                .replace("", "Unknown")
            )

        frame[TARGET] = frame[TARGET].astype(np.float32)

    return training_frame, test_frame


def load_data(
    *,
    train_path: Path | None,
    test_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean the official Adult training and test samples."""

    resolved_train_path = resolve_data_file(
        local_path=train_path,
        url=TRAIN_URL,
        filename="adult.data",
    )

    resolved_test_path = resolve_data_file(
        local_path=test_path,
        url=TEST_URL,
        filename="adult.test",
    )

    print(f"Training file: {resolved_train_path}")
    print(f"Test file:     {resolved_test_path}")

    raw_training_frame = read_adult_file(
        resolved_train_path,
        is_test=False,
    )

    raw_test_frame = read_adult_file(
        resolved_test_path,
        is_test=True,
    )

    return clean_frames(
        raw_training_frame,
        raw_test_frame,
    )


def stratified_train_validation_split(
    frame: pd.DataFrame,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a stratified training/validation split."""

    rng = np.random.default_rng(seed)

    training_indices: list[int] = []
    validation_indices: list[int] = []

    for label in sorted(frame[TARGET].unique()):
        class_indices = frame.index[
            frame[TARGET] == label
        ].to_numpy()

        rng.shuffle(class_indices)

        validation_size = int(
            round(validation_fraction * len(class_indices))
        )

        validation_indices.extend(
            class_indices[:validation_size].tolist()
        )
        training_indices.extend(
            class_indices[validation_size:].tolist()
        )

    rng.shuffle(training_indices)
    rng.shuffle(validation_indices)

    training_frame = (
        frame.loc[training_indices]
        .reset_index(drop=True)
    )

    validation_frame = (
        frame.loc[validation_indices]
        .reset_index(drop=True)
    )

    return training_frame, validation_frame


# ----------------------------------------------------------------------
# TensorFlow input handling
# ----------------------------------------------------------------------


def dataframe_to_tensors(
    frame: pd.DataFrame,
) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
    """
    Convert a DataFrame to explicitly typed TensorFlow tensors.

    Explicitly constructing tf.string tensors prevents categorical pandas
    columns from reaching Keras as NumPy arrays with dtype=object.
    """

    features: dict[str, tf.Tensor] = {}

    for feature in NUMERIC_FEATURES:
        values = (
            frame[feature]
            .to_numpy(dtype=np.float32)
            .reshape(-1, 1)
        )

        features[feature] = tf.convert_to_tensor(
            values,
            dtype=tf.float32,
        )

    for feature in CATEGORICAL_FEATURES:
        # Converting through a Python list avoids a NumPy object array.
        values = (
            frame[feature]
            .astype(str)
            .tolist()
        )

        values_tensor = tf.constant(
            values,
            dtype=tf.string,
        )

        features[feature] = tf.reshape(
            values_tensor,
            shape=(-1, 1),
        )

    label_values = (
        frame[TARGET]
        .to_numpy(dtype=np.float32)
        .reshape(-1, 1)
    )

    labels = tf.convert_to_tensor(
        label_values,
        dtype=tf.float32,
    )

    return features, labels


def make_dataset(
    frame: pd.DataFrame,
    *,
    batch_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    """
    Construct a tf.data training or evaluation pipeline.

    The pipeline demonstrates:
      - from_tensor_slices
      - cache
      - shuffle
      - batch
      - prefetch
    """

    features, labels = dataframe_to_tensors(frame)

    dataset = tf.data.Dataset.from_tensor_slices(
        (features, labels)
    )

    dataset = dataset.cache()

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(frame),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


def make_prediction_dataset(
    frame: pd.DataFrame,
    *,
    batch_size: int,
) -> tf.data.Dataset:
    """
    Construct a feature-only prediction dataset.

    This avoids passing dtype=object NumPy arrays to model.predict().
    """

    features, _ = dataframe_to_tensors(frame)

    dataset = tf.data.Dataset.from_tensor_slices(features)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ----------------------------------------------------------------------
# Model construction
# ----------------------------------------------------------------------


def build_model(
    training_frame: pd.DataFrame,
    *,
    learning_rate: float,
    normalize_numeric: bool,
) -> tf.keras.Model:
    """
    Build and compile the Adult Income classifier.

    All preprocessing layers are included in the model and are therefore
    saved together with the neural network.
    """

    inputs: dict[str, tf.keras.Input] = {}
    encoded_features: list[tf.Tensor] = []

    # ------------------------------------------------------------------
    # Numerical inputs
    # ------------------------------------------------------------------

    numeric_inputs: list[tf.Tensor] = []

    for feature in NUMERIC_FEATURES:
        feature_input = tf.keras.Input(
            shape=(1,),
            name=feature,
            dtype=tf.float32,
        )

        inputs[feature] = feature_input
        numeric_inputs.append(feature_input)

    numeric_values = tf.keras.layers.Concatenate(
        name="numeric_features",
    )(numeric_inputs)

    if normalize_numeric:
        numeric_preprocessing = tf.keras.layers.Normalization(
            name="numeric_normalization",
        )

        # Adapt only on the actual training split.
        numeric_preprocessing.adapt(
            training_frame[NUMERIC_FEATURES].to_numpy(
                dtype=np.float32
            )
        )

        encoded_numeric_values = numeric_preprocessing(
            numeric_values
        )

    else:
        encoded_numeric_values = tf.keras.layers.Activation(
            activation="linear",
            name="unnormalized_numeric_features",
        )(numeric_values)

    encoded_features.append(encoded_numeric_values)

    # ------------------------------------------------------------------
    # Categorical inputs
    # ------------------------------------------------------------------

    for feature in CATEGORICAL_FEATURES:
        feature_input = tf.keras.Input(
            shape=(1,),
            name=feature,
            dtype=tf.string,
        )

        inputs[feature] = feature_input

        vocabulary = sorted(
            training_frame[feature]
            .astype(str)
            .unique()
            .tolist()
        )

        encoder = tf.keras.layers.StringLookup(
            vocabulary=vocabulary,
            output_mode="one_hot",
            num_oov_indices=1,
            name=f"{feature}_one_hot",
        )

        encoded_feature = encoder(feature_input)
        encoded_features.append(encoded_feature)

    # ------------------------------------------------------------------
    # Dense classifier
    # ------------------------------------------------------------------

    all_encoded_features = tf.keras.layers.Concatenate(
        name="all_encoded_features",
    )(encoded_features)

    x = tf.keras.layers.Dense(
        units=128,
        activation="relu",
        name="hidden_1",
    )(all_encoded_features)

    x = tf.keras.layers.Dropout(
        rate=0.25,
        name="dropout_1",
    )(x)

    x = tf.keras.layers.Dense(
        units=64,
        activation="relu",
        name="hidden_2",
    )(x)

    x = tf.keras.layers.Dropout(
        rate=0.15,
        name="dropout_2",
    )(x)

    outputs = tf.keras.layers.Dense(
        units=1,
        activation="sigmoid",
        name="high_income_probability",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="adult_income_classifier",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            tf.keras.metrics.AUC(
                name="auc"
            ),
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            ),
        ],
    )

    return model


# ----------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------


def make_callbacks(
    *,
    output_dir: Path,
    patience: int,
) -> list[tf.keras.callbacks.Callback]:
    """Create training callbacks."""

    checkpoint_path = output_dir / "best_model.keras"

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=max(2, patience // 3),
            min_lr=1.0e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------


def evaluate_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    *,
    title: str,
) -> dict[str, float]:
    """Evaluate a model and print its metrics."""

    results = model.evaluate(
        dataset,
        return_dict=True,
        verbose=0,
    )

    print()
    print(title)
    print("-" * len(title))

    for metric_name, value in results.items():
        print(f"{metric_name:>10}: {value:.6f}")

    return results


def calculate_confusion_matrix(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    """Calculate a binary confusion matrix."""

    labels = labels.reshape(-1).astype(np.int32)

    predictions = (
        probabilities.reshape(-1) >= threshold
    ).astype(np.int32)

    true_negative = np.sum(
        (labels == 0) & (predictions == 0)
    )

    false_positive = np.sum(
        (labels == 0) & (predictions == 1)
    )

    false_negative = np.sum(
        (labels == 1) & (predictions == 0)
    )

    true_positive = np.sum(
        (labels == 1) & (predictions == 1)
    )

    return np.array(
        [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
        dtype=np.int64,
    )


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------


def plot_diagnostics(
    *,
    history: tf.keras.callbacks.History,
    labels: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
) -> None:
    """Plot learning curves, classifier scores, and confusion matrix."""

    labels = labels.reshape(-1).astype(np.int32)
    probabilities = probabilities.reshape(-1)

    confusion_matrix = calculate_confusion_matrix(
        labels,
        probabilities,
    )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(11, 8),
    )

    # Loss history
    axes[0, 0].plot(
        history.history["loss"],
        label="Training",
    )

    axes[0, 0].plot(
        history.history["val_loss"],
        label="Validation",
    )

    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Binary cross-entropy")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # AUC history
    axes[0, 1].plot(
        history.history["auc"],
        label="Training",
    )

    axes[0, 1].plot(
        history.history["val_auc"],
        label="Validation",
    )

    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("ROC AUC")
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].set_title("AUC")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Classifier output
    bins = np.linspace(0.0, 1.0, 31)

    axes[1, 0].hist(
        probabilities[labels == 0],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label="Income <=50K",
    )

    axes[1, 0].hist(
        probabilities[labels == 1],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label="Income >50K",
    )

    axes[1, 0].axvline(
        0.5,
        color="black",
        linestyle="--",
        label="Threshold = 0.5",
    )

    axes[1, 0].set_xlabel("Predicted P(income > $50K)")
    axes[1, 0].set_ylabel("Normalized people")
    axes[1, 0].set_title("Test-set classifier output")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Confusion matrix
    image = axes[1, 1].imshow(
        confusion_matrix,
        cmap="Blues",
    )

    class_names = ["<=50K", ">50K"]

    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_xticklabels(class_names)
    axes[1, 1].set_yticklabels(class_names)
    axes[1, 1].set_xlabel("Predicted class")
    axes[1, 1].set_ylabel("True class")
    axes[1, 1].set_title("Test-set confusion matrix")

    maximum = max(int(confusion_matrix.max()), 1)

    for row in range(2):
        for column in range(2):
            value = int(confusion_matrix[row, column])

            axes[1, 1].text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > maximum / 2
                    else "black"
                ),
                fontsize=12,
            )

    figure.colorbar(
        image,
        ax=axes[1, 1],
        fraction=0.046,
        pad=0.04,
    )

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def print_dataset_information(
    *,
    training_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    normalize_numeric: bool,
) -> None:
    """Print dataset and preprocessing information."""

    print()
    print("Adult Income dataset")
    print("====================")
    print(f"Numerical features:   {NUMERIC_FEATURES}")
    print(f"Categorical features: {CATEGORICAL_FEATURES}")
    print(f"Number of input features: {len(ALL_FEATURES)}")
    print(f"Numerical normalization: {normalize_numeric}")
    print()

    samples = [
        ("Training", training_frame),
        ("Validation", validation_frame),
        ("Test", test_frame),
    ]

    for name, frame in samples:
        positive_fraction = float(frame[TARGET].mean())

        print(
            f"{name:10s}: "
            f"{len(frame):6d} examples, "
            f">50K fraction = {positive_fraction:.3f}"
        )


def print_example_predictions(
    *,
    model: tf.keras.Model,
    frame: pd.DataFrame,
    batch_size: int,
    number_of_examples: int = 10,
) -> None:
    """Print predictions for several test examples."""

    number_of_examples = min(
        number_of_examples,
        len(frame),
    )

    example_frame = (
        frame.iloc[:number_of_examples]
        .copy()
        .reset_index(drop=True)
    )

    # Use a feature-only tf.data dataset. This guarantees that categorical
    # inputs reach Keras with dtype=tf.string rather than dtype=object.
    prediction_dataset = make_prediction_dataset(
        example_frame,
        batch_size=batch_size,
    )

    probabilities = model.predict(
        prediction_dataset,
        verbose=0,
    ).reshape(-1)

    print()
    print("Example predictions")
    print("===================")

    for index, probability in enumerate(probabilities):
        row = example_frame.iloc[index]

        predicted_class = (
            ">50K"
            if probability >= 0.5
            else "<=50K"
        )

        true_class = (
            ">50K"
            if float(row[TARGET]) == 1.0
            else "<=50K"
        )

        print()
        print(f"Person {index + 1}")

        print(
            f"  age={int(row['age'])}, "
            f"education={row['education']}, "
            f"occupation={row['occupation']}, "
            f"hours/week={int(row['hours_per_week'])}"
        )

        print(
            f"  P(income > $50K): {probability:.3f}"
        )
        print(f"  predicted:        {predicted_class}")
        print(f"  true:             {true_class}")


# ----------------------------------------------------------------------
# Main program
# ----------------------------------------------------------------------


def main() -> None:
    """Run the complete training and evaluation workflow."""

    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tf.keras.utils.set_random_seed(args.seed)

    complete_training_frame, test_frame = load_data(
        train_path=args.train_csv,
        test_path=args.test_csv,
    )

    training_frame, validation_frame = (
        stratified_train_validation_split(
            complete_training_frame,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
    )

    print_dataset_information(
        training_frame=training_frame,
        validation_frame=validation_frame,
        test_frame=test_frame,
        normalize_numeric=args.normalize_numeric,
    )

    # ------------------------------------------------------------------
    # Feature 5: tf.data input pipelines
    # ------------------------------------------------------------------

    training_dataset = make_dataset(
        training_frame,
        batch_size=args.batch_size,
        training=True,
        seed=args.seed,
    )

    validation_dataset = make_dataset(
        validation_frame,
        batch_size=args.batch_size,
        training=False,
        seed=args.seed,
    )

    test_dataset = make_dataset(
        test_frame,
        batch_size=args.batch_size,
        training=False,
        seed=args.seed,
    )

    # A feature-only dataset is used for prediction.
    test_prediction_dataset = make_prediction_dataset(
        test_frame,
        batch_size=args.batch_size,
    )

    model = build_model(
        training_frame,
        learning_rate=args.learning_rate,
        normalize_numeric=args.normalize_numeric,
    )

    print()
    model.summary()

    # ------------------------------------------------------------------
    # Feature 6: callbacks
    # ------------------------------------------------------------------

    callbacks = make_callbacks(
        output_dir=args.output_dir,
        patience=args.patience,
    )

    history = model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    evaluate_model(
        model,
        test_dataset,
        title="Original-model test results",
    )

    # ------------------------------------------------------------------
    # Feature 7: save and restore the complete model
    # ------------------------------------------------------------------

    final_model_path = args.output_dir / "final_model.keras"

    model.save(final_model_path)

    print()
    print(f"Saved complete model to: {final_model_path}")

    restored_model = tf.keras.models.load_model(
        final_model_path
    )

    evaluate_model(
        restored_model,
        test_dataset,
        title="Restored-model test results",
    )

    # Both calls use a tf.data prediction dataset. No NumPy object arrays
    # are passed directly to Keras.
    original_probabilities = model.predict(
        test_prediction_dataset,
        verbose=0,
    )

    restored_probabilities = restored_model.predict(
        test_prediction_dataset,
        verbose=0,
    )

    predictions_match = np.allclose(
        original_probabilities,
        restored_probabilities,
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    maximum_prediction_difference = float(
        np.max(
            np.abs(
                original_probabilities
                - restored_probabilities
            )
        )
    )

    print()
    print(
        "Original and restored predictions match: "
        f"{predictions_match}"
    )

    print(
        "Maximum prediction difference: "
        f"{maximum_prediction_difference:.3e}"
    )

    # Labels are explicitly converted to a numerical NumPy array.
    test_labels = test_frame[TARGET].to_numpy(
        dtype=np.float32
    )

    diagnostics_path = (
        args.output_dir / "training_diagnostics.pdf"
    )

    plot_diagnostics(
        history=history,
        labels=test_labels,
        probabilities=restored_probabilities,
        output_path=diagnostics_path,
    )

    print(f"Saved diagnostics to: {diagnostics_path}")

    print_example_predictions(
        model=restored_model,
        frame=test_frame,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
