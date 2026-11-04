#!/usr/bin/env python3
"""Compare a CIFAR-10 CNN trained with and without data augmentation.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, mixed_precision, models


CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def get_splits(seed):
    """Load CIFAR-10 and create deterministic train/val/test splits."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.reshape(-1).astype(np.int32)
    y_test = y_test.reshape(-1).astype(np.int32)

    # Carve a validation set out of the training data.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x_train))
    x_train, y_train = x_train[perm], y_train[perm]

    n_val = int(0.1 * len(x_train))
    splits = (
        (x_train[n_val:], y_train[n_val:]),
        (x_train[:n_val], y_train[:n_val]),
        (x_test, y_test),
    )

    print(f"Dataset: CIFAR-10 (built in, {x_train.shape[1:]} images)")
    print(f"Classes: {list(CLASSES)}")
    print(f"Images: {len(x_train) + len(x_test)}; "
          f"split: {[len(x) for x, _ in splits]}")
    return splits


def make_dataset(split, *, img_size, batch_size, training, seed):
    """Construct a GPU-friendly tf.data pipeline from in-memory arrays."""
    images, labels = split
    ds = tf.data.Dataset.from_tensor_slices((images, labels))

    if training:
        ds = ds.shuffle(
            len(images),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def load(image, label):
        image = tf.image.resize(image, (img_size, img_size))
        return tf.cast(image, tf.uint8), label  # stay small until the model

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(*, img_size, augment, seed):
    """Build and compile one CNN variant."""
    tf.keras.utils.set_random_seed(seed)

    augmentation = [
        layers.RandomFlip("horizontal", seed=seed + 1),
        layers.RandomRotation(0.1, seed=seed + 2),
        layers.RandomZoom(0.1, seed=seed + 3),
        layers.RandomTranslation(0.1, 0.1, seed=seed + 4),
        layers.RandomContrast(0.1, seed=seed + 5),
    ] if augment else []

    convolution = [
        layer
        for filters in (32, 64, 128)
        for layer in (
            layers.Conv2D(filters, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
        )
    ]

    model = models.Sequential([
        layers.Input((img_size, img_size, 3)),
        *augmentation,
        layers.Rescaling(1 / 255.0),  # also casts uint8 -> float
        *convolution,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3, seed=seed + 6),
        layers.Dense(128, activation="relu"),
        # Keep logits in float32 for numerical stability under mixed precision.
        layers.Dense(len(CLASSES), dtype="float32"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        ],
    )
    return model


def run(label, augment, splits, args):
    """Train and evaluate one model."""
    datasets = [
        make_dataset(
            split,
            img_size=args.img_size,
            batch_size=args.batch_size,
            training=(i == 0),
            seed=args.seed,
        )
        for i, split in enumerate(splits)
    ]
    train, val, test = datasets

    tf.keras.backend.clear_session()
    model = build_model(
        img_size=args.img_size, augment=augment, seed=args.seed
    )

    print(f"\n=== {label} ===")
    history = model.fit(
        train, validation_data=val, epochs=args.epochs,
        verbose=2, shuffle=False,
    ).history
    loss, accuracy = model.evaluate(test, verbose=0)

    print(f"{label}: test loss={loss:.4f}, accuracy={accuracy:.4f}")
    return history, (loss, accuracy)


def plot(results, output):
    """Plot training and validation accuracy and loss."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for (label, (history, _)), color in zip(results.items(), ("C0", "C1")):
        for ax, key in zip(axes, ("accuracy", "loss")):
            ax.plot(history[key], "--", color=color,
                    label=f"{label} train")
            ax.plot(history[f"val_{key}"], color=color,
                    label=f"{label} val")

    for ax, title in zip(axes, ("Accuracy", "Loss")):
        ax.set(title=title, xlabel="Epoch", ylabel=title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("CIFAR-10: with vs. without data augmentation")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    print(f"Wrote {output}")


def configure_gpu(*, no_gpu, no_mixed_precision):
    """Set up the Tesla T4 (memory growth + mixed precision)."""
    gpus = tf.config.list_physical_devices("GPU")

    if no_gpu or not gpus:
        tf.config.set_visible_devices([], "GPU")
        print("Running on CPU.")
        return

    for gpu in gpus:
        # Avoid grabbing all 16 GB up front; play nice on shared nodes.
        tf.config.experimental.set_memory_growth(gpu, True)

    if not no_mixed_precision:
        # T4 has FP16 tensor cores; mixed_float16 is a free speedup.
        mixed_precision.set_global_policy("mixed_float16")

    names = [g.name for g in gpus]
    print(f"Running on GPU(s): {names}; "
          f"policy={mixed_precision.global_policy().name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gpu", action="store_true",
                        help="force CPU even if a GPU is available")
    parser.add_argument("--no-mixed-precision", action="store_true",
                        help="disable mixed_float16 on the GPU")
    parser.add_argument("-o", "--output", default="aug_comparison.pdf")
    args = parser.parse_args()

    configure_gpu(
        no_gpu=args.no_gpu,
        no_mixed_precision=args.no_mixed_precision,
    )
    tf.keras.utils.set_random_seed(args.seed)

    print(f"TensorFlow {tf.__version__}; GPUs: "
          f"{tf.config.get_visible_devices('GPU')}")

    splits = get_splits(args.seed)
    results = {
        label: run(label, augment, splits, args)
        for label, augment in (("no aug", False), ("aug", True))
    }

    print("\n=== Summary ===")
    for label, (h, (loss, accuracy)) in results.items():
        gap = h["accuracy"][-1] - h["val_accuracy"][-1]
        print(f"{label:>7}: best val={max(h['val_accuracy']):.4f}, "
              f"test={accuracy:.4f}, loss={loss:.4f}, gap={gap:+.4f}")

    plot(results, args.output)


if __name__ == "__main__":
    main()