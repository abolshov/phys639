#!/usr/bin/env python3
"""Compare a CNN on tf_flowers trained with vs. without data augmentation."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

AUTO = tf.data.AUTOTUNE


def get_datasets(*, img_size, batch_size):
    (train, val, test), info = tfds.load(
        "tf_flowers",
        split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
        as_supervised=True, with_info=True,
    )
    resize = layers.Resizing(img_size, img_size)
    prep = lambda x, y: (resize(x), y)
    train = train.map(prep, AUTO).cache().shuffle(1000).batch(batch_size).prefetch(AUTO)
    val = val.map(prep, AUTO).cache().batch(batch_size).prefetch(AUTO)
    test = test.map(prep, AUTO).cache().batch(batch_size).prefetch(AUTO)
    return train, val, test, info.features["label"].num_classes


def build_model(*, img_size, n_classes, augment, seed):
    tf.keras.utils.set_random_seed(seed)  # same init/shuffling for both variants
    aug = [
        layers.RandomFlip("horizontal", seed=seed),
        layers.RandomRotation(0.1, seed=seed),
        layers.RandomZoom(0.1, seed=seed),
        layers.RandomTranslation(0.1, 0.1, seed=seed),
        layers.RandomContrast(0.1, seed=seed),
    ] if augment else []
    model = models.Sequential(
        [layers.Input((img_size, img_size, 3))] + aug + [layers.Rescaling(1 / 255.0)]
        + [l for f in (32, 64, 128) for l in (
            layers.Conv2D(f, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
        )]
        + [
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dense(n_classes),
        ],
        name="cnn_aug" if augment else "cnn_noaug",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def run(*, label, augment, train_ds, val_ds, test_ds, img_size, n_classes, epochs, seed):
    model = build_model(img_size=img_size, n_classes=n_classes, augment=augment, seed=seed)
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2)
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"[{label}] test loss={loss:.4f} acc={acc:.4f}")
    return hist.history, (loss, acc)


def plot(*, results, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for (label, (hist, _)), color in zip(results.items(), ("C0", "C1")):
        for ax, key, name in ((axes[0], "accuracy", "accuracy"), (axes[1], "loss", "loss")):
            ax.plot(hist[key], color=color, ls="--", label=f"{label} train")
            ax.plot(hist[f"val_{key}"], color=color, label=f"{label} val")
            ax.set_xlabel("epoch"), ax.set_ylabel(name)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_title("Accuracy"), axes[1].set_title("Loss")
    fig.suptitle("tf_flowers CNN: with vs. without data augmentation")
    fig.tight_layout()
    fig.savefig(output)
    print(f"wrote {output}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--img-size", type=int, default=180)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-o", "--output", default="aug_comparison.pdf")
    args = p.parse_args()

    train_ds, val_ds, test_ds, n_classes = get_datasets(
        img_size=args.img_size, batch_size=args.batch_size
    )
    results = {
        label: run(
            label=label, augment=aug, train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
            img_size=args.img_size, n_classes=n_classes, epochs=args.epochs, seed=args.seed,
        )
        for label, aug in (("no aug", False), ("aug", True))
    }
    print("\n=== summary ===")
    for label, (hist, (loss, acc)) in results.items():
        gap = hist["accuracy"][-1] - hist["val_accuracy"][-1]
        print(f"{label:>7}: best val acc={max(hist['val_accuracy']):.4f} "
              f"test acc={acc:.4f} test loss={loss:.4f} train-val gap={gap:+.4f}")
    plot(results=results, output=args.output)


if __name__ == "__main__":
    main()