#!/usr/bin/env python3
"""Imbalanced binary classification on the credit-card fraud dataset.

Compares three strategies under an identical architecture/init:
  baseline      - plain BCE on the natural 0.17% positive rate
  class_weight  - per-sample loss reweighting (Keras `class_weight`)
  oversample    - 50/50 batches via tf.data.Dataset.sample_from_datasets

Usage:
  python fraud_imbalance.py --seeds 0 1 2 --epochs 30 -o imbalance_comparison.pdf
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

AUTOTUNE = tf.data.AUTOTUNE
URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
STRATEGIES = ("baseline", "class_weight", "oversample")


# ---------------------------------------------------------------- data ------
def load_data(*, seed, clip=5.0):
    df = pd.read_csv(keras.utils.get_file("creditcard.csv", URL))
    df["LogAmount"] = np.log(df.pop("Amount") + 1e-3)   # heavy-tailed -> log
    df.pop("Time")
    y, X = df.pop("Class").to_numpy("float32"), df.to_numpy("float32")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y,
                                          random_state=seed)
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=.2, stratify=ytr,
                                          random_state=seed)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = (np.clip(sc.transform(a), -clip, clip) for a in (Xtr, Xva, Xte))

    neg, pos = np.bincount(ytr.astype(int))
    return dict(Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte,
                neg=int(neg), pos=int(pos), total=int(neg + pos))


# --------------------------------------------------------------- model ------
def metrics():
    return [keras.metrics.AUC(name="roc_auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
            keras.metrics.Precision(name="prec"),
            keras.metrics.Recall(name="rec")]


def make_model(*, n_features, bias=0.0, seed=0, lr=1e-3, dropout=0.3):
    keras.utils.set_random_seed(seed)                   # identical init per strategy
    model = keras.Sequential([
        keras.Input((n_features,)),
        keras.layers.Dense(64, activation="relu"), keras.layers.Dropout(dropout),
        keras.layers.Dense(64, activation="relu"), keras.layers.Dropout(dropout),
        keras.layers.Dense(1, activation="sigmoid",
                           bias_initializer=keras.initializers.Constant(bias)),
    ])
    model.compile(keras.optimizers.Adam(lr), "binary_crossentropy", metrics=metrics())
    return model


def balanced_dataset(*, X, y, batch, seed):
    """Infinite stream of ~50/50 batches (minority oversampled with replacement)."""
    def one_class(mask):
        n = int(mask.sum())
        return (tf.data.Dataset.from_tensor_slices((X[mask], y[mask]))
                .shuffle(n, seed=seed).repeat())
    # alternative (undersamples the majority instead of oversampling the minority):
    #   ds.rejection_resample(lambda x, l: tf.cast(l, tf.int32), target_dist=[.5, .5])
    return (tf.data.Dataset
            .sample_from_datasets([one_class(y == 1), one_class(y == 0)],
                                  weights=[.5, .5], seed=seed)
            .batch(batch).prefetch(AUTOTUNE))


# ------------------------------------------------------------- training -----
def train(*, strategy, data, seed, batch=2048, epochs=30, patience=8, verbose=0):
    bias = float(np.log(data["pos"] / data["neg"]))     # start at the true prior
    stop = keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max",
                                         patience=patience, restore_best_weights=True)
    val = (data["Xva"], data["yva"])
    kw = dict(epochs=epochs, validation_data=val, callbacks=[stop], verbose=verbose)

    if strategy == "oversample":
        model = make_model(n_features=data["Xtr"].shape[1], bias=0.0, seed=seed)
        ds = balanced_dataset(X=data["Xtr"], y=data["ytr"], batch=batch, seed=seed)
        model.fit(ds, steps_per_epoch=int(np.ceil(2 * data["neg"] / batch)), **kw)
        return model, 1.0                               # train pos/neg ratio

    model = make_model(n_features=data["Xtr"].shape[1], bias=bias, seed=seed)
    cw = None
    if strategy == "class_weight":
        cw = {0: data["total"] / (2 * data["neg"]), 1: data["total"] / (2 * data["pos"])}
    model.fit(data["Xtr"], data["ytr"], batch_size=batch, class_weight=cw, **kw)
    return model, 1.0 if cw else data["pos"] / data["neg"]


def recalibrate(p, *, prior, train_ratio):
    """Map scores trained at `train_ratio` pos/neg back onto the true `prior`."""
    odds = np.clip(p, 1e-7, 1 - 1e-7)
    odds = odds / (1 - odds) * prior / train_ratio
    return odds / (1 + odds)


def score(*, name, p, yte, flag_budget):
    thr = np.quantile(p, 1 - flag_budget)               # fixed review capacity
    sel, sig = p >= thr, yte == 1
    return {"strategy": name,
            "pr_auc": average_precision_score(yte, p),
            "roc_auc": roc_auc_score(yte, p),
            "prec@0.5": ((p >= .5) & sig).sum() / max((p >= .5).sum(), 1),
            "rec@0.5": ((p >= .5) & sig).sum() / sig.sum(),
            f"rec@{flag_budget:.1%}flag": (sel & sig).sum() / sig.sum(),
            "mean_pred": p.mean()}


# ---------------------------------------------------------------- plots -----
def plot_curves(*, preds, yte, out):
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(16, 4.5))
    for name, p in preds.items():
        pr, rc, thr = precision_recall_curve(yte, p)
        a0.plot(rc, pr, label=f"{name} (AP={average_precision_score(yte, p):.3f})")
        a1.plot(np.clip(thr, 1e-6, 1 - 1e-6), rc[:-1], label=name)
        fpr, tpr, _ = roc_curve(yte, p)
        a2.plot(np.clip(fpr, 1e-6, None), tpr,
                label=f"{name} (AUC={roc_auc_score(yte, p):.4f})")

    a0.set(xlabel="recall", ylabel="precision", title="Precision-Recall (test)")
    a1.set(xlabel="threshold", ylabel="recall", xscale="log",
            title="operating point shifts, ranking barely moves")
    a2.plot([1e-5, 1], [1e-5, 1], "k--", lw=.8, label="chance")
    a2.set(xlabel="FPR", ylabel="TPR", xscale="log", xlim=(1e-5, 1), ylim=(0, 1),
            title="ROC (log FPR; linear axis hides everything)")
    for ax in (a0, a1, a2):
        ax.legend(loc="lower left" if ax is a1 else "lower right", fontsize=8)
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ----------------------------------------------------------------- main -----
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--flag-budget", type=float, default=0.002,
                    help="fraction of transactions a review team can inspect")
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES),
                    choices=list(STRATEGIES))
    ap.add_argument("-o", "--output", default="imbalance_comparison.pdf")
    ap.add_argument("-v", "--verbose", type=int, default=0)
    args = ap.parse_args()

    rows, preds_first = [], {}
    for seed in args.seeds:
        data = load_data(seed=seed)
        if seed == args.seeds[0]:
            print(f"train: {data['total']} rows, {data['pos']} pos "
                  f"({100 * data['pos'] / data['total']:.3f}%)  "
                  f"test pos: {int(data['yte'].sum())}")

        for strategy in args.strategies:
            model, train_ratio = train(strategy=strategy, data=data, seed=seed,
                                       batch=args.batch, epochs=args.epochs,
                                       patience=args.patience, verbose=args.verbose)
            p = model.predict(data["Xte"], batch_size=args.batch, verbose=0).ravel()
            row = score(name=strategy, p=p, yte=data["yte"],
                        flag_budget=args.flag_budget)
            row["seed"] = seed
            rows.append(row)

            # recalibrated view: undo the artificial training prior
            if train_ratio == 1.0:
                pc = recalibrate(p, prior=data["pos"] / data["neg"], train_ratio=1.0)
                rec = score(name=f"{strategy}+recal", p=pc, yte=data["yte"],
                            flag_budget=args.flag_budget)
                rec["seed"] = seed
                rows.append(rec)

            if seed == args.seeds[0]:
                preds_first[strategy] = p
            keras.backend.clear_session()

    df = pd.DataFrame(rows).set_index(["strategy", "seed"]).sort_index()
    pd.set_option("display.width", 200)
    print("\n--- per seed ---")
    print(df.round(4).to_string())

    if len(args.seeds) > 1:
        agg = df.groupby("strategy").agg(["mean", "std"])
        print(f"\n--- mean +/- std over {len(args.seeds)} seeds ---")
        print(agg.round(4).to_string())
        print("\nnote: compare differences against the per-seed std before "
              "claiming a strategy wins.")

    plot_curves(preds=preds_first, yte=load_data(seed=args.seeds[0])["yte"],
                out=args.output)


if __name__ == "__main__":
    main()