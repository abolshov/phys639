import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Activation function
# ============================================================

def sigmoid(z):
    """
    Numerically stable sigmoid.

    Works with both scalars and NumPy arrays.
    """
    z = np.asarray(z, dtype=np.float64)
    result = np.empty_like(z)

    positive = z >= 0

    # Stable expression for non-negative values
    result[positive] = 1.0 / (1.0 + np.exp(-z[positive]))

    # Stable expression for negative values
    exp_z = np.exp(z[~positive])
    result[~positive] = exp_z / (1.0 + exp_z)

    return result


# ============================================================
# 2. Forward pass
# ============================================================

def forward(x, y, w, b):
    """
    Forward pass for a scalar sigmoid classifier.

    Model
    -----
        z_i = w*x_i + b
        a_i = sigmoid(z_i)

    Loss
    ----
        loss = mean((a_i - y_i)^2)

    Parameters
    ----------
    x : ndarray, shape (n_samples,)
        Input values.
    y : ndarray, shape (n_samples,)
        Binary target values.
    w : float
        Weight parameter.
    b : float
        Bias parameter.

    Returns
    -------
    loss : float
        Mean squared error.
    cache : dict
        Values needed during backpropagation.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    z = w * x + b
    a = sigmoid(z)

    loss = np.mean((a - y)**2)

    cache = {
        "x": x,
        "y": y,
        "w": w,
        "b": b,
        "z": z,
        "a": a,
    }

    return loss, cache


# ============================================================
# 3. Backward pass
# ============================================================

def backward(cache):
    """
    Compute analytic gradients using backpropagation.

    For each sample:

        L = mean((a - y)^2)
        a = sigmoid(z)
        z = w*x + b

    The factor 1/N comes from taking the mean over samples.

    Returns
    -------
    gradients : dict
        Gradients with respect to w, b, and x.
    """
    x = cache["x"]
    y = cache["y"]
    w = cache["w"]
    a = cache["a"]

    n_samples = len(x)

    # --------------------------------------------------------
    # Loss:
    #
    #     L = (1/N) sum_i (a_i - y_i)^2
    #
    # Therefore:
    #
    #     dL/da_i = (2/N) (a_i - y_i)
    # --------------------------------------------------------
    dL_da = (2.0 / n_samples) * (a - y)

    # --------------------------------------------------------
    # Sigmoid:
    #
    #     a_i = sigmoid(z_i)
    #
    #     da_i/dz_i = a_i * (1 - a_i)
    # --------------------------------------------------------
    da_dz = a * (1.0 - a)

    # Chain rule
    dL_dz = dL_da * da_dz

    # --------------------------------------------------------
    # Affine operation:
    #
    #     z_i = w*x_i + b
    #
    #     dz_i/dw = x_i
    #     dz_i/db = 1
    #     dz_i/dx_i = w
    # --------------------------------------------------------
    dL_dw = np.sum(dL_dz * x)
    dL_db = np.sum(dL_dz)
    dL_dx = dL_dz * w

    gradients = {
        "w": float(dL_dw),
        "b": float(dL_db),
        "x": dL_dx,
    }

    return gradients


# ============================================================
# 4. Predictions and accuracy
# ============================================================

def predict_proba(x, w, b):
    """
    Return the probability-like sigmoid outputs.
    """
    x = np.asarray(x, dtype=np.float64)
    return sigmoid(w * x + b)


def predict(x, w, b, threshold=0.5):
    """
    Convert sigmoid outputs into binary predictions.
    """
    probabilities = predict_proba(x, w, b)
    return (probabilities >= threshold).astype(int)


def accuracy(y_true, y_pred):
    """
    Fraction of correctly classified samples.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(y_true == y_pred)


# ============================================================
# 5. Numerical gradient checking
# ============================================================

def numerical_derivative(function, value, eps=1e-5):
    """
    Centered finite-difference derivative.
    """
    return (
        function(value + eps) - function(value - eps)
    ) / (2.0 * eps)


def relative_error(a, b, eps=1e-12):
    """
    Relative difference between two scalar values.
    """
    return abs(a - b) / max(eps, abs(a) + abs(b))


def check_gradients(x, y, w, b, eps=1e-5):
    """
    Compare analytic gradients to finite-difference gradients.
    """
    loss, cache = forward(x, y, w, b)
    analytic_gradients = backward(cache)

    # Loss as a function of w, keeping b fixed
    def loss_as_function_of_w(w_value):
        current_loss, _ = forward(
            x=x,
            y=y,
            w=w_value,
            b=b,
        )
        return current_loss

    # Loss as a function of b, keeping w fixed
    def loss_as_function_of_b(b_value):
        current_loss, _ = forward(
            x=x,
            y=y,
            w=w,
            b=b_value,
        )
        return current_loss

    numerical_dw = numerical_derivative(
        loss_as_function_of_w,
        value=w,
        eps=eps,
    )

    numerical_db = numerical_derivative(
        loss_as_function_of_b,
        value=b,
        eps=eps,
    )

    analytic_dw = analytic_gradients["w"]
    analytic_db = analytic_gradients["b"]

    print("Gradient check")
    print("=" * 65)
    print(f"Loss: {loss:.10f}")
    print()
    print(
        f"{'Parameter':>10s}"
        f"{'Analytic':>18s}"
        f"{'Numerical':>18s}"
        f"{'Relative error':>18s}"
    )
    print("-" * 65)

    print(
        f"{'w':>10s}"
        f"{analytic_dw:18.10f}"
        f"{numerical_dw:18.10f}"
        f"{relative_error(analytic_dw, numerical_dw):18.3e}"
    )

    print(
        f"{'b':>10s}"
        f"{analytic_db:18.10f}"
        f"{numerical_db:18.10f}"
        f"{relative_error(analytic_db, numerical_db):18.3e}"
    )

    print("=" * 65)


# ============================================================
# 6. Training function
# ============================================================

def train(
    x,
    y,
    initial_w,
    initial_b,
    learning_rate=0.5,
    n_steps=1000,
    print_every=100,
):
    """
    Train the scalar sigmoid classifier using gradient descent.

    Returns
    -------
    w : float
        Final weight.
    b : float
        Final bias.
    history : dict
        Loss, accuracy, parameters, and gradient values.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    w = float(initial_w)
    b = float(initial_b)

    history = {
        "step": [],
        "loss": [],
        "accuracy": [],
        "w": [],
        "b": [],
        "gradient_w": [],
        "gradient_b": [],
    }

    for step in range(n_steps + 1):

        # Forward pass
        loss, cache = forward(x, y, w, b)

        # Backward pass
        gradients = backward(cache)

        # Evaluate classification accuracy
        predictions = predict(x, w, b)
        current_accuracy = accuracy(y, predictions)

        # Save quantities before updating the parameters
        history["step"].append(step)
        history["loss"].append(loss)
        history["accuracy"].append(current_accuracy)
        history["w"].append(w)
        history["b"].append(b)
        history["gradient_w"].append(gradients["w"])
        history["gradient_b"].append(gradients["b"])

        if step % print_every == 0 or step == n_steps:
            print(
                f"step={step:4d} | "
                f"loss={loss:.6f} | "
                f"accuracy={current_accuracy:.3f} | "
                f"w={w:+.5f} | "
                f"b={b:+.5f} | "
                f"dL/dw={gradients['w']:+.5f} | "
                f"dL/db={gradients['b']:+.5f}"
            )

        # Do not perform another update after the final evaluation
        if step == n_steps:
            break

        # Gradient-descent parameter update
        w = w - learning_rate * gradients["w"]
        b = b - learning_rate * gradients["b"]

    # Convert lists to arrays for convenient plotting
    for key in history:
        history[key] = np.asarray(history[key])

    return w, b, history


# ============================================================
# 7. Create a small toy dataset
# ============================================================

# The input contains one feature per sample.
#
# Negative values mostly belong to class 0.
# Positive values mostly belong to class 1.

x = np.array([
    -3.0,
    -2.5,
    -2.0,
    -1.5,
    -1.0,
    -0.5,
     0.2,
     0.7,
     1.2,
     1.8,
     2.4,
     3.0,
], dtype=np.float64)

y = np.array([
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
], dtype=np.float64)

print("Toy data")
print("=" * 30)

for xi, yi in zip(x, y):
    print(f"x = {xi:+.2f}, y = {int(yi)}")


# ============================================================
# 8. Initialize the model
# ============================================================

# Deliberately choose a negative weight so that the initial model
# orders the classes in the wrong direction.

initial_w = -0.8
initial_b = 0.3

initial_loss, initial_cache = forward(
    x=x,
    y=y,
    w=initial_w,
    b=initial_b,
)

initial_probabilities = initial_cache["a"]
initial_predictions = predict(x, initial_w, initial_b)
initial_accuracy = accuracy(y, initial_predictions)

print("\nInitial model")
print("=" * 30)
print(f"w        = {initial_w:+.6f}")
print(f"b        = {initial_b:+.6f}")
print(f"loss     = {initial_loss:.6f}")
print(f"accuracy = {initial_accuracy:.3f}")


# ============================================================
# 9. Check the gradients before training
# ============================================================

print()
check_gradients(
    x=x,
    y=y,
    w=initial_w,
    b=initial_b,
    eps=1e-5,
)


# ============================================================
# 10. Train the model
# ============================================================

print("\nTraining")
print("=" * 100)

final_w, final_b, history = train(
    x=x,
    y=y,
    initial_w=initial_w,
    initial_b=initial_b,
    learning_rate=0.5,
    n_steps=1000,
    print_every=100,
)


# ============================================================
# 11. Inspect the final model
# ============================================================

final_loss, final_cache = forward(
    x=x,
    y=y,
    w=final_w,
    b=final_b,
)

final_probabilities = final_cache["a"]
final_predictions = predict(x, final_w, final_b)
final_accuracy = accuracy(y, final_predictions)

print("\nFinal model")
print("=" * 50)
print(f"w        = {final_w:+.6f}")
print(f"b        = {final_b:+.6f}")
print(f"loss     = {final_loss:.6f}")
print(f"accuracy = {final_accuracy:.3f}")

if abs(final_w) > 1e-12:
    decision_boundary = -final_b / final_w
    print(f"0.5 decision boundary: x = {decision_boundary:+.6f}")


# ============================================================
# 12. Print predictions for individual samples
# ============================================================

print("\nFinal predictions")
print("=" * 62)
print(
    f"{'x':>8s}"
    f"{'target':>10s}"
    f"{'probability':>16s}"
    f"{'prediction':>14s}"
)
print("-" * 62)

for xi, yi, probability, prediction in zip(
    x,
    y,
    final_probabilities,
    final_predictions,
):
    print(
        f"{xi:8.2f}"
        f"{int(yi):10d}"
        f"{probability:16.6f}"
        f"{prediction:14d}"
    )


# ============================================================
# 13. Plot the result
# ============================================================

x_plot = np.linspace(
    x.min() - 1.0,
    x.max() + 1.0,
    500,
)

initial_curve = predict_proba(
    x_plot,
    initial_w,
    initial_b,
)

final_curve = predict_proba(
    x_plot,
    final_w,
    final_b,
)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))


# ------------------------------------------------------------
# Plot 1: learned probability curve
# ------------------------------------------------------------

axes[0].scatter(
    x[y == 0],
    y[y == 0],
    color="tab:blue",
    marker="o",
    s=70,
    label="Class 0",
)

axes[0].scatter(
    x[y == 1],
    y[y == 1],
    color="tab:orange",
    marker="s",
    s=70,
    label="Class 1",
)

axes[0].plot(
    x_plot,
    initial_curve,
    linestyle="--",
    color="gray",
    linewidth=2,
    label="Initial model",
)

axes[0].plot(
    x_plot,
    final_curve,
    color="black",
    linewidth=2,
    label="Trained model",
)

axes[0].axhline(
    0.5,
    color="red",
    linestyle=":",
    label="Threshold",
)

if abs(final_w) > 1e-12:
    axes[0].axvline(
        decision_boundary,
        color="red",
        linestyle=":",
    )

axes[0].set_xlabel("Input x")
axes[0].set_ylabel("Sigmoid output")
axes[0].set_title("Toy classification problem")
axes[0].set_ylim(-0.08, 1.08)
axes[0].legend()
axes[0].grid(alpha=0.3)


# ------------------------------------------------------------
# Plot 2: loss history
# ------------------------------------------------------------

axes[1].plot(
    history["step"],
    history["loss"],
    color="tab:blue",
)

axes[1].set_xlabel("Gradient-descent step")
axes[1].set_ylabel("Mean squared error")
axes[1].set_title("Training loss")
axes[1].grid(alpha=0.3)


# ------------------------------------------------------------
# Plot 3: parameter history
# ------------------------------------------------------------

axes[2].plot(
    history["step"],
    history["w"],
    label="w",
)

axes[2].plot(
    history["step"],
    history["b"],
    label="b",
)

axes[2].set_xlabel("Gradient-descent step")
axes[2].set_ylabel("Parameter value")
axes[2].set_title("Parameter evolution")
axes[2].legend()
axes[2].grid(alpha=0.3)


plt.tight_layout()
plt.show()