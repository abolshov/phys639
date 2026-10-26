#!/usr/bin/env python3
"""
Demonstrate TensorFlow automatic differentiation and optimization.

The script contains two examples:

1. Compute the derivative of
       f(x) = x^2 + 2x + 1

2. Minimize
       L(x) = (x - target)^2

using stochastic gradient descent.
"""

from __future__ import annotations

import argparse

import tensorflow as tf


def demonstrate_derivative(*, x_value: float) -> None:
    """Evaluate f(x) and df/dx using tf.GradientTape."""

    x = tf.Variable(x_value, dtype=tf.float32)

    with tf.GradientTape() as tape:
        y = x**2 + 2.0 * x + 1.0

    gradient = tape.gradient(y, x)

    # Analytical derivative: df/dx = 2x + 2
    expected_gradient = 2.0 * x + 2.0

    print("Automatic differentiation")
    print("--------------------------")
    print(f"x                  = {x.numpy():.6f}")
    print(f"f(x)               = {y.numpy():.6f}")
    print(f"TensorFlow df/dx   = {gradient.numpy():.6f}")
    print(f"Analytical df/dx   = {expected_gradient.numpy():.6f}")
    print()


def demonstrate_optimization(
    *,
    initial_x: float,
    target: float,
    learning_rate: float,
    steps: int,
) -> None:
    """Minimize (x - target)^2 using gradient descent."""

    x = tf.Variable(initial_x, dtype=tf.float32)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    print("Gradient-based optimization")
    print("---------------------------")
    print(f"Objective: minimize (x - {target})^2")
    print(f"Initial x: {initial_x}")
    print(f"Learning rate: {learning_rate}")
    print()
    print(f"{'Step':>6} {'x':>14} {'Loss':>14} {'Gradient':>14}")
    print("-" * 52)

    for step in range(steps + 1):
        with tf.GradientTape() as tape:
            loss = (x - target) ** 2

        gradient = tape.gradient(loss, x)

        print(
            f"{step:6d} "
            f"{x.numpy():14.8f} "
            f"{loss.numpy():14.8f} "
            f"{gradient.numpy():14.8f}"
        )

        # Do not apply an additional update after the final reported step.
        if step < steps:
            optimizer.apply_gradients([(gradient, x)])

    print()
    print(f"Final x: {x.numpy():.8f}")
    print(f"Target:  {target:.8f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Demonstrate TensorFlow automatic differentiation and "
            "gradient-based optimization."
        )
    )

    parser.add_argument(
        "--derivative-x",
        type=float,
        default=3.0,
        help="Point at which to evaluate the derivative (default: 3.0).",
    )
    parser.add_argument(
        "--initial-x",
        type=float,
        default=5.0,
        help="Initial value used in the optimization (default: 5.0).",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=2.0,
        help="Target value in the optimization objective (default: 2.0).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="SGD learning rate (default: 0.1).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of optimization steps (default: 20).",
    )

    args = parser.parse_args()

    if args.steps < 0:
        parser.error("--steps must be non-negative")

    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")

    return args


def main() -> None:
    args = parse_args()

    demonstrate_derivative(x_value=args.derivative_x)

    demonstrate_optimization(
        initial_x=args.initial_x,
        target=args.target,
        learning_rate=args.learning_rate,
        steps=args.steps,
    )


if __name__ == "__main__":
    main()
