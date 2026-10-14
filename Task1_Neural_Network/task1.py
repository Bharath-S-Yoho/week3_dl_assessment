import numpy as np
import matplotlib.pyplot as plt
import torch

# XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def softmax(x):
    x = np.array(x, dtype=float)
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def mse(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)


def cross_entropy(y_true, y_pred):
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def train_xor(lr=0.5, epochs=5000, seed=1):
    # Small random weights; network structure is 2 inputs, 4 hidden units, 1 output.
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 0.5, (2, 4))
    b1 = np.zeros((1, 4))
    W2 = rng.normal(0, 0.5, (4, 1))
    b2 = np.zeros((1, 1))
    losses = []

    for epoch in range(epochs):
        # Forward pass
        z1 = X @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        pred = sigmoid(z2)
        losses.append(cross_entropy(y, pred))

        # Backpropagation for sigmoid output + binary cross entropy
        dz2 = (pred - y) / len(X)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Gradient descent update
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    final_pred = sigmoid(relu(X @ W1 + b1) @ W2 + b2)
    return final_pred.ravel(), losses


if __name__ == "__main__":
    print("softmax([2.0, 1.0, 0.1]) =", np.round(softmax([2.0, 1.0, 0.1]), 3))
    print("mse example:", mse([1, 0], [0.8, 0.2]))
    print("cross entropy example:", round(cross_entropy(y, np.full_like(y, 0.5)), 4))

    predictions, losses = train_xor(lr=0.5, epochs=5000)
    print("XOR predictions after training:", np.round(predictions, 2))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy loss")
    plt.title("XOR training loss")
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.close()
    print("Saved loss_curve.png")

    # Compare learning rates. Very large learning rates can diverge or become unstable.
    plt.figure()
    for rate in [0.01, 0.5, 5.0]:
        _, rate_losses = train_xor(lr=rate, epochs=500, seed=1)
        safe_losses = np.clip(np.array(rate_losses), 0, 10)
        plt.plot(safe_losses, label=f"lr={rate}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (clipped at 10 for display)")
    plt.title("Learning-rate comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("learning_rates.png")
    plt.close()
    print("Saved learning_rates.png")

    # Hand calculation: L = (w*x + b - t)^2
    # dL/dw = 2*(w*x+b-t)*x = 2*(2-10)*2 = -32
    # dL/db = 2*(w*x+b-t) = 2*(2-10) = -16
    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    x = torch.tensor(2.0)
    target = torch.tensor(10.0)
    loss = (w * x + b - target) ** 2
    loss.backward()
    print("Hand gradients: dw=-32, db=-16")
    print("Autograd: w.grad =", w.grad.item(), ", b.grad =", b.grad.item())

    print("\nRegularisation concepts:")
    print("Dropout: Randomly turns off some neurons during training to reduce overfitting.")
    print("BatchNorm: Normalises layer values in a batch to make training more stable.")
    print("Early stopping: Stops training when validation performance stops improving.")
