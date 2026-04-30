import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt


def initialize_weights(num_of_inputs, num_outputs=10):
    return np.random.rand(num_outputs, num_of_inputs)


def initialise_bias(num_outputs=10):
    return np.random.rand(num_outputs)


def compute_raw_outputs(inputs, weights, bias):

    raw_outputs = []

    for i in range(len(weights)):
        dot_sum = 0
        for y in range(len(inputs)):
            dot_sum += inputs[y] * weights[i][y]
        raw_outputs.append(dot_sum + bias[i])

    return np.array(raw_outputs)


def softmax(raw_outputs):
    shifted = raw_outputs - np.max(raw_outputs)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


def entropy_loss_gradient(probabilities, target):
    return probabilities - target


def iterate_update(weights, bias, input, target_output, learning_rate=0.15):

    raw_outputs = compute_raw_outputs(input, weights, bias)
    probabilities = softmax(raw_outputs)

    gradients = entropy_loss_gradient(probabilities, target_output)

    for i in range(len(weights)):
        for y in range(len(weights[i])):
            weights[i][y] -= learning_rate * gradients[i] * input[y]

    for i in range(len(bias)):
        bias[i] -= learning_rate * gradients[i]

    return weights, bias


def train_one_epoch(train_set, train_labels, weights, bias, lr=0.25):

    for i in range(len(train_set)):
        target = np.zeros(10)
        target[train_labels[i]] = 1

        weights, bias = iterate_update(
            weights, bias, train_set[i], target, learning_rate=lr
        )

    return weights, bias


def train_epochs(train_set, train_labels, test_set, test_labels, weights, bias, epochs=5, lr=0.1):

    train_accs = []
    test_accs = []

    for e in range(epochs):

        weights, bias = train_one_epoch(train_set, train_labels, weights, bias, lr)

        train_acc = accuracy(weights, bias, train_set, train_labels)
        test_acc = accuracy(weights, bias, test_set, test_labels)

        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Epoch {e+1}: Train={train_acc:.3f}, Test={test_acc:.3f}")

    return weights, bias, train_accs, test_accs


def plot_training(train_accs, test_accs):
    plt.figure()
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(test_accs, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training Progress")
    plt.show()


def ReLU(x):
    return x * (x > 0)


def Convolution(png):

    image = png
    rows = len(image)
    columns = len(image[0])

    kernel = [[-1, -1, -1],
              [-1, 8, -1],
              [-1, -1, -1]]

    feature_map = []

    for i in range(rows):
        row = []
        for j in range(columns):

            pixel_envoirment = []

            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:

                    ni = i + di
                    nj = j + dj

                    if 0 <= ni < rows and 0 <= nj < columns:
                        pixel_envoirment.append(image[ni][nj])
                    else:
                        pixel_envoirment.append(0)

            dot_sum = 0

            for a in range(3):
                for b in range(3):
                    dot_sum += pixel_envoirment[a * 3 + b] * kernel[a][b]

            row.append(dot_sum)

        feature_map.append(row)

    return [[ReLU(feature_map[i][j]) for j in range(len(feature_map[0]))]
            for i in range(len(feature_map))]


def pooling_layer(feature_map):

    dim = len(feature_map)
    compressed_img = []

    for i in range(0, dim, 2):
        row = []
        for y in range(0, dim, 2):

            x1 = feature_map[i][y]
            x2 = feature_map[i][y + 1]
            x3 = feature_map[i + 1][y]
            x4 = feature_map[i + 1][y + 1]

            row.append((x1 + x2 + x3 + x4) / 4)

        compressed_img.append(row)

    return compressed_img


def flatten(x):
    return np.array(x).flatten()


def preprocess(x):
    x = x.astype(np.int16)

    c1 = Convolution(x)
    p1 = pooling_layer(c1)

    c2 = Convolution(p1)
    p2 = pooling_layer(c2)

    return flatten(p2)


def accuracy(weights, bias, X, y):

    correct = 0

    for i in range(len(X)):
        raw = compute_raw_outputs(X[i], weights, bias)
        probs = softmax(raw)
        pred = np.argmax(probs)

        if pred == y[i]:
            correct += 1

    return correct / len(X)


def visualize_predictions(x_test, X_test, y_test, weights, bias, n=10):

    plt.figure(figsize=(12, 3))

    for i in range(n):

        img_raw = x_test[i]
        label = y_test[i]

        raw = compute_raw_outputs(X_test[i], weights, bias)
        probs = softmax(raw)

        pred = np.argmax(probs)
        conf = np.max(probs)

        plt.subplot(1, n, i+1)
        plt.imshow(img_raw, cmap="gray")
        plt.title(f"T:{label}\nP:{pred}\n{conf:.2f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train[:2000]
    y_train = y_train[:2000]

    x_test = x_test[:200]
    y_test = y_test[:200]

    print("Preprocessing train...")
    X_train = np.array([preprocess(img) for img in x_train])

    print("Preprocessing test...")
    X_test = np.array([preprocess(img) for img in x_test])

    input_size = len(X_train[0])

    weights = initialize_weights(input_size)
    bias = initialise_bias()

    print("Initial train accuracy:", accuracy(weights, bias, X_train, y_train))
    print("Initial test accuracy:", accuracy(weights, bias, X_test, y_test))

    print("Training...")

    weights, bias, train_accs, test_accs = train_epochs(
        X_train, y_train, X_test, y_test,
        weights, bias,
        epochs=8,
        lr=0.1
    )

    plot_training(train_accs, test_accs)

    print("Final train accuracy:", accuracy(weights, bias, X_train, y_train))
    print("Final test accuracy:", accuracy(weights, bias, X_test, y_test))

    visualize_predictions(x_test, X_test, y_test, weights, bias, n=10)


if __name__ == "__main__":
    main()