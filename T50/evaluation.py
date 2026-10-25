import numpy as np
import matplotlib.pyplot as plt


def calculate_confusion_matrix(y_true, y_pred, classes):
    # classes = [negative_label, positive_label], e.g. [-1, 1]
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    neg, pos = classes

    tn = fp = fn = tp = 0
    for t, p in zip(y_true, y_pred):
        if t == pos and p == pos:
            tp += 1
        elif t == neg and p == pos:
            fp += 1
        elif t == pos and p == neg:
            fn += 1
        elif t == neg and p == neg:
            tn += 1

    return np.array([[tn, fp],
                     [fn, tp]])


def calculate_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    correct = 0
    for t, p in zip(y_true, y_pred):
        if t == p:
            correct += 1

    return correct / len(y_true)


def show_confusion_matrix(cm, class_names):
    # class_names = [positive_name, negative_name], same order as class_n in prepare_data
    labels = [class_names[1], class_names[0]]  # rows/cols go negative first, then positive

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", color=color, fontsize=14)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def plot_decision_boundary(X, y, weights, bias, feature_names, class_names):
    # y is +1 for class_names[0] and -1 for class_names[1] (same as prepare_data)
    X = np.asarray(X)
    y = np.asarray(y)
    w1, w2 = weights

    plt.figure(figsize=(6, 5))
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="tab:blue", marker="o", label=class_names[0])
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c="tab:red", marker="s", label=class_names[1])

    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # x2 = -(w1*x1 + b) / w2
    if w2 != 0:
        x1_line = np.linspace(x1_min, x1_max, 100)
        x2_line = -(w1 * x1_line + bias) / w2
        plt.plot(x1_line, x2_line, "k--", label="Decision boundary")
    elif w1 != 0:
        # w2 == 0 -> vertical line at x1 = -b / w1
        plt.axvline(-bias / w1, color="k", linestyle="--", label="Decision boundary")
    else:
        print("Both weights are 0, can't draw a boundary")

    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(f"{class_names[0]} vs {class_names[1]}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()