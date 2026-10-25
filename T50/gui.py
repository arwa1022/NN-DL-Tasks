# gui.py
# Part 1: training and classifying (no widgets here)
# Part 2: the window (tkinter)
# Run this file to start the program. penguins.csv must be in the same folder.

import os
import warnings
import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib
matplotlib.use("Agg")   # the plot functions only draw, we show them in our window
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from preprocessing import (CLASSES, FEATURES, prepare_data,
                           transform_sample, pos, negative)
from models import SL_Perceptron, adaline          # the 2 algorithms
from evaluation import (show_confusion_matrix, plot_decision_boundary,
                        calculate_confusion_matrix, calculate_accuracy)

warnings.filterwarnings("ignore")   # plt.show() warns when we use Agg

folder = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(folder, "penguins.csv")
PICTURE_SIZE = 3.5

# ---------- colors ----------
BACKGROUND = "#eaf2f8"
CARD = "#ffffff"
HEADER = "#00695c"
BLUE = "#0e63b9"
ORANGE = "#c26609"
PURPLE = "#631a83"
GREEN = "#378f3b"

# message colors: (background, text)
MESSAGE_COLORS = {
    "error": ("#ffe0e0", "#a00000"),
    "warning": ("#fff3e0", "#e65100"),
    "success": ("#d9f2d9", "#1b5e20"),
    "info": ("#e3f2fd", "#0d47a1"),
}

# start values (Reset puts them back). An empty seed = a different random split every time
DEFAULTS = {"eta": "0.01", "epochs": "100", "mse": "0.01", "seed": "42"}



# PART 1 - training and classifying
PERCEPTRON = "Perceptron"
ADALINE = "Adaline"

# the trained model is saved here when we press Train (None = not trained yet)
model = None


# testing phase: x -> net value v -> signum -> class ID y
def net_value(X, weights, bias):
    # v = w1*x1 + w2*x2 + b
    return X @ weights + bias


def signum(v):
    # +1 if v >= 0, otherwise -1
    return np.where(v >= 0, pos, negative)


def predict(X, weights, bias):
    # y for every row of X
    return signum(net_value(X, weights, bias))


def train_model(features, classes, eta, epochs, mse_threshold,
                use_bias, algorithm, seed):
    # check the settings
    if eta <= 0:
        raise ValueError("Learning rate must be bigger than 0")
    if epochs <= 0:
        raise ValueError("Epochs must be bigger than 0")
    if algorithm == ADALINE and mse_threshold <= 0:
        raise ValueError("MSE threshold must be bigger than 0")

    X_train, y_train, X_test, y_test, scaler = prepare_data(
        features, classes, csv_path, seed)

    # small random weights and bias
    rng = np.random.default_rng(seed)
    weights = rng.uniform(-0.1, 0.1, 2)
    if use_bias:
        bias = rng.uniform(-0.1, 0.1)
        b = bias
    else:
        bias = 0.0
        b = None        # None means: no bias

    # train
    if algorithm == PERCEPTRON:
        result = SL_Perceptron(X_train, y_train, epochs, weights, eta, b)
        if len(result) == len(weights) + 1:     # the bias is the first value
            bias = result[0]
            weights = result[1:]
        else:
            weights = result
    else:
        result = adaline(X_train, y_train, eta, epochs, weights, mse_threshold, b)
        if isinstance(result, tuple):           # adaline gave back (weights, bias)
            weights = result[0]
            bias = result[1]
        else:
            weights = result

    # test on the samples the model never saw
    y_pred = predict(X_test, weights, bias)
    cm = calculate_confusion_matrix(y_test, y_pred, [negative, pos])
    accuracy = calculate_accuracy(y_test, y_pred)

    # everything we need later is kept in one dictionary
    trained = {
        "weights": weights,
        "bias": bias,
        "scaler": scaler,
        "features": list(features),
        "classes": list(classes),
        "X_train": X_train,
        "y_train": y_train,
        "cm": cm,
        "accuracy": accuracy,
    }
    return trained


def classify_sample(trained, value1, value2):
    # given sample x -> net value v -> signum -> y -> class name
    x = transform_sample(value1, value2, trained["scaler"])
    v = float(net_value(x, trained["weights"], trained["bias"]))
    y = int(signum(v))

    if y == pos:
        name = trained["classes"][0]
    else:
        name = trained["classes"][1]

    return v, y, name


# PART 2 - the window
def read_number(text, name, number_type):
    # turn the text of a box into a number, or say what is wrong
    try:
        return number_type(text)
    except ValueError:
        if number_type is int:
            raise ValueError(name + " must be a whole number")
        raise ValueError(name + " must be a number")


def show_message(text, kind, details="", size=14):
    # text = the big line, details = the small lines under it
    # kind = "error", "warning", "success" or "info" (chooses the colors)
    background, foreground = MESSAGE_COLORS[kind]
    result_card.config(bg=background)
    headline.config(text=text, bg=background, fg=foreground,
                    font=("Arial", size, "bold"))
    details_label.config(text=details, bg=background, fg=foreground)


def card_resize(event):
    # keep the text inside the box when the window size changes
    width = max(event.width - 30, 200)
    headline.config(wraplength=width)
    details_label.config(wraplength=width)


def update_mse_state():
    # only Adaline uses the MSE threshold, so grey it out for Perceptron
    if algorithm_var.get() == ADALINE:
        mse_entry.config(state="normal")
    else:
        mse_entry.config(state="disabled")


def clear_pictures():
    for widget in picture_frame.winfo_children():
        widget.destroy()


def show_placeholder():
    # what the picture area shows before the first training
    clear_pictures()
    tk.Label(picture_frame, bg=CARD, fg="#90a4ae", font=("Arial", 14),
             text="The confusion matrix and the graph\nwill appear here "
                  "after you press Train").pack(expand=True)


def draw_picture(plot_function, *arguments):
    # run a plot function from evaluation.py, take the figure it drew
    # and put it inside our window
    plot_function(*arguments)
    figure = plt.gcf()
    plt.close(figure)
    figure.set_size_inches(PICTURE_SIZE, PICTURE_SIZE)
    figure.tight_layout()

    canvas = FigureCanvasTkAgg(figure, master=picture_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side="left", fill="both", expand=True)


def train_click():
    global model

    try:
        eta = read_number(eta_var.get(), "Learning rate", float)
        epochs = read_number(epochs_var.get(), "Epochs", int)

        # the MSE threshold is only needed for Adaline
        if algorithm_var.get() == ADALINE:
            mse_threshold = read_number(mse_var.get(), "MSE threshold", float)
        else:
            mse_threshold = float(DEFAULTS["mse"])

        # empty seed box = random every time
        seed_text = seed_var.get().strip()
        if seed_text == "":
            seed = None
        else:
            seed = read_number(seed_text, "Seed", int)

        result = train_model((feature1_var.get(), feature2_var.get()),
                             (class1_var.get(), class2_var.get()),
                             eta, epochs, mse_threshold,
                             bias_var.get(), algorithm_var.get(), seed)
    except FileNotFoundError:
        show_message("File not found: " + csv_path, "error")
        return
    except ValueError as error:
        show_message(str(error), "error")
        return

    model = result      # saved only after everything worked

    # the confusion matrix and the graph, side by side
    clear_pictures()
    draw_picture(show_confusion_matrix, model["cm"], model["classes"])
    draw_picture(plot_decision_boundary, model["X_train"], model["y_train"],
                 model["weights"], model["bias"], model["features"],
                 model["classes"])

    # the sample boxes get the names of the features we trained on
    sample1_label.config(text=model["features"][0] + ":")
    sample2_label.config(text=model["features"][1] + ":")

    # the accuracy: green if it is good, orange if it is medium, red if it is bad
    percent = model["accuracy"] * 100
    if percent >= 90:
        kind = "success"
    elif percent >= 70:
        kind = "warning"
    else:
        kind = "error"

    weights = model["weights"]
    details = (f"Weights: [{weights[0]:.3f}, {weights[1]:.3f}]   "
               f"Bias: {model['bias']:.3f}\n"
               f"Model trained on: {model['features'][0]}, {model['features'][1]}"
               f"  |  {model['classes'][0]} vs {model['classes'][1]}")
    show_message(f"Test accuracy: {percent:.2f}%", kind, details, size=24)


def classify_click():
    # no training yet -> error message, nothing else happens
    if model is None:
        show_message("Train a model first", "error")
        return

    try:
        v, y, name = classify_sample(model, sample1_var.get(), sample2_var.get())
    except ValueError as error:
        show_message("Bad sample: " + str(error), "error")
        return

    if y == 1:
        sign = "+1"
    else:
        sign = "-1"
    show_message("Predicted class: " + name, "info",
                 f"Net value v = {v:.4f}   ->   signum(v) = y = {sign}", size=20)


def enter_pressed(event):
    # pressing Enter in a sample box is the same as pressing Classify
    classify_click()


def reset_click():
    global model

    feature1_var.set(FEATURES[0])
    feature2_var.set(FEATURES[2])
    class1_var.set(CLASSES[0])
    class2_var.set(CLASSES[1])

    eta_var.set(DEFAULTS["eta"])
    epochs_var.set(DEFAULTS["epochs"])
    mse_var.set(DEFAULTS["mse"])
    seed_var.set(DEFAULTS["seed"])
    bias_var.set(True)
    algorithm_var.set(PERCEPTRON)
    update_mse_state()

    sample1_var.set("")
    sample2_var.set("")
    sample1_label.config(text="Value 1:")
    sample2_label.config(text="Value 2:")

    model = None        # forget the trained model
    show_placeholder()
    show_message("Choose your settings and press Train.", "info")


# building the window
window = tk.Tk()
window.title("Penguins Classifier")
window.configure(bg=BACKGROUND)

# variables that hold what the user chose
feature1_var = tk.StringVar(value=FEATURES[0])
feature2_var = tk.StringVar(value=FEATURES[2])
class1_var = tk.StringVar(value=CLASSES[0])
class2_var = tk.StringVar(value=CLASSES[1])
eta_var = tk.StringVar(value=DEFAULTS["eta"])
epochs_var = tk.StringVar(value=DEFAULTS["epochs"])
mse_var = tk.StringVar(value=DEFAULTS["mse"])
seed_var = tk.StringVar(value=DEFAULTS["seed"])
bias_var = tk.BooleanVar(value=True)
algorithm_var = tk.StringVar(value=PERCEPTRON)
sample1_var = tk.StringVar()
sample2_var = tk.StringVar()


def make_box(parent, title, color):
    # a white box with a colored title, gives back the white part
    outer = tk.Frame(parent, bg=CARD, bd=1, relief="solid")
    outer.pack(fill="x", pady=4)
    tk.Label(outer, text=title, bg=color, fg="white", anchor="w", padx=8,
             font=("Arial", 11, "bold")).pack(fill="x")
    inside = tk.Frame(outer, bg=CARD, padx=10, pady=5)
    inside.pack(fill="x")
    return inside


def add_dropdown(parent, row, text, variable, options):
    tk.Label(parent, text=text, bg=CARD, font=("Arial", 10), width=14,
             anchor="w").grid(row=row, column=0, pady=2)
    ttk.Combobox(parent, textvariable=variable, values=options,
                 state="readonly", width=18).grid(row=row, column=1, pady=2)


def add_entry(parent, row, text, variable):
    # gives back the label and the entry box
    label = tk.Label(parent, text=text, bg=CARD, font=("Arial", 10), width=14,
                     anchor="w")
    label.grid(row=row, column=0, pady=2)
    entry = tk.Entry(parent, textvariable=variable, width=21, relief="solid",
                     bd=1, disabledbackground="#eeeeee", disabledforeground="#9e9e9e")
    entry.grid(row=row, column=1, pady=2)
    return label, entry


def make_button(parent, text, color, command):
    return tk.Button(parent, text=text, bg=color, fg="white", relief="flat",
                     font=("Arial", 11, "bold"), cursor="hand2", command=command)


# title
tk.Label(window, text="Penguins Classifier", bg=HEADER, fg="white", pady=6,
         font=("Arial", 16, "bold")).pack(fill="x")

left = tk.Frame(window, bg=BACKGROUND)
left.pack(side="left", fill="y", padx=(10, 5), pady=5)
right = tk.Frame(window, bg=BACKGROUND)
right.pack(side="right", fill="both", expand=True, padx=(5, 10), pady=10)

# 1 - choose data
box1 = make_box(left, "1. Choose data", BLUE)
add_dropdown(box1, 0, "Feature 1:", feature1_var, FEATURES)
add_dropdown(box1, 1, "Feature 2:", feature2_var, FEATURES)
add_dropdown(box1, 2, "Class 1:", class1_var, CLASSES)
add_dropdown(box1, 3, "Class 2:", class2_var, CLASSES)

# 2 - settings
box2 = make_box(left, "2. Settings", ORANGE)
add_entry(box2, 0, "Learning rate:", eta_var)
add_entry(box2, 1, "Epochs (m):", epochs_var)
mse_label, mse_entry = add_entry(box2, 2, "MSE threshold:", mse_var)
add_entry(box2, 3, "Seed (optional):", seed_var)

tk.Label(box2, text="Algorithm:", bg=CARD, font=("Arial", 10),
         anchor="w").grid(row=4, column=0, sticky="w")
algorithm_frame = tk.Frame(box2, bg=CARD)
algorithm_frame.grid(row=4, column=1, sticky="w")
for name in (PERCEPTRON, ADALINE):
    tk.Radiobutton(algorithm_frame, text=name, variable=algorithm_var, value=name,
                   bg=CARD, activebackground=CARD, font=("Arial", 10),
                   command=update_mse_state).pack(side="left")

tk.Checkbutton(box2, text="Add bias", variable=bias_var, bg=CARD,
               activebackground=CARD, font=("Arial", 10)).grid(
    row=5, column=0, columnspan=2, sticky="w")

# Train and Reset side by side
button_frame = tk.Frame(box2, bg=CARD)
button_frame.grid(row=6, column=0, columnspan=2, sticky="we", pady=(6, 0))
button_frame.columnconfigure(0, weight=3)
button_frame.columnconfigure(1, weight=1)
make_button(button_frame, "Train", GREEN, train_click).grid(
    row=0, column=0, sticky="we", padx=(0, 4))
make_button(button_frame, "Reset", "#757575", reset_click).grid(
    row=0, column=1, sticky="we")

# 3 - classify one sample
box3 = make_box(left, "3. Classify one sample", PURPLE)
tk.Label(box3, text="Train first. For OriginLocation write\nTorgersen, Biscoe or Dream.",
         bg=CARD, fg="#555555", font=("Arial", 9, "italic"),
         justify="left").grid(row=0, column=0, columnspan=2, sticky="w")
sample1_label, sample1_entry = add_entry(box3, 1, "Value 1:", sample1_var)
sample2_label, sample2_entry = add_entry(box3, 2, "Value 2:", sample2_var)
sample1_entry.bind("<Return>", enter_pressed)
sample2_entry.bind("<Return>", enter_pressed)
make_button(box3, "Classify", PURPLE, classify_click).grid(
    row=3, column=0, columnspan=2, sticky="we", pady=(6, 0))

# the result box (top right): a big line and a small line under it
result_card = tk.Frame(right, bd=1, relief="solid")
result_card.pack(fill="x", pady=(0, 10))
headline = tk.Label(result_card, wraplength=680, justify="left", anchor="w",
                    padx=12, pady=6)
headline.pack(fill="x")
details_label = tk.Label(result_card, font=("Arial", 10), wraplength=680,
                         justify="left", anchor="w", padx=12)
details_label.pack(fill="x", pady=(0, 6))
result_card.bind("<Configure>", card_resize)

# the pictures (bottom right)
picture_frame = tk.Frame(right, bg=CARD, bd=1, relief="solid")
picture_frame.pack(fill="both", expand=True)

update_mse_state()
show_placeholder()
show_message("Choose your settings and press Train.", "info")

window.mainloop()