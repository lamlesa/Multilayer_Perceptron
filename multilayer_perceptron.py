"""
    THE PROGRAM HAS 4 OPERATING MODES:

    python multilayer_perceptron.py eda     --dataset data.csv
    python multilayer_perceptron.py split   --dataset data.csv --val_ratio 0.2 --seed 42
    python multilayer_perceptron.py train   --train data_training.csv --valid data_validation.csv \
                                            --layer 24 24 24 --epochs 84 --loss binaryCrossentropy \
                                            --batch_size 8 --learning_rate 0.0314
    python multilayer_perceptron.py predict --dataset data_validation.csv --model saved_model.npy
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

EPS = 1e-12

def download_data_raw(path):
    df = pd.read_csv(path, header = None, sep=",")

    if df.shape[1] == 1:
        df = df[0].astype(str).str.strip('"').str.split(",", expand=True)
    if str(df.iloc[0, 1]).strip() not in ("M", "B"):
        df = df.iloc[1:].reset_index(drop=True)
    df[1] = df[1].astype(str).str.strip()

    return df


def separate_target_from_features(df):
    X = df.iloc[:, 2:].astype(float).to_numpy()
    y = (df[1] == "M").astype(int).to_numpy()
    return X, y

def one_hot_encoding(y):
    Y = np.zeros((len(y), 2))
    Y[np.arange(0, len(y), 1), y] = 1.0
    return Y


def sigmoid(z):
    return 1.0/(1+np.exp(-np.clip(z, -555, 555))) # np.clip() от переполнения

def softmax(z):
    # чтобы не было переполнения вычитаем константу
    z = z - z.max(axis = 1, keepdims = True) 
    e = np.exp(z)
    return e/e.sum(axis = 1, keepdims = True)

ACTIVATIONS = {
    "tanh": (np.tanh, lambda a: 1.0 - a**2),
    "sigmoid": (sigmoid, lambda a: a*(1.0 - a)),
    "relu": (lambda z: np.maximum(0.0, z), lambda a: (a > 0).astype(float)),
}

def initialize_weights(fan_in, fan_out, kind, rand_num_gen):
    if kind == "HeUniform":
        lim = np.sqrt(6.0 / fan_in)
        return rand_num_gen.uniform(-lim, lim, (fan_in, fan_out))
    if kind == "HeNormal":
        return rand_num_gen.standard_normal((fan_in, fan_out)) * np.sqrt(2.0/fan_in)
    if kind == "Xavier":
        lim = np.sqrt(6.0 / (fan_in + fan_out))
        return rand_num_gen.uniform(-lim, lim, (fan_in, fan_out))
    raise ValueError("Unknown kind of initializer!")

def BCE(P, Y):
    p = np.clip(P[:, 1], EPS, 1 - EPS)
    y = Y[:, 1]
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(P, Y):
    return float(np.mean(P.argmax(axis = 1) == Y.argmax(axis = 1)))


class MLP:
    # конструктор класса
    def __init__(self, n_in, hidden, n_out = 2, activation = "sigmoid", initializer = "Xavier", seed = 42):
        self.sizes = [n_in] + list(hidden) + [n_out]
        self.activation = activation
        self.initializer = initializer
        rand_num_gen = np.random.default_rng(seed)
        self.W = [initialize_weights(a, b, initializer, rand_num_gen) for a, b in zip(self.sizes[:-1], self.sizes[1:])]
        self.b = [np.zeros((1, b)) for b in self.sizes[1:]]

    def forward(self, X):
        f, _ = ACTIVATIONS[self.activation]
        acts = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1]@W + b
            acts.append(softmax(z) if i == len(self.W)-1 else f(z))
        return acts


    def predict_probability(self, X):
        return self.forward(X)[-1]

    def backward_grads(self, acts, Y):
        _, df = ACTIVATIONS[self.activation]
        batch_size = Y.shape[0]
        n_layers = len(self.W)
        grad_W = [None] * n_layers
        grad_b = [None] * n_layers

        error = (acts[-1] - Y) / batch_size

        for layer in reversed(range(n_layers)):
            a_in = acts[layer]

            grad_W[layer] = a_in.T @ error
            grad_b[layer] = error.sum(axis=0, keepdims=True)

            if layer > 0:
                error_to_previous = error @ self.W[layer].T
                error = error_to_previous * df(a_in)

        return grad_W, grad_b

    def save(self, path, mean, std):
        np.save(path, {"sizes": self.sizes, "activation": self.activation,
                       "initializer": self.initializer, "W": self.W, "b": self.b,
                       "mean": mean, "std": std}, allow_pickle=True)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True).item()
        m = cls(d["sizes"][0], d["sizes"][1:-1], d["sizes"][-1], d["activation"], d["initializer"])
        m.W, m.b = d["W"], d["b"]
        return m, d["mean"], d["std"]


def command_EDA(a):
    df = download_data_raw(a.dataset)
    X, y = separate_target_from_features(df)
    names = [f"f{i}" for i in range(X.shape[1])] #имена признаков
    features = pd.DataFrame(X, columns = names)

    print(f"shape: {df.shape}")
    print(f"blanks: {int(features.isna().sum().sum())}")
    print("classes:")
    print(f"B={(y == 0).sum()} ({(y == 0).mean():.1%})")
    print(f"M={(y == 1).sum()} ({(y == 1).mean():.1%})")
    print(features.describe().T[["mean", "std", "min", "max"]].round(3).to_string())

    correlation = features.corrwith(pd.Series(y)).abs().sort_values(ascending=False)
    print("\n10 features with the highest correlation:\n", correlation.head(10).round(3).to_string())

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].bar(["B", "M"], [(y == 0).sum(), (y == 1).sum()], color=["tab:green", "tab:pink"])
    ax[0].set_title("Class balance")
    im = ax[1].imshow(features.corr().to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax[1].set_title("Корреляции признаков")
    fig.colorbar(im, ax=ax[1]) #добавляет легенду цветов

    fig.tight_layout()
    fig.savefig("Exploratory_Data_Analysis.png", dpi=120)


    top = list(correlation.index[:6])
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    for ax_, c in zip(axs.ravel(), top):
        j = names.index(c)
        ax_.hist(X[y == 0, j], bins=25, alpha=0.6, label="B")
        ax_.hist(X[y == 1, j], bins=25, alpha=0.6, label="M", color="tab:red")
        ax_.set_title(f"{c} (|corr|={correlation[c]:.2f})")
        ax_.legend()
    
    fig.tight_layout()
    fig.savefig("top_correlation_features.png", dpi=120)

    print("\nfiles with graphics: Exploratory_Data_Analysis.png, top_correlation_features.png")

    if a.show:
        plt.show()

def command_split(a):
    df = download_data_raw(a.dataset)
    y = (df[1] == "M").astype(int).to_numpy()
    rand_num_gen = np.random.default_rng(a.seed)
    val_idx = []
    for c in (0,1):
        idx = rand_num_gen.permutation(np.where(y==c)[0])
        val_idx.extend(idx[:int(round(len(idx)*a.val_ratio))])

    mask = np.zeros(len(df), bool)
    mask[val_idx] = True
    train = df[~mask].sample(frac=1, random_state=a.seed)
    valid = df[mask].sample(frac=1, random_state=a.seed)

    train.to_csv(a.out_train, header=False, index=False)
    valid.to_csv(a.out_valid, header=False, index=False)
    print(f"train: {len(train)} -> {a.out_train}, valid: {len(valid)} -> {a.out_valid} (seed={a.seed})")

def command_train(a):
    X_train, y_train = separate_target_from_features(download_data_raw(a.train))
    X_valid, y_valid = separate_target_from_features(download_data_raw(a.valid))
    mean, std = X_train.mean(0), X_train.std(0)
    std[std == 0] = 1.0
    X_train, X_valid = (X_train - mean)/std, (X_valid - mean)/std
    Y_train, Y_valid = one_hot_encoding(y_train), one_hot_encoding(y_valid)

    net = MLP(X_train.shape[1], a.layer, 2, a.activation, a.weights_initializer, a.seed)
    rand_num_gen = np.random.default_rng(a.seed)

    history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}

    for ep in range(1, a.epochs + 1):
        order = rand_num_gen.permutation(len(X_train))
        for s in range(0, len(order), a.batch_size):
            idx = order[s:s + a.batch_size]
            acts = net.forward(X_train[idx])
            dW, db = net.backward_grads(acts, Y_train[idx])
            for i in range(len(net.W)):
                net.W[i] -= a.learning_rate * dW[i]
                net.b[i] -= a.learning_rate * db[i]

        Pr_tr, Pr_va = net.predict_probability(X_train), net.predict_probability(X_valid)
        history["loss"].append(BCE(Pr_tr, Y_train)); history["val_loss"].append(BCE(Pr_va, Y_valid))
        history["acc"].append(accuracy(Pr_tr, Y_train)); history["val_acc"].append(accuracy(Pr_va, Y_valid))
        print(f"epoch {ep:02d}/{a.epochs} - loss: {history['loss'][-1]:.4f} - val_loss: {history['val_loss'][-1]:.4f}"
              f" - acc: {history['acc'][-1]:.4f} - val_acc: {history['val_acc'][-1]:.4f}")

    net.save(a.model, mean, std)
    print(f"> saving model '{a.model}' to disk...")


    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    ax[0].plot(history["loss"], label="training loss")
    ax[0].plot(history["val_loss"], "--", label="validation loss")
    ax[0].set(xlabel="epochs", ylabel="loss", title="Learning Curves: loss"); ax[0].legend(); ax[0].grid(alpha=.3)
    
    ax[1].plot(history["acc"], label="training accuracy")
    ax[1].plot(history["val_acc"], label="validation accuracy")
    ax[1].set(xlabel="epochs", ylabel="accuracy", title="Learning Curves: accuracy"); ax[1].legend(); ax[1].grid(alpha=.3)
    
    fig.tight_layout()
    fig.savefig("learning_curves.png", dpi=120)
    print("> learning curves saved to 'learning_curves.png'")

    if a.show:
        plt.show()


def command_predict(a):
    net, mean, std = MLP.load(a.model)
    X, y = separate_target_from_features(download_data_raw(a.dataset))
    P = net.predict_probability((X - mean)/std)
    Y = one_hot_encoding(y)
    prediction = P.argmax(axis=1)
    TP = int(((prediction == 1) & (y == 1)).sum())
    TN = int(((prediction == 0) & (y == 0)).sum())
    FP = int(((prediction == 1) & (y == 0)).sum())
    FN = int(((prediction == 0) & (y == 1)).sum())
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1) # max для защиты от деления на 0
    f1 = 2*precision*recall / max(precision + recall, EPS)
    
    print(f"samples: {len(y)}")
    print(f"binary cross-entropy: {BCE(P, Y):.4f}")
    print(f"accuracy: {accuracy(P, Y):.4f}  precision: {precision:.4f}  recall: {recall:.4f}  F1: {f1:.4f}")
    print(f"confusion matrix (rows=in fact B/M, columns=prediction B/M):\n[[{TN} {FP}]\n [{FN} {TP}]]")

    if a.output:
        pd.DataFrame({"true": np.where(y == 1, "M", "B"), "prediction": np.where(prediction == 1, "M", "B"),
                      "p_M": P[:, 1]}).to_csv(a.output, index=False)
        
        print(f"predictions saved to '{a.output}'")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest = "cmd", required = True)

    # EDA-режим
    e = sub.add_parser("eda", help="Exploratory Data Analysis")
    e.add_argument("--dataset", default="data.csv")
    e.add_argument("--show", action="store_true")
    e.set_defaults(fn=command_EDA)


    # SPLIT-режим
    s = sub.add_parser("split", help="Train/Valid data split")
    s.add_argument("--dataset", default="data.csv")
    s.add_argument("--val_ratio", type=float, default=0.2)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out_train", default="data_training.csv")
    s.add_argument("--out_valid", default="data_validation.csv")
    s.set_defaults(fn=command_split)


    # TRAIN-режим
    t = sub.add_parser("train", help="Training model")
    t.add_argument("--train", default="data_training.csv")
    t.add_argument("--valid", default="data_validation.csv")
    t.add_argument("--layer", type=int, nargs="+", default=[24, 24], help="hidden layer sizes")
    t.add_argument("--epochs", type=int, default=84)
    t.add_argument("--loss", default="binaryCrossentropy", choices=["binaryCrossentropy"])
    t.add_argument("--batch_size", type=int, default=8)
    t.add_argument("--learning_rate", type=float, default=0.0314)
    t.add_argument("--activation", default="sigmoid", choices=list(ACTIVATIONS))
    t.add_argument("--weights_initializer", default="HeUniform", choices=["HeUniform", "HeNormal", "Xavier"])
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--model", default="saved_model.npy")
    t.add_argument("--show", action="store_true")
    t.set_defaults(fn=command_train)
    
    
    # PREDICT-режим
    r = sub.add_parser("predict", help="Prediction and estimation (BCE)")
    r.add_argument("--dataset", default="data_validation.csv")
    r.add_argument("--model", default="saved_model.npy")
    r.add_argument("--output", default=None, help="csv for saving predictions")
    r.set_defaults(fn=command_predict)

    return p


def main(argv = None):
    args, _ = build_parser().parse_known_args(argv)
    if not getattr(args, "show", False):
        matplotlib.use("Agg")
    args.fn(args)

if __name__ == "__main__":
    main(sys.argv[1:])