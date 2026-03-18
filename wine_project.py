import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42
DATA_FILE = "winequality-white.csv"
FIG_DIR = "figures"
OUT_DIR = "outputs"

def make_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

def load_data(path):
    return pd.read_csv(path, sep=";")

def to_three_class_label(q):
    if q <= 5:
        return 0
    elif q == 6:
        return 1
    else:
        return 2

def class_label_name(i):
    return {0: "Low (<=5)", 1: "Medium (=6)", 2: "High (>=7)"}[i]

def make_eda_figures(df):
    grouped = df["quality"].apply(to_three_class_label)

    # Figure 1: original label distribution + grouped label distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    original_counts = df["quality"].value_counts().sort_index()
    axes[0].bar(original_counts.index.astype(str), original_counts.values)
    axes[0].set_title("Original Quality Score Distribution")
    axes[0].set_xlabel("Quality Score")
    axes[0].set_ylabel("Count")

    grouped_counts = grouped.value_counts().sort_index()
    grouped_names = [class_label_name(i) for i in grouped_counts.index]
    axes[1].bar(grouped_names, grouped_counts.values)
    axes[1].set_title("Grouped 3-Class Distribution")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figure1_label_distributions.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 2: correlation heatmap
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    ax.set_title("Feature Correlation Heatmap")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figure2_correlation_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()

def split_data(df):
    X = df.drop(columns=["quality"])
    y = df["quality"].apply(to_three_class_label)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=RANDOM_STATE, stratify=y_trainval
    )

    return X_train, X_val, X_test, X_trainval, y_train, y_val, y_test, y_trainval

def tune_models(X_train, y_train, X_val, y_val):
    best_models = {}
    tuning_rows = []

    # kNN
    best_score = -1
    best_params = None
    for params in ParameterGrid({
        "n_neighbors": [3, 5, 9, 15, 25],
        "weights": ["uniform", "distance"]
    }):
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(**params))
        ])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_val)
        macro_f1 = f1_score(y_val, pred, average="macro")
        acc = accuracy_score(y_val, pred)
        tuning_rows.append(["kNN", str(params), acc, macro_f1])
        if macro_f1 > best_score:
            best_score = macro_f1
            best_params = params
    best_models["kNN"] = {"params": best_params}

    # Logistic Regression
    best_score = -1
    best_params = None
    for params in ParameterGrid({
        "C": [0.01, 0.1, 1, 10],
        "class_weight": [None, "balanced"]
    }):
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=3000,
                solver="lbfgs",
                random_state=RANDOM_STATE,
                **params
            ))
        ])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_val)
        macro_f1 = f1_score(y_val, pred, average="macro")
        acc = accuracy_score(y_val, pred)
        tuning_rows.append(["Logistic Regression", str(params), acc, macro_f1])
        if macro_f1 > best_score:
            best_score = macro_f1
            best_params = params
    best_models["Logistic Regression"] = {"params": best_params}

    # MLP
    best_score = -1
    best_params = None
    for params in ParameterGrid({
        "hidden_layer_sizes": [(32,), (64,), (64, 32)],
        "alpha": [1e-4, 1e-3, 1e-2]
    }):
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                max_iter=2000,
                learning_rate_init=0.001,
                random_state=RANDOM_STATE,
                early_stopping=True,
                n_iter_no_change=20,
                **params
            ))
        ])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_val)
        macro_f1 = f1_score(y_val, pred, average="macro")
        acc = accuracy_score(y_val, pred)
        tuning_rows.append(["MLP", str(params), acc, macro_f1])
        if macro_f1 > best_score:
            best_score = macro_f1
            best_params = params
    best_models["MLP"] = {"params": best_params}

    # Random Forest
    best_score = -1
    best_params = None
    for params in ParameterGrid({
        "n_estimators": [100, 300],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 3, 5],
        "class_weight": [None, "balanced"]
    }):
        model = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        macro_f1 = f1_score(y_val, pred, average="macro")
        acc = accuracy_score(y_val, pred)
        tuning_rows.append(["Random Forest", str(params), acc, macro_f1])
        if macro_f1 > best_score:
            best_score = macro_f1
            best_params = params
    best_models["Random Forest"] = {"params": best_params}

    tuning_df = pd.DataFrame(
        tuning_rows,
        columns=["model", "params", "val_accuracy", "val_macro_f1"]
    )
    tuning_df.to_csv(os.path.join(OUT_DIR, "tuning_results.csv"), index=False)

    return best_models

def build_final_models(best_models):
    models = {
        "kNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(**best_models["kNN"]["params"]))
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=3000,
                solver="lbfgs",
                random_state=RANDOM_STATE,
                **best_models["Logistic Regression"]["params"]
            ))
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                max_iter=2000,
                learning_rate_init=0.001,
                random_state=RANDOM_STATE,
                early_stopping=True,
                n_iter_no_change=20,
                **best_models["MLP"]["params"]
            ))
        ]),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **best_models["Random Forest"]["params"]
        )
    }
    return models

def evaluate_models(models, X_trainval, y_trainval, X_test, y_test):
    rows = []
    confusion_info = {}

    for name, model in models.items():
        model.fit(X_trainval, y_trainval)
        pred = model.predict(X_test)

        acc = accuracy_score(y_test, pred)
        macro_f1 = f1_score(y_test, pred, average="macro")
        weighted_f1 = f1_score(y_test, pred, average="weighted")

        rows.append([name, acc, macro_f1, weighted_f1])
        confusion_info[name] = confusion_matrix(y_test, pred)

    results_df = pd.DataFrame(
        rows,
        columns=["model", "test_accuracy", "test_macro_f1", "test_weighted_f1"]
    ).sort_values(by="test_macro_f1", ascending=False)

    results_df.to_csv(os.path.join(OUT_DIR, "test_results.csv"), index=False)
    return results_df, confusion_info

def make_results_figures(results_df, confusion_info):
    # Figure 3: performance comparison
    plot_df = results_df.copy()
    x = np.arange(len(plot_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, plot_df["test_accuracy"], width, label="Accuracy")
    ax.bar(x + width / 2, plot_df["test_macro_f1"], width, label="Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"], rotation=20)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Test Performance by Model")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figure3_model_performance.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 4: confusion matrices
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    axes = axes.ravel()
    class_names = ["Low", "Medium", "High"]

    for ax, (name, cm) in zip(axes, confusion_info.items()):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(name)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figure4_confusion_matrices.png"), dpi=300, bbox_inches="tight")
    plt.close()

def main():
    make_dirs()
    df = load_data(DATA_FILE)

    make_eda_figures(df)

    X_train, X_val, X_test, X_trainval, y_train, y_val, y_test, y_trainval = split_data(df)
    best_models = tune_models(X_train, y_train, X_val, y_val)
    final_models = build_final_models(best_models)
    results_df, confusion_info = evaluate_models(final_models, X_trainval, y_trainval, X_test, y_test)
    make_results_figures(results_df, confusion_info)

    print("Best hyperparameters:")
    for name, info in best_models.items():
        print(f"{name}: {info['params']}")

    print("\nTest results:")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
