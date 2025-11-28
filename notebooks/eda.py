# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.17.0",
#     "pyzmq",
# ]
# ///

import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from fraud_detection.data.loader import read_data

    raw_data = read_data()

    raw_data
    return (raw_data,)


@app.cell
def _(raw_data):
    import numpy as np

    # Set a random seed for reproducibility
    np.random.seed(42)

    # Shuffle the data
    raw_data_sample = raw_data.sample(n=raw_data.height, with_replacement=False)

    # Determine the sizes of each set
    total_rows = raw_data_sample.height
    train_size = int(total_rows * 0.7)  # 70% for training
    validation_size = int(total_rows * 0.15)  # 15% for validation

    # Split the data
    train_data = raw_data_sample[:train_size]
    validation_data = raw_data_sample[train_size : train_size + validation_size]
    test_data = raw_data_sample[train_size + validation_size :]
    return test_data, train_data, validation_data


@app.cell
def _(test_data, validation_data):
    test_data

    validation_data
    return


@app.cell
def _(train_data):
    train_data.describe()
    return


@app.cell
def _(train_data):
    train_data
    return


@app.cell
def _(train_data):
    X = train_data.drop("Class").to_numpy()
    y = train_data["Class"].to_numpy()
    return X, y


@app.cell
def _():
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        max_iter=1000,
        solver="liblinear",  # works well for binary, sparse-ish data
        class_weight="balanced",  # handles the heavy class imbalance
    )
    return (clf,)


@app.cell
def _():
    return


@app.cell
def _(X, clf, y):
    clf.fit(X, y)
    return


@app.cell
def _(X, clf):
    preds = clf.predict_proba(X)[:, 1]
    return (preds,)


@app.cell
def _(preds, y):
    from rtichoke.performance_data.performance_data import prepare_performance_data

    prepare_performance_data(probs={"model": preds}, reals={"model": y}, by=0.1)
    return


@app.cell
def _(preds, y):
    from rtichoke.discrimination.roc import create_roc_curve

    create_roc_curve(probs={"model": preds}, reals={"model": y}, by=0.1)
    return


@app.cell
def _(y):
    y
    return


@app.cell
def _(preds, y):
    from rtichoke.discrimination.precision_recall import (
        create_precision_recall_curve,
    )

    create_precision_recall_curve(
        probs={"model": preds}, reals={"model": y}, by=0.01
    )
    return


@app.cell
def _(pl):
    pl.__version__
    return


@app.cell
def _():
    import rtichoke

    rtichoke.__version__
    return


if __name__ == "__main__":
    app.run()
