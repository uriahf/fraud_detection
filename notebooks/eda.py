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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # EDA
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This part is dedicated for exploratory data analysis of the data. In order to avoid "overfitting" by exploration, I've splitted the data into train / val / test sets and created basic analysis only over the
    """)
    return


@app.cell(hide_code=True)
def _():
    from fraud_detection.data.loader import read_data

    raw_data = read_data()

    raw_data
    return (raw_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The outcome is exteremely "imbalanced":
    """)
    return


@app.cell
def _(raw_data):
    raw_data["Class"].value_counts()
    return


@app.cell(hide_code=True)
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
    return np, test_data, train_data, validation_data


@app.cell(hide_code=True)
def _(alt, cols, train_data):
    # Exploratory Data Analysis and Visualizations

    from scipy.stats import describe

    # Statistical summary of the data
    _stats_summary = train_data.select(cols).to_dict(as_series=True)


    # Proportion of Class Labels
    _class_counts = (
        train_data.select("Class").to_pandas().value_counts().reset_index()
    )
    _class_counts.columns = ["Class", "count"]

    _class_proportion = (
        alt.Chart(_class_counts)
        .mark_bar()
        .encode(
            x=alt.X("Class:N", title="Transaction Class"),
            y=alt.Y("count:Q", title="Number of Transactions"),
            color=alt.Color("Class:N", legend=alt.Legend(title="Class")),
            tooltip=["count:Q"],
        )
        .properties(width=200, height=300, title="Class Proportion")
    )

    _class_proportion
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Most of the features are continuous, unfortunately we don't have any prior information about their nature and they are described as a result of PCA algorithm. The only features that might be interpratable are "Amount" and "Time".
    """)
    return


@app.cell
def _(train_data):
    train_data.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Because we are facing continuous features it is fairly convinient to explore Correlation Matrix. It is no surprise to find out that the V1-28 features are mostly uncorrelated with one another, while some of them seem to be correlated with the class label.
    """)
    return


@app.cell(hide_code=True)
def _(train_data):
    import polars as pl
    import altair as alt

    cols = train_data.columns

    rows = []
    for c1 in cols:
        for c2 in cols:
            corr_val = train_data.select(pl.corr(c1, c2)).item()
            rows.append({"var1": c1, "var2": c2, "corr": corr_val})

    # 3. Construct DataFrame with orient="row" (removes warning)
    corr_long = pl.DataFrame(rows, orient="row")

    # 4. Altair heatmap
    heatmap = (
        alt.Chart(corr_long)
        .mark_rect()
        .encode(
            x=alt.X("var1:O", sort=cols, title=""),
            y=alt.Y("var2:O", sort=cols, title=""),
            color=alt.Color(
                "corr:Q",
                scale=alt.Scale(
                    domain=[-1, 0, 1], range=["#2166ac", "white", "#b2182b"]
                ),
                legend=alt.Legend(title="Correlation"),
            ),
            tooltip=["var1", "var2", alt.Tooltip("corr:Q", format=".3f")],
        )
        .properties(width=500, height=500)
    )

    heatmap
    return alt, cols, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model Fitting
    """)
    return


@app.cell
def _(test_data, train_data, validation_data):
    X_train = train_data.drop("Class").to_numpy()
    y_train = train_data["Class"].to_numpy()

    X_val = validation_data.drop("Class").to_numpy()
    y_val = validation_data["Class"].to_numpy()

    X_test = test_data.drop("Class").to_numpy()
    y_test = test_data["Class"].to_numpy()
    return X_test, X_train, X_val, y_train, y_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Logistic Regression
    """)
    return


@app.cell(hide_code=True)
def _(X_train, X_val, np, y_train):
    import plotly.express as px
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        max_iter=1000,
        solver="liblinear",  # works well for binary, sparse-ish data
        class_weight="balanced",  # handles the heavy class imbalance
    )

    clf.fit(X_train, y_train)

    preds_logistic = clf.predict_proba(X_val)[:, 1]

    import pandas as pd

    preds_logistic_val = clf.predict_proba(X_val)[:, 1]
    preds_logistic_train = clf.predict_proba(X_train)[:, 1]

    # coefficients from logistic regression
    coeffs = clf.coef_[0]

    # generic feature names
    feature_names = [f"f{i}" for i in range(X_train.shape[1])]

    # build table
    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coeffs,
            "abs_coefficient": np.abs(coeffs),
        }
    ).sort_values("abs_coefficient", ascending=False)


    fig_logistic = px.bar(
        coef_df.sort_values("coefficient"),
        x="coefficient",
        y="feature",
        orientation="h",
        title="Logistic Regression Feature Coefficients",
    )
    fig_logistic.update_layout(template="none", height=600)
    fig_logistic
    return preds_logistic_train, preds_logistic_val, px


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Light GBM
    """)
    return


@app.cell(hide_code=True)
def _(X_test, X_train, X_val, y_train, y_val):
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    lgbm_model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        objective="binary",
        random_state=42,
    )

    lgbm_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[
            early_stopping(stopping_rounds=50),
            log_evaluation(50),
        ],
    )
    preds_lgbm_val = lgbm_model.predict_proba(X_val)[:, 1]
    preds_lgbm_train = lgbm_model.predict_proba(X_train)[:, 1]
    preds_lgbm_test = lgbm_model.predict_proba(X_test)[:, 1]


    from lightgbm import plot_importance
    import matplotlib.pyplot as plt

    # Plotting feature importances for lgbm_model
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_importance(lgbm_model, ax=ax, importance_type="gain", max_num_features=10)
    plt.title("Feature Importance (LGBM Model)")
    plt.ylabel("Features")
    plt.xlabel("Importance Score (Gain)")
    plt.gca()
    return preds_lgbm_train, preds_lgbm_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Compairsion
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Distribution of the Predictions (Only Validation Set)

    In this section I explored the distribution of the predictions of both models on the validation set.

    Stratifying the predictions by Probability Threshold shows a nicely distributed set for the logistic regression model and a very skewed distribution for the light-gbm model. While this is usually considered a disadvandage that translates to poor net-benefit results it seems that on this case the light-gbm performs well.

    Stratifying the prediction by PPCR shows a similar result for both models, they are doing well for the highest risk percentile with a high proportion of Real-Positives. Extanding the Resource constraint beyond the first risk-percentile won't add much to a random-guess baseline strategy.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    fill_color_radio = mo.ui.radio(
        options=["classification_outcome", "reals_labels"],
        value="classification_outcome",
        label="Fill Colors",
    )

    fill_color_radio
    return fill_color_radio, mo


@app.cell(hide_code=True)
def _(mo):
    stratified_by_radio = mo.ui.radio(
        options=["probability_threshold", "ppcr"],
        value="probability_threshold",
        label="Stratified By",
    )

    stratified_by_radio
    return (stratified_by_radio,)


@app.cell(hide_code=True)
def _(mo):
    by = 0.01

    slider_cutoff = mo.ui.slider(start=0, stop=1, step=by, label="Cutoff")
    slider_cutoff
    return by, slider_cutoff


@app.cell(hide_code=True)
def _(mo):
    reference_group_radio = mo.ui.radio(
        options=["logistic", "lgbm"],
        value="logistic",
        label="Model",
    )

    reference_group_radio
    return (reference_group_radio,)


@app.cell(hide_code=True)
def _(
    by,
    fill_color_radio,
    pl,
    preds_lgbm_val,
    preds_logistic_val,
    px,
    reference_group_radio,
    slider_cutoff,
    stratified_by_radio,
    y_val,
):
    from rtichoke import prepare_binned_classification_data


    binned_classification_data = prepare_binned_classification_data(
        probs={"lgbm": preds_lgbm_val, "logistic": preds_logistic_val},
        reals=y_val,
        stratified_by=["ppcr", "probability_threshold"],
        by=by,
    )

    chosen_cutoff_data = (
        binned_classification_data.filter(
            pl.col("chosen_cutoff") == slider_cutoff.value,
            pl.col("stratified_by") == stratified_by_radio.value,
            pl.col("reference_group") == reference_group_radio.value,
        )
        .sort(pl.col("strata"))
        .with_columns(pl.col("reals_estimate").cast(pl.Float64))
    )

    color_discrete_map = {
        "real_positives": "#4C5454",
        "real_competing": "#C880B7",
        "real_negatives": "#E0E0E0",
        "real_censored": "#E3F09B",
        "true_negatives": "#009e73",
        "true_positives": "#009e73",
        "false_negatives": "#FAC8CD",
        "false_positives": "#FAC8CD",
    }

    fig_new = px.bar(
        chosen_cutoff_data,
        x="mid_point",
        y="reals_estimate",
        color=fill_color_radio.value,
        color_discrete_map=color_discrete_map,
        # color="reals_labels",
        # color_discrete_map=color_discrete_map,
        category_orders={
            "reals_labels": list(color_discrete_map.keys())
        },  # fixes domain order
        hover_data=chosen_cutoff_data.columns,  # like tip: true
    )

    fig_new.update_layout(
        barmode="stack",  # stacked bars (use "group" for side-by-side)
        plot_bgcolor="rgba(0,0,0,0)",  # transparent background
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(title=""),
    )

    if stratified_by_radio.value == "probability_threshold":
        vertical_line = slider_cutoff.value
    else:
        vertical_line = 1 - slider_cutoff.value + by / 2

    fig_new.add_vline(
        x=vertical_line,
        line=dict(color="red", width=2, dash="dash"),
        annotation_text=f"Cutoff: {slider_cutoff.value}",
        annotation_position="top right",
    )

    fig_new
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### By Probability Threshold

    Assuming relative cost for each False Positive, the Probability Threshold implies the exchange rate between True-Positives and False-Positives.

    We compare our prediction models to two baseline approaches: Treat-All strategy (i.e, flag every transaction as a Fraud) and Treat-None strategy (i.e flag every transaction as a non-Fraud).

    Because we face an extremely low-prevalence outcome the Treat-All strategy is the worst one even for low probability thresholds. The Treat-None strategy is better than the logistic regression for all probability threshold, while the Light-GBM model performs better than all strategies even for high probability threshold.

    The performance seems to be consistent for both train and validation sets.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    set_to_evaluate_radio = mo.ui.radio(
        options=["train", "validation"],
        value="validation",
        label="Set to Evaluate",
    )

    set_to_evaluate_radio
    return (set_to_evaluate_radio,)


@app.cell(hide_code=True)
def _(
    preds_lgbm_train,
    preds_lgbm_val,
    preds_logistic_train,
    preds_logistic_val,
    set_to_evaluate_radio,
    y_train,
    y_val,
):
    from rtichoke import prepare_performance_data, plot_decision_curve

    train_performance_data_by_threshold = prepare_performance_data(
        probs={"lgbm": preds_lgbm_train, "logistic": preds_logistic_train},
        reals={"lgbm": y_train, "logistic": y_train},
        stratified_by=["probability_threshold"],
        by=0.01,
    )

    validation_performance_data_by_threshold = prepare_performance_data(
        probs={"lgbm": preds_lgbm_val, "logistic": preds_logistic_val},
        reals={"lgbm": y_val, "logistic": y_val},
        stratified_by=["probability_threshold"],
        by=0.01,
    )

    train_decision_curve = plot_decision_curve(train_performance_data_by_threshold)

    validation_decision_curve = plot_decision_curve(
        validation_performance_data_by_threshold
    )

    selected_curve = (
        train_decision_curve
        if set_to_evaluate_radio.value == "train"
        else validation_decision_curve
    )

    selected_curve
    return (prepare_performance_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### By PPCR

    Assuming Flexible Resource Constraint we can now explore the performance of the model on both train / validation sets in the context of Cumulative PPV compared to the expected PPV of a random guess. The light-gbm performs slightly better than the logistic model, but the important thing to notice is the slope for both models between the first and the second risk percentile. The lift drops dramatically, which implies that it might be a good idea to focus only on high-risk observations.
    """)
    return


@app.cell(hide_code=True)
def _(
    preds_lgbm_train,
    preds_lgbm_val,
    preds_logistic_train,
    preds_logistic_val,
    prepare_performance_data,
    set_to_evaluate_radio,
    y_train,
    y_val,
):
    from rtichoke import plot_lift_curve

    train_performance_data_by_ppcr = prepare_performance_data(
        probs={"lgbm": preds_lgbm_train, "logistic": preds_logistic_train},
        reals={"lgbm": y_train, "logistic": y_train},
        stratified_by=["ppcr"],
        by=0.01,
    )

    validation_performance_data_by_ppcr = prepare_performance_data(
        probs={"lgbm": preds_lgbm_val, "logistic": preds_logistic_val},
        reals={"lgbm": y_val, "logistic": y_val},
        stratified_by=["ppcr"],
        by=0.01,
    )

    train_lift_curve = plot_lift_curve(
        train_performance_data_by_ppcr, stratified_by=["ppcr"]
    )

    validation_lift_curve = plot_lift_curve(
        validation_performance_data_by_ppcr, stratified_by=["ppcr"]
    )

    selected_curve_ppcr = (
        train_lift_curve
        if set_to_evaluate_radio.value == "train"
        else validation_lift_curve
    )

    selected_curve_ppcr
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    After considering the limitation of interpratability and the performance of both models I've decided to choose the light-gbm as my final model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Performance for selected Model (Test Set)
    """)
    return


if __name__ == "__main__":
    app.run()
