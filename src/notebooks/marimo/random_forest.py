import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium", app_title="Random Forest", html_head_file="")


@app.cell
def _():
    import numpy as np
    from matplotlib import pyplot as plt
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    import marimo as mo

    return RandomForestClassifier, load_iris, mo, np, plt, train_test_split


@app.cell
def _():
    from utils import plot_decision_regions_2features

    return (plot_decision_regions_2features,)


@app.cell
def _(load_iris):
    iris = load_iris()

    X = iris.data[:, [2, 3]]
    y = iris.target
    return X, y


@app.cell
def _(X, np, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=1,
        test_size=0.3,
        stratify=y,
    )

    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    return X_combined, X_test, X_train, y_combined, y_test, y_train


@app.cell
def _(RandomForestClassifier, X_train, y_train):
    random_forest = RandomForestClassifier(
        n_estimators=25,
        random_state=1,
        n_jobs=2,
    )
    random_forest.fit(X_train, y_train)
    return (random_forest,)


@app.cell
def _(
    X_combined,
    plot_decision_regions_2features,
    plt,
    random_forest,
    y_combined,
):
    plot_decision_regions_2features(
        X=X_combined,
        y=y_combined,
        classifier=random_forest,
        test_idx=range(105, 150),
    )
    plt.xlabel("Длина лепестка [см]")
    plt.ylabel("Ширина лепестка [см]")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(X_test, mo, random_forest, y_test):
    all_cnt = y_test.shape[0]
    correct_cnt = random_forest.score(X_test, y_test) * all_cnt
    mo.md(f"## Итог: {int(correct_cnt)}/{all_cnt}")
    return


if __name__ == "__main__":
    app.run()
