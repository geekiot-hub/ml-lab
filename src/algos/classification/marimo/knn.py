import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium", app_title="K-Nearest Neighboor")


@app.cell
def _():
    import numpy as np
    from matplotlib import pyplot as plt
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    import marimo as mo

    return (
        KNeighborsClassifier,
        StandardScaler,
        load_iris,
        mo,
        np,
        plt,
        train_test_split,
    )


@app.cell
def _():
    from utils import plot_decision_regions_2features

    return (plot_decision_regions_2features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Скачиваем датасет и нормализуем данные
    """)
    return


@app.cell
def _(StandardScaler, load_iris):
    iris = load_iris()

    X = iris.data[:, [2, 3]]
    scaler = StandardScaler()
    scaler.fit(X)
    X_std = scaler.transform(X)

    y = iris.target
    return X_std, y


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Делим данные на тестовые и тренировочные
    """)
    return


@app.cell
def _(X_std, np, train_test_split, y):
    X_train_std, X_test_std, y_train, y_test = train_test_split(
        X_std,
        y,
        random_state=1,
        stratify=y,
        test_size=0.3,
    )

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    return X_combined_std, X_test_std, X_train_std, y_combined, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Обучаем модель KNN
    """)
    return


@app.cell
def _(KNeighborsClassifier, X_train_std, y_train):
    knn = KNeighborsClassifier(
        n_neighbors=5,
        p=2,
        metric="minkowski",
    )
    knn.fit(X_train_std, y_train)
    return (knn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Отрисовка границ принятия решений
    """)
    return


@app.cell
def _(X_combined_std, knn, plot_decision_regions_2features, plt, y_combined):
    plot_decision_regions_2features(
        X_combined_std,
        y_combined,
        classifier=knn,
        test_idx=range(105, 150),
    )
    plt.xlabel("Длина лепестка [см]")
    plt.ylabel("Ширина лепестка [см]")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Итоговая метрика точности
    """)
    return


@app.cell(hide_code=True)
def _(X_test_std, knn, mo, y_test):
    accuracy = knn.score(X_test_std, y_test)
    all_test_cnt = y_test.shape[0]
    correct_test_cnt = int(accuracy * all_test_cnt)

    mo.md(
        f"### Точность: {accuracy}\n"
        f"### Правильно: {correct_test_cnt}/{all_test_cnt}"
    )
    return


if __name__ == "__main__":
    app.run()
