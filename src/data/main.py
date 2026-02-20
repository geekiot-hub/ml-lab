import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    from io import StringIO

    import marimo as mo
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import LabelEncoder

    return LabelEncoder, SimpleImputer, StringIO, mo, np, pd


@app.cell
def _(mo):
    mo.md("""
    ### Формируем синтетический CSV
    """)
    return


@app.cell
def _():
    csv_data = """A,B,C,D
    1.0,2.0,3.0,4.0
    5.0,6.0,,8.0
    10.0,11.0,12.0,"""
    return (csv_data,)


@app.cell
def _(StringIO, csv_data, pd):
    df = pd.read_csv(StringIO(csv_data))
    df
    return (df,)


@app.cell
def _(mo):
    mo.md("""
    ### Обрабатываем NaN значения
    """)
    return


@app.cell
def _(df):
    df.isnull().sum()
    return


@app.cell
def _(df):
    # Простое удаление NaN-содержащих строк
    df.dropna(axis=0)
    return


@app.cell
def _(df):
    # Простое удаление NaN-содержащих столбцов
    df.dropna(axis=1)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Параметры для .dropna()
    """)
    return


@app.cell
def _(df):
    # how='all' # Если все строчки NaN
    df.dropna(axis=1, how="all")
    return


@app.cell
def _(df):
    # thresh=5 # Удаляет строки где меньше 5 реальных значений
    df.dropna(axis=0, thresh=4)
    return


@app.cell
def _(df):
    # subset=List # Ищет NaN только в заданных столбцах
    df.dropna(axis=0, subset=["C"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Метод удаления столбцов не удобен:
    * Риск потери информации.
    * Сокращение кол-ва строк для обучения алгоритма.

    Методы интерполяции - используются для подстановки в пустые строчки значений, чтобы те в дальнейшем можно было использовать:
    * Подстановка среднего (mean imputation), медианы (median imputation), моды (most frequent).
    """)
    return


@app.cell
def _(SimpleImputer, df, np):
    mean_impute = SimpleImputer(missing_values=np.nan, strategy="mean")
    mean_impute.fit(df.to_numpy())

    mean_impute.transform(df.to_numpy())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Работа с категориальными данными
    """)
    return


@app.cell
def _(pd):
    categorial_df = pd.DataFrame(
        [
            ["green", "M", 10.1, "class2"],
            ["red", "L", 13.5, "class1"],
            ["blue", "XL", 15.3, "class2"],
        ]
    )

    categorial_df.columns = ["color", "size", "price", "classlabel"]

    categorial_df
    return (categorial_df,)


@app.cell
def _(categorial_df):
    # from string size to number size.
    size_mapping = {
        "XL": 3,
        "L": 2,
        "M": 1,
    }

    categorial_df["size"] = categorial_df["size"].map(size_mapping)

    categorial_df
    return


@app.cell
def _(categorial_df, np):
    # from string classlabel to numerical classlabel
    class_mapping = {
        label: idx
        for idx, label in enumerate(np.unique(categorial_df["classlabel"]))
    }

    categorial_df["classlabel"] = categorial_df["classlabel"].map(
        class_mapping,
    )

    categorial_df
    return


@app.cell
def _(LabelEncoder, categorial_df):
    # or use numpy
    class_le = LabelEncoder()
    class_le.fit(categorial_df["classlabel"].to_numpy())

    y = class_le.transform(categorial_df["classlabel"].to_numpy())
    y
    return


if __name__ == "__main__":
    app.run()
