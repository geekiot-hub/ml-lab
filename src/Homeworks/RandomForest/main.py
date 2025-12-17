import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Загрузка данных
base_features = [
    "PassengerId",
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "Survived",
]

# Только тренировочные данные для обучения
train_df = pd.read_csv("./titanic_train.csv")[base_features]

# Тестовые данные БЕЗ Survived для финального предсказания
test_df = pd.read_csv("./titanic_test.csv")[base_features[:-1]]  # Без Survived

# Gender submission нужен только для проверки ПОСЛЕ предсказаний
gender_submission = pd.read_csv("./gender_submission.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)


# 2. Функция предобработки
def update_df(df: pd.DataFrame, is_train=True, encoders=None) -> pd.DataFrame:
    """Обработка DataFrame"""
    updated_df = df.copy()

    if encoders is None:
        encoders = {}

    # Обработка Sex
    if is_train or "Sex" not in encoders:
        le_sex = LabelEncoder()
        updated_df["Sex"] = le_sex.fit_transform(updated_df["Sex"])
        encoders["Sex"] = le_sex
    else:
        updated_df["Sex"] = encoders["Sex"].transform(updated_df["Sex"])

    # Обработка Embarked
    if is_train or "Embarked" not in encoders:
        le_embarked = LabelEncoder()
        # Заполняем пропуски перед кодированием
        updated_df["Embarked"] = updated_df["Embarked"].fillna("S")
        updated_df["Embarked"] = le_embarked.fit_transform(
            updated_df["Embarked"]
        )
        encoders["Embarked"] = le_embarked
    else:
        updated_df["Embarked"] = updated_df["Embarked"].fillna("S")
        updated_df["Embarked"] = encoders["Embarked"].transform(
            updated_df["Embarked"]
        )

    # Семейный размер
    updated_df["Family"] = updated_df["SibSp"] + updated_df["Parch"] + 1

    # Заполнение Fare
    updated_df["Fare"] = updated_df.groupby("Pclass")["Fare"].transform(
        lambda x: x.fillna(x.median())
    )

    # Заполнение Age
    if "Sex" in updated_df.columns:
        updated_df["Age"] = updated_df.groupby(["Pclass", "Sex"])[
            "Age"
        ].transform(lambda x: x.fillna(x.median()))
    else:
        updated_df["Age"] = updated_df.groupby("Pclass")["Age"].transform(
            lambda x: x.fillna(x.median())
        )

    # Выбор финальных признаков
    if "Survived" in updated_df.columns:
        updated_features = [
            "Pclass",
            "Sex",
            "Age",
            "Family",
            "Embarked",
            "Fare",
            "Survived",
        ]
    else:
        updated_features = [
            "Pclass",
            "Sex",
            "Age",
            "Family",
            "Embarked",
            "Fare",
        ]

    # Оставляем только нужные столбцы
    available_features = [
        f for f in updated_features if f in updated_df.columns
    ]
    updated_df = updated_df[available_features]

    return updated_df, encoders


# 3. Обработка тренировочных данных
cleared_train_df, encoders = update_df(train_df, is_train=True)

# 4. Подготовка данных для обучения (ТОЛЬКО из train_df!)
train_X = cleared_train_df.iloc[:, :-1]  # Все кроме последнего столбца
train_y = cleared_train_df["Survived"]  # Последний столбец

print("\nTrain features shape:", train_X.shape)
print("Train target shape:", train_y.shape)

# 5. Разделение на train/val
X_train, X_val, y_train, y_val = train_test_split(
    train_X,
    train_y,
    test_size=0.2,
    random_state=42,
    stratify=train_y,
)

print(f"\nРазмеры данных после split:")
print(f"X_train: {X_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_val: {y_val.shape}")

# 6. Обучение модели
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    min_samples_split=10,
    min_samples_leaf=4,
)

rf_model.fit(X_train, y_train)

# 7. Оценка на валидационной выборке
y_pred = rf_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print(f"\nAccuracy на валидации: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_val, y_pred))

# 8. Обработка тестовых данных для финального предсказания
cleared_test_df, _ = update_df(test_df, is_train=False, encoders=encoders)

# 9. Предсказание на тестовых данных
test_predictions = rf_model.predict(cleared_test_df)

# 10. Создание submission файла
submission = pd.DataFrame(
    {"PassengerId": test_df["PassengerId"], "Survived": test_predictions}
)

# 11. Оценка на gender_submission (только для проверки!)
# В реальном соревновании у вас НЕ БУДЕТ этих правильных ответов
if "Survived" in gender_submission.columns:
    test_accuracy = accuracy_score(
        gender_submission["Survived"], test_predictions
    )
    print(f"\nAccuracy на тестовых данных (для проверки): {test_accuracy:.4f}")
    print("Это оценка на настоящих тестовых данных Kaggle")
    print("В реальной жизни вы не увидите эти правильные ответы!")

# 12. Сохранение результата
submission.to_csv("my_submission.csv", index=False)
print("\nSubmission файл сохранен: my_submission.csv")
