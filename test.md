---
title: Test
marimo-version: 0.19.9
width: medium
---

```python {.marimo}
import numpy as np
import marimo as mo
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
```

```python {.marimo}
iris = load_iris()
X = iris.data
y = iris.target

X = X[:, [2, 3]]
```

```python {.marimo}
scaler = StandardScaler()
scaler.fit(X)

X_std = scaler.transform(X)
```

# Try to train model. $\Sigma{x_i}$

```python {.marimo}
svm = SVC(gamma=gamma_value_slider.value, C=100.0, kernel="rbf")
```

```python {.marimo}
svm.fit(X_std, y)
mo.md("All done.")
```

```python {.marimo}
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ("o", "s", "^", "v", "<")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")

    cmap = ListedColormap(colors[: len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution),
    )

    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)

    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[cl],
            marker=markers[cl],
            label=f"Class {cl}",
            edgecolor="black",
        )

    if test_idx:
        X_test, _ = X[test_idx, :], y[test_idx]

        plt.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c="none",
            edgecolor="black",
            alpha=1.0,
            linewidth=1,
            marker="o",
            s=100,
            label="Test set",
        )
```

```python {.marimo}
gamma_value_slider = mo.ui.slider(0, 100, 0.1, value=0.1)
```

```python {.marimo}
mo.md(f"""
Gamma value for SVM: {gamma_value_slider}
""")
```

```python {.marimo}
plot_decision_regions(X_std, y, svm)
plt.xlabel("Длина лепестка [standard]")
plt.ylabel("Ширина лепестка [standard]")
plt.tight_layout()
plt.legend(loc="upper left")
plt.show()
```