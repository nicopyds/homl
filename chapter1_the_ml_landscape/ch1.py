# Este script es un ejemplo básico de
# un modelo de Machine Learning SUPERVISADO

import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification


def get_X_y():
    X, y = make_classification()

    X = pd.DataFrame(data=X).iloc[:, :5]
    X.columns = ["Edad", "Altura", "Peso", "Nómina", "Sexo"]
    print(f"Yo tengo aquí un dataset (features) y es {X.head(5)}")

    y = pd.DataFrame(data=y)
    y.columns = ["Target"]
    print(
        f"Yo tengo otro dataset que contiene el hecho que quiero que mi modelo aprenda {y.head(5)}"
    )

    return X, y


def main():
    X, y = get_X_y()

    model = DecisionTreeClassifier()

    # donde yo antes tenía que picar manualmente todas las reglas para determinar si algo es fraude
    # ahora el modelo va a "aprender" dadas las features X como se relacionan con la y
    # decimos que nuestro proyecto es SUPERVISADO
    # porque le enseño al modelo un montón de ejemplos de
    # compra un producto (1)
    # no compra un producto (0)

    model.fit(X, y)


if __name__ == "__main__":
    main()
