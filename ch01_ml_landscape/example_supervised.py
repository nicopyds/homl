# Este script es un ejemplo básico de
# un modelo de Machine Learning SUPERVISADO

import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification


def get_X_y():
    X, y = make_classification()

    # Esta matriz X son nuestras variables del dataset
    # Estas variables/atributos/características
    # una empresa las recopila y guarda en las BBDD
    # Puede ser Edad, Altura, Nómina, sus hábitos de gastos etc etc etc
    # Normalmente estas features los generan los Data Scientists
    X = pd.DataFrame(data=X).iloc[:, :5]
    X.columns = ["Edad", "Altura", "Peso", "Nómina", "Sexo"]
    print(f"Yo tengo aquí un dataset (features) y es {X.head(5)}")

    # Esta matriz/vector y es nuestro target
    # Es lo que queremos que nuestro modelo aprenda
    # Este target también provienen de una BBDD
    # Por ejemplo: si yo detectar fraude
    # puedo mirar en mi BBDD aquellas operaciones que han sido
    # reportadas como fraude por parte de otros usuarios
    # y el Data Scientist va a construir el indicar de 1 (fraude)
    # 0 (no fraude) y es lo que vamos a suministrar a nuestro modelo
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
