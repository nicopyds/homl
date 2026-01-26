# Este script es un ejemplo básico de
# un modelo de Machine Learning NO SUPERVISADO

import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_classification


def get_X():
    # Al contrario que en el caso clasificación (supervisado)
    # si yo tengo un problema no supervisado
    # únicamente tenemos la X
    # por esto la y la hemos llamado _
    X, _ = make_classification()

    # Esta matriz X son nuestras variables del dataset
    # Estas variables/atributos/características
    # una empresa las recopila y guarda en las BBDD
    # Puede ser Edad, Altura, Nómina, sus hábitos de gastos etc etc etc
    # Normalmente estas features los generan los Data Scientists
    X = pd.DataFrame(data=X).iloc[:, :5]
    X.columns = ["Edad", "Altura", "Peso", "Nómina", "Sexo"]
    print(f"Yo tengo aquí un dataset (features) y es {X.head(5)}")

    return X


def main():
    X = get_X()

    # En vez de tu tener que visualizar y buscar las relaciones
    # que puedan existir en un datset con muchas features (proceso muy costoso)
    # podemos "delegar" esta tarea a un modelo
    # el modelo aprende/busca estos grupos - relaciones en los datos
    # y nosotros luego lo podemos inspeccionar de manera mucho más fácil
    model = DBSCAN()
    model.fit(X)


if __name__ == "__main__":
    main()
