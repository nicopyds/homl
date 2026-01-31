import os
import pandas as pd
import sklearn
from sklearn.decomposition import PCA

sklearn.set_config(transform_output="pandas")


FILE_PATH = os.path.abspath(__file__)
DATASETS_PATH = os.path.join(os.path.dirname(FILE_PATH), "datasets")


def main():
    df = pd.read_csv(os.path.join(DATASETS_PATH, "titanic.csv"), index_col=0)

    # SUPONGAMOS QUE ESTE DATSET ES MUY GRANDE
    # Y YO NO PUEDO ANALIZARLO O VISUALIZARLO SÓLO
    # vamos a ejecutar un algoritmo de reducción de la dimensionalidad
    # porque así me será más fácil
    X = df.select_dtypes("number")
    X = X[["Age", "Fare"]]
    X.fillna(0, inplace=True)

    # le decimos al modelo
    # reduce el dataset a la mitad
    # porque la X tiene 6 features/columnas

    model = PCA(n_components=1)
    Xt = model.fit_transform(X)

    print(X.head())
    print(Xt.head())
    print(model.explained_variance_ratio_)
    print(sum(model.explained_variance_ratio_))


if __name__ == "__main__":
    main()
