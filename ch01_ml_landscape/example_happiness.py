# Ejemplo de un sistema de ML
# que usa Model Based Learning
# este script sirve como referencia a la hora de entender
# el script de ./example_instance_based_learning.py

import warnings
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings("ignore")

FILE_PATH = os.path.abspath(__file__)
DATASETS_PATH = os.path.join(os.path.dirname(FILE_PATH), "datasets")


def load_data():

    gdp = pd.read_csv(os.path.join(DATASETS_PATH, "gdp_per_capita.csv"), skiprows=4)
    happiness = pd.read_csv(os.path.join(DATASETS_PATH, "happiness_index.csv"))

    return gdp, happiness


def clean_data(gdp, happiness):

    gdp = gdp[["Country Name", "2024"]]
    happiness = happiness[["Country name", "Ladder score"]]

    gdp.columns = ["Name", "GDP_2024"]
    happiness.columns = ["Name", "Happiness_2024"]

    gdp.dropna(axis=0, inplace=True)

    common_countries = list(
        set(gdp["Name"].values.tolist()).intersection(happiness["Name"].values.tolist())
    )

    gdp = gdp[gdp["Name"].isin(common_countries)]
    happiness = happiness[happiness["Name"].isin(common_countries)]

    gdp.sort_values("Name", inplace=True)
    happiness.sort_values("Name", inplace=True)

    gdp.set_index("Name", inplace=True)
    happiness.set_index("Name", inplace=True)

    return gdp, happiness


def main():
    gdp, happiness = load_data()
    gdp, happiness = clean_data(gdp=gdp, happiness=happiness)

    #    print(gdp.head())
    #    print(happiness.head())
    #
    #    print(gdp.shape)
    #    print(happiness.shape)

    # Vimos en el ejemplo de Instance Based
    # que cuando queremos hacer una predicción el KNNClassifier
    # usa el dataset de train para calcular la similitud y hacer su predicción

    # En cambio con el modelo de LinearRegression (o cualquier algoritmo de Model Based Learning)
    # el algoritmo ajusta unos parámetros (en nuestro caso tan sencillo son el model.intercept_ y model.coef_)
    # estos parámetros se ajustan a partir de los datos
    # una vez que has ajustado los params y suponiendo que el modelo es bueno
    # podemos usar únicamente estos params para hacer nuestras predicciones
    # no necesitas recurrir al dataset de origen
    model = LinearRegression()
    model.fit(X=gdp, y=happiness)

    knn_model = KNeighborsRegressor(n_neighbors=3)
    knn_model.fit(X=gdp, y=happiness)

    message = (
        "La felicidad de un país depende de su PIB per Cápita de la siguiente manera\n"
    )
    message += (
        f"Felicidad = {round(model.intercept_[0], 2)} + {model.coef_[0][0]}*PIBpc"
    )
    print(message)

    # PIBpc de Puerto Rico según el autor del libro
    y_pred = model.predict([[33_442.8]])
    print(
        f"La felicidad del país que tiene una PIBpc de 33442.8$ (Puerto Rico) es de {y_pred}"
    )

    # Hacemos la predicción usando el modelo de KNN
    # va a buscar a los 3 países que tienen la renta per cápita más parecida a Puerto Rico
    # y la predicción será el promedio
    # Importante: que recurrimos al dataset de origen
    y_pred_knn = knn_model.predict([[33_442.8]])
    print(
        f"La felicidad del país que tiene una PIBpc de 33442.8$ (Puerto Rico) según el modelo de KNN es de {y_pred_knn}"
    )


if __name__ == "__main__":
    main()
