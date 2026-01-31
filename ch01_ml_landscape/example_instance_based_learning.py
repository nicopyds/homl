# Este script es un ejemplo básico de
# un modelo de Machine Learning SUPERVISADO Instance Based Learning

import os

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

FILE_PATH = os.path.abspath(__file__)
DATASETS_PATH = os.path.join(os.path.dirname(FILE_PATH), "datasets")


def get_X_y():

    # cargamos nuestro dataset
    df = pd.read_csv(os.path.join(DATASETS_PATH, "titanic.csv"), index_col=0)

    # la y es nuestro target
    # por tanto estamos en ML Supervisado (en este caso de clasificación)
    y = df.pop("Survived")

    # vamos a sacar nuestras features
    # pero para seguir con un ejemplo muy sencillo vamos a tirar
    # sólo de columnas numéricas
    X = df.select_dtypes("number")
    X.fillna(0, inplace=True)

    return X, y


def main():

    # obtengo mi X (features) e y (target)
    X, y = get_X_y()

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # el algoritmo de KNN: k-nearest neighbors
    # es un algoritmo instance based
    # esto lo que implica que el algoritmo usará una métrica de similitud
    # para comprar siempre poder clasificar tus clientes

    # esto es distinto de un algoritmo model based learning
    # donde durante el fit el algoritmo aprende unas reglas determinas

    # esto es muy relevante por varios motivos:
    # 1. el algortimo modelo based aprende 1 vez y predice a partir de las reglas
    #    en cambio el algoritmo instance based NECESITA  un subset o todos los datos
    #    originales para hacer una clasificación (debe comparar contra el ground truth)
    #    Esto funciona bien con pequeños datasets pero no escala para nada bien con muchos datos.
    #    Tampoco funciona bien con datasets con muchas dimensiones (por ejemplo, imágenes o texto).
    #    Por este motivo no se usa mucho porque se vuelve muy lento.

    model = KNeighborsClassifier()

    # En teoría podemos omitir la parte del fit y cuando hacemos el predict, usar el dataset y la fuerza
    # bruta para encontrar el vecino más cercano (pero es una implementación muy poco eficiente).

    # scikit-learn hace otras cosas más inteligentes y es que construye o bien

    # KD-Tree o Ball-Tree
    # https://medium.com/@geethasreemattaparthi/ball-tree-and-kd-tree-algorithms-a03cdc9f0af9

    # KD-Tree: https://www.google.com/search?sca_esv=1fb5ba756bd76908&rlz=1C5CHFA_enES1022ES1022&sxsrf=ANbL-n6_ivKYJYi2oIB-veFNuIHf-MAelg:1769884815090&udm=2&fbs=ADc_l-bpk8W4E-qsVlOvbGJcDwpn60DczFdcvPnuv8WQohHLTaMb_WtLz8zQ41bNqiqMK_0GCDA2eBSrpJajLJh54y7KhefI_dvRXyUnknSrVPAkUiebdeZMsnQIiDvY2RbGM467VORe-GZB7s0qVo2EbQCqu6z19XftDKvJxKS8mznUYmUGsXlAgy55KuDFeqKt0pkAy4uo&q=kd+tree&sa=X&sqi=2&ved=2ahUKEwi9o8ndtraSAxUia0EAHc3OC9AQtKgLegQIFRAB&biw=1920&bih=934&dpr=1&aic=0#sv=CAMSVhoyKhBlLVBpUEJENWpVMkVDLXVNMg5QaVBCRDVqVTJFQy11TToObnE0bGVlRnhCdWxkVE0gBCocCgZtb3NhaWMSEGUtUGlQQkQ1alUyRUMtdU0YADABGAcgzIzBtQcwAkoKCAEQAhgCIAIoAg
    # En el KD-Tree puede calcular por ejemplo la mediana del dataset de train para cada una de las dimensiones
    # de esta manera, cuando llega el momento de predecir algo, puede accelerar mucho la búsqueda de vecinos
    # igualmente vas a tener que usar los datos del train

    # con el Ball-Tree pasa algo parecido
    # https://www.google.com/search?sca_esv=1fb5ba756bd76908&rlz=1C5CHFA_enES1022ES1022&sxsrf=ANbL-n5aBQN_wBfxAMvgyQKm-l8g5ADdbw:1769884913624&udm=2&fbs=ADc_l-bpk8W4E-qsVlOvbGJcDwpn60DczFdcvPnuv8WQohHLTaMb_WtLz8zQ41bNqiqMK_1FsVyo-5Z6JkhWuoPGkuRh7kwNpMSz91P5qhCEeQU0Q41O9aPgAb97ScGu3g8aT69VdIVBtoA_ApLdm7PCSAJJDUJNK4KBEpjhs2P9YgFZjLK9FJw1ESkL1OeRrBRM56JzUCgeQck01zj79aPrcptkZnQeog&q=ball+tree&sa=X&ved=2ahUKEwjlrMeMt7aSAxW-K_sDHci3BiUQtKgLegQIEhAB&biw=1920&bih=934&dpr=1&aic=0#sv=CAMSVhoyKhBlLTRUNmptb0s2Y3gyeUFNMg40VDZqbW9LNmN4MnlBTToOZVoxMXI0Y29oa1lNWE0gBCocCgZtb3NhaWMSEGUtNFQ2am1vSzZjeDJ5QU0YADABGAcgq-HJ5gYwAkoKCAEQAhgCIAIoAg
    # durante el fit, el modelo va seleccionando intancias de manera aleatoria y luego usa un radio para "agrupar" clientes parecidos.
    # de esta manera luego la búsqueda es más rápida

    # En cambio, como veremos con otros algoritmo (por ejemplo: regresión logística), el deploy es más ligero
    # porque el modelo ya aprendió de los datos y únicamente puedo desplegar el modelo (y no el dataset de entrego).

    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    score_ = accuracy_score(y_true=y_test, y_pred=y_test_pred)
    print(f"The accuracy score of the KNN model is {score_}")


if __name__ == "__main__":
    main()
