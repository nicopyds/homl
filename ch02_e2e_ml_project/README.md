En este cápitulo vamos a hacer un proyecto de ML end to end. 

Los principales puntos de un proyecto ML son:
1. Entender el problema (look at the big picture).
1. Obtener los datos (muy relevantes, sin datos no hay ML).
1. Hacer exploración de datos (EDA). 
1. Preparar los datos para el algoritmo.
   NOTA: muchas veces preparamos los datos en función del algoritmo que vamos a 
   usar. La elección de un algoritmo u otro, condiciona el preprocessing. 
1. Entrenar el modelo.
1. Hacer un fine-tunning.
1. Presentar la solución.
1. Desplegar la solución y monitorizar.

El [script principal es el main.py](./main.py)

Vamos a utilizar el dataset de `California Housing Prices`.
Un dataset que contiene los precios del año 1990 de California.

La idea es hacer un modelo que vaya a predecir los precios medianos de las casas 
en cada unidad censal por tanto estamos en un problema de regresión.

---

NOTA: en los problemas de regresión normalmente usamos más de 1 variable/feature para
predecir un target (en nuestro dataset usamos latitud, longitud etc para predecir el
precio mediano de la unidad censal).
En este caso estamos hablando de "multiple regression" (tenemos varias features).
A la vez, si vamos a predecir únicamente 1 target (el precio mediano) estamos delante
de un problema "univariate regression".
Se puede dar el caso, que a partir de tus variables quieres predecir no sólo el precio
`mediano` sino también el precio `medio` y/o el `máximo`. En este caso, estamos
ante "multivariate regression".

Lo mismo aplica a los problemas de clasificación.

Algunas referencias muy útiles de scikit-learn
[MultiOutputRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html)
[MultiOutputClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html)

Dentro de nuestro proyecto, vamos a usar el `RMSE` como métrica.

Mirar el ejemplo de [./example_rmse.py](./example_rmse.py) para entender bien la métrica.


