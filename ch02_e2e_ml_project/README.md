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

