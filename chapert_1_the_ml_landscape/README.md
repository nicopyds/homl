La primera aplicación muy práctica y mainstream del Machine Learning es el filtro de
spam que se implementó de manera masiva en los 90 con el boom del email.

No obstante, antes de esto había aplicaciones un poco más "especializadas" como es el
OCR (Optical Characther Recognition).

Con la mejora de hardware, big data (sobre todo gracias a internet) y con la aparición
de nuevos algortimos, el ML se hizo presente en un montón de aplicaciones.

1. Voice Prompts (Siri)
1. Automatic Translation (Google Traductor)
1. Recomendadores (Netlix, YouTube, todas las RR.SS.) 

Y luego se publicó el paper de Transformer y nació ChatGPT, Gemini etc y ahora mismo
el `chatbot/AI ES EL PRODUCTO`.

¿Que significa que una máquina aprenda?
¿Si yo me descargo toda la Wikipedia en mi ordenador, mi ordenador es más listo?

La respuesta es que NO.

El mundo de ML se puede "dividir" en varios grupos.
1. Supervised (predecir el precio de una casa) vs Unsupervised Learning (segmentar unos clientes).
1. Online vs Batch Learning.
1. Instance Based (KNN) vs Model Based Learning (DecisionTree).

¿Que es el Machine Learning?
`Machine Learning`: es el arte y la ciencia de usar la programación para hacer que nuestras máquinas **APRENDAD DE LOS DATOS**.

Otras definiciones más técnicas del ML son:
`
El ML es el campo que hace que los ordenadores aprendan de los datos SIN SER EXPLICITAMENTE PROGRAMADAS.
`
Arthur Samuel, 1959

Y otra definición un poco más de "ingenería":

`
Decimos que los ordenadores aprenden dada una Experiencia E, para una tarea en concreto T y sus resultados son medibles P si,
el resultado P para la tarea T mejora con E.
`
Tom Mitchell, 1997

Puedes tener un sistema de Machine Learning que va a predecir si tu email es spam o no.
La parte del sistema que aprende y hace predicciones si algo es spam o no se llama el `modelo`. Por ejemplo: un RandomForestClassifier
o una Red Neural se dice que son modelos.

Estos modelos aprenden de datos que se llama `Training Set` (por ejemplos: filas en una tabla).
Cada una de estas filas se llamada `training instance` o `training sample`.
















