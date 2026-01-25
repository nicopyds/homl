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
   La diferencia clave es si le voy a guiar al algoritmo durante el entrenamiento.
1. Online vs Batch Learning.
   Si mi algoritmo puede aprende "poco a poco/on the fly" o en cambia necesitar un "batch" de ejemplos.
1. Instance Based (KNN) vs Model Based Learning (DecisionTree).
   Si voy a "aprender" una regla de los datos (model based) o bien lo que voy a hacer es "comprar" las instancias
   a la hora de hacer una predicción.

Estas técnicas en realidad no son excluyentes sino que un sistema de ML en producción podría una Red Neural que aprendió de
ejemplos clasificados por personas (Supervisados) pero fue entrenado usando un método batch (Batch Learning).

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

¿Porque necesitamos Machine Learning?
Supongmaos que queremos clasificar si algo es spam o no SIN ML.

Nosotros podríamos mirar un montón de mails de spam, podríamos ver a menudo contienen ciertas palabras: `gratis`, `íncreible` u otras.
También en el nombre del remitente podría contener algún patrón (en vez de poner apple podría poner appple).

Sin ML, tendría que implmentar un montón de `if-else` para poder clasificar un correo como spam.

Vas iterando los dos pasos de antes y despliegas tu sistema.

El spammer a su vez, ve que tiene menos éxito y lo que va a cambiar son sus emails (por ejemplo appple - aapple).

Tu vas a necesitar cambiar tus reglas y es un juego sin fin.

Sin lugar a dudas, tu sistema va a ser MUY MUY COMPLEJO Y TENDRÁ UN MONTÓN DE REGLAS, REGEX, IF-ELSE.

`
Con Machine Learning la gracia está en que no necesitas programar las reglas,
el algoritmo/modelo aprende que es un spam(normalmente son los usuarios quienes han dicho que algo es spam) de los datos sin ser explicitamente programado.
`

Otro uso muy bueno del ML es para problemas que no tienen una solución "fácil" como podría detección de voz.
Si tu quieres un progama que debe detectar la palabra `íncreible` puedes escribir un algoritmo que va medir el sonido y el accento en la í. 
El problema lo tenemos en que esto no va escalar a millones personas de diferentes países y en diferentes entornos.

La mejor solución hoy por hoy es mostrar a un algoritmo miles de millones de ejemplos de muchas palabras y que el aprenda de los datos.

Por último el ML también ayuda a los humanos a aprender. Por ejemplo: despúes de haber visto millones de ejemplos de spam podemos 
inspeccionar nuestro modelo y en este caso ver que combinación de palabras y/o frases hace un spam.

Es decir, ML nos puede revelar nuevas correlaciones o patrones en los datos y nos complementa el `data mining`.

En resumen, Machine Learning es bueno para este tipo de problemas:
1. Cuando tenemos un montón de reglas y listas a mantener (podemos hacer que un modelo aprenda estas reglas de los datos).
1. Entorno muy cambiante. Podemos volver a entrenar nuestro modelo con nuevos datos y actualizarlo.
1. Problemas donde las técnicas tradicionales no danel resultado esperado. Puede ser que un modelo de ML de una mejor solución.
1. Descubrir los patrones en los datos gracias al ML.

Aplicaciones prácticas de Machine Learning.
1. Analizar imágenes en una línea de producción y determinar si tienen o no defecto (CNN, Vision Transformer etc).
1. Chatbot (NLP, RNN, Transformer).
1. Predecir las ventas futuras de una empresa teniendo en cuenta ciertas variables (RandomForest, Redes Neuronales, ARIMA, RNN, Transfomer).
1. Reducir la dimensionalidad de tu dataset para hacer un análisis más rápido (PCA, KMeans).
1. Cluster a tus clientes (DBSCAN, KMeans, Hierarchical Clustering).
1. Detectar fraude (IsolationForest, CatBoost, Red Neuronal).
1. Text summarization (Transformer).
1. Entrenar un agente a que juegue a un juego (Reinforcement Learning).
1. Recomendar productos/vídeos/feed a un cliente: personalización (Collaborative Filtering).

HAY UN MONTÓN DE APLICACIONES PRÁCTICAS DE MACHINE LEARNING ADEMÁS DEL CHATGPT.

Aprendizaje Supervisado


