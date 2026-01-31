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

El punto clave para ML Supervisado es la existencia de un target.
En el ejemplo del fichero example_supervised.py el target en nuestra caso es el dataset
llamado `y` y nuestra matrix `X` son nuestras variables.

`X`: se llaman features, variables, atributos.
`y`: se llaman labels, target, éxito.

El target en problemas supervisados puede ser tanto binario (1 - 0: fraude -
no fraude) como números continuos (- infinito - + infinito: predecir
precio de un activo, el número de ventas que tendrás etc etc etc).

Dicho lo anterior, podemos tener problemas de Aprendizaje Supervisado de 
Clasificación o Regresión.

Aprendizaje No Supervisado

En el aprendizaje no supervisado no existe el target `y`, sólo disponemos de
features.

`X`: se llaman features, variables, atributos.
`y`: NO EXISTE.

Cuando quieres DESCUBRIR tus datos. Dejas hablar a los datos.
Por ejemplo: tengo clientes que visitan mi página web y me gustaría segmentar
a estos clientes en diferentes grupos para personalizar su página web.

Una tarea relacionada con Aprendizaje No Supervisado son las técnicas
de `reducción de la dimensionalidad` (si yo tengo una dataset muy grande (miles de columnas)
como puedo yo "entender" mi dataset de manera más rápida).
Esto se consigue, "proyectando" tu dataset a menos dismensiones (se parece
como una algoritmo de compresión tipo .zip pero no es loseless).
¡ HAY PÉRDIDA DE INFORMACIÓN !

A veces puede ser buena idea, ejecutar una reducción de la dimensionalidad
antes de entrenar un modelo final de clasificaicón o regresión.
Irá más rápido, ocupará menos espacion y/o puede mejorar el resultado.

CUANDO ES MALA IDEA: si trabajas en un entorno regulado y ojo que a veces
la pérdida de información puede ser muy relevante.
¡No toméis el ejemplo de [example_dimensionality_reduction.py](./example_dimensionality_reduction.py)
como ejemplo, los datasets de la vida real son mucho más complejos!

*Normlamente, se usan algorimtos de Unsupervised Learning y/o Reducción de la dimensionalidad como un paso
intermedio en un proyecto de Data Science. (Muchas veces no es el fin en si mismo, sino un medio).*

---

Dentro del mundo de Aprendizaje no Supervisado tenemos "Anomaly Detection" que consiste en encontrar instancias "raras/anómalas" 
dentro de tu dataset. 

Básicamente estos algoritmos, durante el entrenamiento (fit) aprenden a reconocer los patrones "normales/habituales" y cuando
se le muestra nuevas instancias, el algoritmo determina si es o no "anólama" (un ejemplo sería el IsolationForest).

Para más información pueden leer esta [página web](https://scikit-learn.org/stable/modules/outlier_detection.html).

---

Una tarea muy parecida a la anterior es "Novelty Detection". El funcionamiento es muy parecido, durante el fit del modelo
el dataset debe ser muy "limpio" (no debe contener cosas "raros"). De esta manera, el modelo aprende las características de 
las instancias "habituales" y luego es capaz de detectar cualquier novedad.

Por último, tenemos "Association Rule Learning": sirve para analizar grandes volumenes de datos. Un posible caso de uso sería: 
le muestras las ventas de tu tienda al algoritmo y el aprende que cuando se compra salsa barcaboca y patatas chips también se compra
carne. El algoritmo te ayuda en este caso a repensar la distribuciónd de los artículos en la tienda (ponerlos más cerca uno del otro).

---
Semi-supervised Learning: etiquetar los datos en la vida real es muy costoso por este motivo es muy habitual encontrarse con muchos
datos que no tienen el "target" o la "etiqueta". 

No obstante, tu sí que podrías etiquetar unas pocas instancias. Una vez que haces esto, estas etiquetas se pueden propagar al resto
de instancias que el algoritmo detecta que son iguales.

Ejemplo: subes un montón de fotos a Google Foto y un algoritmo no supervisado detecta que la persona A aparece en la foto 1, 2, 33.
Con etiquetar 1 vez esta persona, el algoritmo automáticamente las clasifica a todas. De esta manera podemos llegar a etiquetar todas las
fotos muy rápido y posteriomente utilizar una algoritmo supervisado.

---

Self-supervised Learning: el propio algoritmo aprende a etiquetar los datos.
Por ejemplo: supongamos que tenemos un montón de fotos con diferentes mascotas. Podemos manipular las fotos y ocultar parte de las fotos.
De esta manera, podemos llegar a entrenar un algoritmo que debe aprender a reconstruir esta foto. 

El input en el algoritmo es la foto con una parte oculta y el target es la foto original.
Cuando entrenamos el algoritmo, el será capaz de distinguir entre diferentes mascotas. 

Posteriormente puedes llegar a "especializar" el algoritmo en predecir las diferentes tipos de mascotas. 
Tendrás que suministrar fotos etiquetadas con perros, gatos, peces etc, pero la cantidad será mucho menor porque el algoritmo ya
sabe que es cada mascota (porque cuando reconstruye la foto de un gato no le pone la cara de un perro).

Básicamente estamos usando una técnica de "Transfer Learning".

ChatGPT es un modelo self-supervised porque durante el entrenamiento parte del texto que se muestra al algoritmo está reemplazado/oculto
y el modelo debe aprender a predecir la palabra más probable.

En cierto aspecto, self-supervised se parece mucho a tareas supervisadas (se usa en clasificación o regresión) pero también se parece
a no supervisado (porque usas datasets sin etiquetas). Por este motivo es mejor dejarlo en su propia categoría.

---
Reinforcement learning: es una bestia aparte. En el contexto de RF, hay un agente (agent) que interactua con el ambiente y nosotros le damos
feedback. Cuando el agente hace algo malo se le penaliza (penalty) y cuando hace una tarea bien se le da un premio (reward).
De esta manera el algoritmo aprende por si sólo cual es la mejor decisión a tomar en cada situación (policy).

Ejemplo de algoritmos de RF: [AlphaGo](https://es.wikipedia.org/wiki/AlphaGo#cite_note-1) el algoritmo aprende a jugar al juego de Go por
su cuenta (tras millones de partidas) y muy rápidamente es capaz de ganar a los mejores
[jugadores del mundo](https://www.youtube.com/watch?v=WXuK6gekU1Y&list=PLqYmG7hTraZBy7J_4ynYPc0Ml1RUGcLmD&index=1).

---
Otra forma de clasificar los sistemas de Machine Learning es `online` vs `batch`.

Un algoritmo online es capaz de aprender de un flujo de datos (por ejemplo a través de Gradient Descend).
Otros en cambio sólo puede aprender usando el dataset completo (batch) (por ejemplo Randon Forest).

---
Batch Learning: le muestras al modelo todos los datos que tienes disponibles para que aprenda.
Pasado el train, el modelo se despliega en producción y en este caso ya no se le muestra nuevos ejemplos (para que se actualice).
Sólo predice lo que había visto en el dataset de train. Muchas veces, este tipo de soluciones sirve para la mayoría de los casos. 

Ahora bien, algunas limitaciones de este enfoque:
1. Por lo general, los modelos se deterioran en producción con el tiempo (el mundo sigue cambiando). En este caso, si detectamos que hay
   `data drift/model rot` debemos reentrenar nuestro algoritmo con los datos viejos y con los datos nuevos.
1. Si me modelo debe trabajar con entornos muy cambiantes (predecir la bolsa), quizás no es el mejor enforque porque tendré que reentrenar
   muy a menudo mi modelo y esto lleva tiempo y cuesta dinero.
1. Entrenar este tipo de modelos es muy costoso porque implica grandes volumenes de datos y por lo tanto CPU, memoria etc. 
1. Si mi modelo se debe desplegar en un edge device (smartphone) cuando tenga que reentenar mi model, también tengo que tener los datos
   originales. Esto por lo tanto consumirá mucho espacio en el dispositivo y no es lo ideal.

---
Un sistema online es aquel que aprende o bien de un flujo de 1 instancia o bien de un mini grupo de instancias (llamados mini-batches).
Una de las ventajas principales es que puedes aprender de datasets masivos que no caben en memoria (out of core learning).
Una técnica de online learning es [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) pero hay otras.

Normalmente, se entrena un modelo en un entorno offline (pero puede ser con mini batches) y se despliega en producción pero el modelo
se actualiza (en producción) con los nuevos datos que ve.

Por ejemplo: un sistema de detección de spam, ve nuevos ejemplos de spam y se actualiza online.

Uno de los paramétros clave es la `learning rate` es una paramétro que le dice al modelo "como de rápido se tiene que actualizar a los
nuevos datos".

Si le pones `lr` alta, el modelo se actualiza muy rápido a los nuevos datos y se olvida de los viejos (se llama Catastrophic Forgetting).
Si le pones `lr` baja, el modelo tiene más inercia y se actualiza poco a poco. 

Uno de los principales retos de online learning es mostrarle datos malos cuando está en producción y hacer que su performace baje.
Por ejemplo: alguien intenta engañar a un sistema de spam o de búsqueda.

En estos casos, es importante monitorizar nuestro sistema y cuando su performance baja, apagar el aprendizaje online y cambiar a la versión
anterior.

Lo mismo debemos hacer con los datos, monitorizar y detectar outliers (por ejemplo con un algoritmo de `Anomaly Detection`)Uno de los principales retos de online learning es mostrarle datos malos cuando está en producción y hacer que su performace baje.
Por ejemplo: alguien intenta engañar a un sistema de spam o de búsqueda.

En estos casos, es importante monitorizar nuestro sistema y cuando su performance baja, apagar el aprendizaje online y cambiar a la versión
anterior.

Lo mismo debemos hacer con los datos, monitorizar y detectar outliers (por ejemplo con un algoritmo de `Anomaly Detection`).













