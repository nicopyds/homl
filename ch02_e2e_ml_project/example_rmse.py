# RMSE
# muy utilizada en problemas de regresión
# pero algunas características que debemos tener en cuenta:
# 1. penaliza los errores grandes
# 2. relaciona con lo anterior, se ve afectada por outliers

from sklearn.metrics import root_mean_squared_error, mean_absolute_error

# imagina que estos son unos precios
y_true = [10, 15, 12, 100, 25_000_000]

# modelo 1, acierta bastante bien de media
y_pred_1 = [9, 12, 10, 95, 10_000_000]

# modelo 2, acierta todo de maravilla pero falla en 1 predicción
y_pred_2 = [10, 15, 12, 75, 10_000_000]

rmse1 = root_mean_squared_error(y_true=y_true, y_pred=y_pred_1)
rmse2 = root_mean_squared_error(y_true=y_true, y_pred=y_pred_2)

mae1 = mean_absolute_error(y_true=y_true, y_pred=y_pred_1)
mae2 = mean_absolute_error(y_true=y_true, y_pred=y_pred_2)

# pylint: disable=line-too-long
message1 = f"El error del modelo que acierta `más o menos` es de {rmse1} de rmse y {mae1} de mae"
message2 = f"El error del modelo que `perfecto` pero que falla a predecir el outlier de 100 es de {rmse2} de rmse y {mae2} de mae2"

print(message1)
print(message2)
