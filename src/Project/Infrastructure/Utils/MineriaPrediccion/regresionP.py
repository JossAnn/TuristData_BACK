# Este es para entrenar el modelo de predicción de visitas a un restaurante (Genera busquedas sinteticas igual que el data_creator.py)
# Antes de correr este archivo, debes tener el csv generado con el data_creator.py

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")


class RestaurantVisitPredictor:
    def __init__(self, degree=2):
        """
        Inicializa el predictor de visitas

        Args:
            degree (int): Grado del polinomio para las características
        """
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        self.model = LinearRegression()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.feature_names = None
        self.metrics = {}

    def prepare_data(self, df):
        """
        Args:
            df (DataFrame): DataFrame con los datos del restaurante

        Returns:
            tuple: independiente_x, dependiente_y preparados para el modelo
        """
        # Copia
        data = df.copy()

        # Por si no viene fecha "año-mes-dia"
        if "fecha" in data.columns:
            data["fecha"] = pd.to_datetime(data["fecha"])

        # Caracteristicas adicionales pa manejarlas con mas detalle
        if "fecha" in data.columns:
            data["dia_semana"] = data["fecha"].dt.dayofweek
            data["mes"] = data["fecha"].dt.month
            data["dia_mes"] = data["fecha"].dt.day

        # Manejo de nulos (Pq "festividad" es string y "festivo" int)
        data["festividad"] = data["festividad"].fillna("Normal")

        # Codificar festividad
        festividad_encoded = self.label_encoder.fit_transform(data["festividad"])

        # Seleccionar Xs
        independiente_x = pd.DataFrame(
            {
                "busquedas": data["busquedas"],
                "festivo": data["festivo"],
                "festividad_encoded": festividad_encoded,
            }
        )

        # Agregar caracteristicas de fecha
        if "dia_semana" in data.columns:
            independiente_x["dia_semana"] = data["dia_semana"]
            independiente_x["mes"] = data["mes"]
            independiente_x["dia_mes"] = data["dia_mes"]

        dependiente_y = data["visitas"]

        return independiente_x, dependiente_y

    def entrenador_modelo(self, train_x, train_y):
        # Aplicar transformacion (polinomial lo que esta en el init)
        polino_train_x = self.poly_features.fit_transform(train_x)

        # Entrenar
        self.model.fit(polino_train_x, train_y)

        # Guardar nombres de características
        self.feature_names = train_x.columns.tolist()
        self.is_fitted = True

        # Info nomas
        print(f"Modelo entrenado con {polino_train_x.shape[1]} caracteristicas")
        #print(f"Caracteristicas de entrenamiento {polino_train_x}")
        print(f"Caracteristicas originales: {self.feature_names}")

    def predictor_modelo(self, independientes):
        # Returns: array: Predicciones
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")

        polino_x = self.poly_features.transform(independientes)
        return self.model.predict(polino_x)

    def evaluador_modelo(self, test_x, test_y):
        """
        Evalar modelo con:
        R^2: Pq nos dice que tan general es
        MAE: Pq nos dara el promedio de cuanto se equivoca
        RMSE: Creo que lo voy a quitar

        Args:
            test_x: Características de prueba = busquedas, festivo, festividad_encoded, dia_semana, mes, dia_mes
            test_y: Variable objetivo de prueba = visitas

        Returns:
            dict: Diccionario con las métricas
        """
        y_pred = self.predictor_modelo(test_x)

        r2 = r2_score(test_y, y_pred)
        mae = mean_absolute_error(test_y, y_pred)
        rmse = np.sqrt(mean_squared_error(test_y, y_pred))

        self.metrics = {"R^2": r2, "MAE": mae, "RMSE": rmse}

        return self.metrics

    def ploter_predictions(self, test_x, test_y):
        y_pred = self.predictor_modelo(test_x)

        plt.figure(figsize=(12, 8))

        # Grafico de dispersion: predicciones vs reales
        # La linea roja es la linea de igualdad (donde las predicciones son iguales a los valores reales)
        # Los puntos azules son las predicciones del modelo
        plt.subplot(2, 2, 1)
        plt.scatter(test_y, y_pred, alpha=0.6)
        plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], "r--", lw=2)
        plt.xlabel("Visitas Reales")
        plt.ylabel("Visitas Predichas")
        plt.title("Predicciones vs Valores Reales")

        # Grafico de residuos: Muestra los residuos (real - predicho)
        # La linea horizontal roja es lo mejor que se puede esperar
        # Si los puntos estan alrededor de esta linea, el modelo es bueno; si estan muy alejados, el modelo es malo
        plt.subplot(2, 2, 2)
        residuos = test_y - y_pred
        plt.scatter(y_pred, residuos, alpha=0.6)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Predicciones")
        plt.ylabel("Residuos")
        plt.title("Gráfico de Residuos")
        
        # Grafico temporal: El mas chingon pq compara las visitas reales y las predichas como serie temporal de predicciones
        # La linea azul es el valor real de visitas, la naranja es la predicción del modelo
        plt.subplot(2, 2, 3)
        indices = range(len(test_y))
        plt.plot(indices, test_y.values, label="Real", alpha=0.7)
        plt.plot(indices, y_pred, label="Predicho", alpha=0.7)
        plt.xlabel("Índice")
        plt.ylabel("Visitas")
        plt.title("Comparación Temporal")
        plt.legend()

        # Histograma de residuos: Se debe ver como una campanita alrededor de 0
        # Si no se ve asi, el modelo no es bueno
        plt.subplot(2, 2, 4)
        plt.hist(residuos, bins=20, alpha=0.7, edgecolor="black")
        plt.xlabel("Residuos")
        plt.ylabel("Frecuencia")
        plt.title("Distribución de Residuos")
        #save
        plt.savefig("histograma_residuos.png")

        plt.tight_layout()
        #plt.show()

        # Mostrar metricas
        print("\n" + "=" * 50)
        print("METRICAS DEL MODELO")
        print("=" * 50)
        for metrica, valor in self.metrics.items():
            print(f"{metrica}: {valor:.4f}")
        print("=" * 50)

    def save_model(self, path):
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado antes de guardarlo")

        model_data = {
            "model": self.model,
            "poly_features": self.poly_features,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "degree": self.degree,
            "metrics": self.metrics,
        }

        joblib.dump(model_data, path)
        print(f"Modelo guardado en: {path}")

    @classmethod
    def load_model(cls, modelpath):
        """
        Pa cargar el modelo

        Args:
            modelpath (str): Ruta del modelo guardado

        Returns:
            RestaurantVisitPredictor: Instancia del predictor cargado
        """
        model_data = joblib.load(modelpath)

        predictor = cls(degree=model_data["degree"])
        predictor.model = model_data["model"]
        predictor.poly_features = model_data["poly_features"]
        predictor.label_encoder = model_data["label_encoder"]
        predictor.feature_names = model_data["feature_names"]
        predictor.metrics = model_data["metrics"]
        predictor.is_fitted = True

        return predictor


def main():

    print("Cargando datos del restaurante...")
    # Esto se debe cambiar por la conexion a a los registros de la db
    # df = pd.read_csv("ejemplo_data_restaurante_2020.csv")
    df = pd.read_csv("data_restaurante_2015_2026.csv")

    print("DATOS ENCONTRADOS:")
    print(f"Datos cargados: {len(df)} registros")
    print(f"Rango de fechas: {df['fecha'].min()} a {df['fecha'].max()}") 

    # Crear el predictor
    predictor = RestaurantVisitPredictor(degree=2)

    # Preparar variables (agrega caracteristicas de dia y etiqueta "el tipo" de dia)
    independiente_x, dependiente_y = predictor.prepare_data(df)

    # (entrenar hasta julio 2025)
    # dividiremos los datos temporalmente
    df["fecha"] = pd.to_datetime(df["fecha"])

    train_mask = df["fecha"] <= datetime.today()  # Entrena hasta hoy
    test_mask = df["fecha"] > datetime.today()  # Evalua de hoy en adelante
    # Este rango es para el csv de ejemplo que va de enero a diciembre 2020
    # split_date = "2020-07-01"  # Entrenar con enero-julio, evaluar con agosto-diciembre
    # train_mask = df["fecha"] < split_date
    # test_mask = df["fecha"] >= split_date

    train_x = independiente_x[train_mask]
    train_y = dependiente_y[train_mask]
    test_x = independiente_x[test_mask]
    test_y = dependiente_y[test_mask]

    print(f"\nPara entrenamiento: {len(train_x)} registros")
    print(f"Para prueba: {len(test_x)} registros")

    # Entrenar
    print("\nEntrenando modelo...")
    predictor.entrenador_modelo(train_x, train_y)

    # Evaluar
    print("\nEvaluando modelo...")
    # metrics = predictor.evaluador_modelo(test_x, test_y)
    # print("Metricas del modelo:", metrics)
    predictor.evaluador_modelo(test_x, test_y)

    # Visualizar resultados
    predictor.ploter_predictions(test_x, test_y)

    # Guardar el modelo
    model_path = "restaurant_visit_predictor.pkl"
    predictor.save_model(model_path)

    return 0

if __name__ == "__main__":
    predictor = main()
