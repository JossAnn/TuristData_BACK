# En este archivo deje las funciones que se deben enrutar
# Se podran usar las funciones despues de que corras el regresionP.py

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from flask import Blueprint, jsonify
import warnings
import random
from pathlib import Path

warnings.filterwarnings("ignore")


class RestaurantPredictor:
    # def __init__(self, model_path="src/Project/Infrastructure/Utils/MineriaPrediccion/restaurant_visit_predictor.pkl"):
    #     self.model_path = model_path
    #     self.model_data = None
    #     self.load_model()
    def __init__(self, model_path=None):
        if model_path is None:
            # Obtener la ruta del directorio donde está este archivo Python
            current_dir = Path(__file__).parent
            self.model_path = current_dir / 'restaurant_visit_predictor.pkl'
        else:
            self.model_path = Path(model_path)  # Convertir a Path object
        
        self.model_data = None
        # NO cargar el modelo aquí, solo cuando se necesite

    # def load_model(self):
    #     # try:
    #     #     self.model_data = joblib.load(self.model_path)
    #     #     #print(f"Modelo cargado exitosamente desde: {self.model_path}")
    #     #     #print("Metricas del modelo:")
    #     #     #for metrica, valor in self.model_data["metrics"].items():
    #     #     #    print(f"  {metrica}: {valor:.4f}")
    #     # except FileNotFoundError:
    #     #     raise FileNotFoundError(f"No se encontró el modelo en: {self.model_path}")
    #     # except Exception as e:
    #     #     raise Exception(f"Error al cargar el modelo: {str(e)}")
    #     try:
    #         print(f"🔍 Intentando cargar modelo desde: {self.model_path}")
    #         print(f"📁 Ruta absoluta: {self.model_path.absolute()}")
    #         print(f"✅ ¿Existe el archivo? {self.model_path.exists()}")
            
    #         if not self.model_path.exists():
    #             # Buscar el archivo en directorios cercanos
    #             possible_paths = [
    #                 Path(__file__).parent / 'restaurant_visit_predictor.pkl',
    #                 Path(__file__).parent.parent / 'restaurant_visit_predictor.pkl',
    #                 Path('/app/src/Project/Infrastructure/Utils/MineriaPrediccion/restaurant_visit_predictor.pkl'),
    #                 Path('/app/restaurant_visit_predictor.pkl')
    #             ]
                
    #             for path in possible_paths:
    #                 print(f"🔍 Buscando en: {path}")
    #                 if path.exists():
    #                     self.model_path = path
    #                     print(f"✅ Modelo encontrado en: {path}")
    #                     break
    #             else:
    #                 raise FileNotFoundError("Modelo no encontrado en ninguna ubicación")
            
    #         self.model_data = joblib.load(self.model_path)
    #         print("🎉 Modelo cargado exitosamente")
            
    #     except Exception as e:
    #         print(f"❌ Error al cargar el modelo: {e}")
    #         raise FileNotFoundError(f"No se encontró el modelo en: {self.model_path}")
    def load_model(self):
        try:
            print(f"🔍 Intentando cargar modelo desde: {self.model_path}")
            print(f"📁 Ruta absoluta: {self.model_path.absolute()}")
            print(f"✅ ¿Existe el archivo? {self.model_path.exists()}")
            
            if not self.model_path.exists():
                # Buscar el archivo en directorios cercanos
                possible_paths = [
                    Path(__file__).parent / 'restaurant_visit_predictor.pkl',
                    Path(__file__).parent.parent / 'restaurant_visit_predictor.pkl',
                    Path('/app/src/Project/Infrastructure/Utils/MineriaPrediccion/restaurant_visit_predictor.pkl'),
                    Path('/app/restaurant_visit_predictor.pkl')
                ]
                
                for path in possible_paths:
                    print(f"🔍 Buscando en: {path}")
                    if path.exists():
                        self.model_path = path
                        print(f"✅ Modelo encontrado en: {path}")
                        break
                else:
                    raise FileNotFoundError("Modelo no encontrado en ninguna ubicación")
            
            self.model_data = joblib.load(self.model_path)
            print("🎉 Modelo cargado exitosamente")
            
        except Exception as e:
            print(f"❌ Error al cargar el modelo: {e}")
            raise FileNotFoundError(f"No se encontró el modelo en: {self.model_path}")

    def prepare_prediction_data(self, data):
        """
        Args:
            data (DataFrame o dict): Datos para predicción

        Returns:
            DataFrame: Datos preparados para el modelo
        """
        # Convertir a DataFrame si viene como diccionario
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        # Validar columnas requeridas
        required_columns = ["busquedas", "festivo", "festividad"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Faltan las siguientes columnas: {missing_columns}")

        # Agregar fecha si no existe (usar fecha actual)
        if "fecha" not in df.columns:
            df["fecha"] = datetime.now().strftime("%Y-%m-%d")
            print("Advertencia: No se proporcionó fecha, usando fecha actual")

        # Convertir fecha a datetime
        df["fecha"] = pd.to_datetime(df["fecha"])

        # Crear características de fecha
        df["dia_semana"] = df["fecha"].dt.dayofweek
        df["mes"] = df["fecha"].dt.month
        df["dia_mes"] = df["fecha"].dt.day

        # Llenar valores nulos en festividad
        df["festividad"] = df["festividad"].fillna("Normal")

        # Codificar festividad usando el encoder entrenado
        festividad_encoded = []
        for fest in df["festividad"]:
            try:
                encoded = self.model_data["label_encoder"].transform([fest])[0]
            except ValueError:
                # Si es una festividad nueva, usar 'Normal'
                print(
                    f"Advertencia: Festividad '{fest}' no reconocida, usando 'Normal'"
                )
                encoded = self.model_data["label_encoder"].transform(["Normal"])[0]
            festividad_encoded.append(encoded)

        # Preparar características en el orden correcto
        X = pd.DataFrame(
            {
                "busquedas": df["busquedas"],
                "festivo": df["festivo"],
                "festividad_encoded": festividad_encoded,
                "dia_semana": df["dia_semana"],
                "mes": df["mes"],
                "dia_mes": df["dia_mes"],
            }
        )

        return X

    def predict(self, data):
        """
        Realiza predicciones de visitas

        Args:
            data: Datos para predicción (DataFrame o dict)

        Returns:
            array: Predicciones de visitas
        """
        # if self.model_data is None:
        #     raise ValueError("El modelo no está cargado")
        if self.model_data is None:
            self.load_model()

        # Preparar datos
        X = self.prepare_prediction_data(data)

        # Aplicar transformación polinomial
        X_poly = self.model_data["poly_features"].transform(X)

        # Realizar predicción
        predictions = self.model_data["model"].predict(X_poly)

        # Asegurar que las predicciones no sean negativas
        predictions = np.maximum(predictions, 0)

        return predictions

    def predict_with_details(self, data):
        """
        Args:
            data: Datos para predicción

        Returns:
            DataFrame: Resultados con detalles de la predicción
        """
        # Convertir a DataFrame si es necesario
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        # Realizar predicciones
        predictions = self.predict(data)

        # Crear DataFrame con resultados
        results = pd.DataFrame(
            {
                "fecha": pd.to_datetime(
                    df.get("fecha", datetime.now().strftime("%Y-%m-%d"))
                ),
                "busquedas": df["busquedas"],
                "festivo": df["festivo"],
                "festividad": df["festividad"],
                "visitas_predichas": np.round(predictions).astype(int),
            }
        )

        # Agregar información adicional
        results["dia_semana"] = results["fecha"].dt.day_name()
        results["es_fin_semana"] = results["fecha"].dt.dayofweek >= 5

        return results

    def get_model_info(self):
        """
        Obtiene información sobre el modelo cargado

        Returns:
            dict: Información del modelo
        """
        if self.model_data is None:
            return None

        info = {
            "grado_polinomio": self.model_data["degree"],
            "caracteristicas_originales": self.model_data["feature_names"],
            "num_caracteristicas_polinomicas": self.model_data[
                "poly_features"
            ].n_output_features_,
            "metricas": self.model_data["metrics"],
            "festividades_conocidas": list(self.model_data["label_encoder"].classes_),
        }

        return info


def predict_single_day(
    busquedas,
    festivo,
    festividad="Normal",
    fecha=None,
    model_path="restaurant_visit_predictor.pkl",
):
    """
    Función de conveniencia para predecir un solo día

    Args:
        busquedas (int): Número de búsquedas
        festivo (int): 1 si es festivo, 0 si no
        festividad (str): Tipo de festividad
        fecha (str): Fecha en formato 'YYYY-MM-DD'
        model_path (str): Ruta al modelo

    Returns:
        int: Predicción de visitas
    """
    predictor = RestaurantPredictor(model_path)

    data = {"busquedas": [busquedas], "festivo": [festivo], "festividad": [festividad]}

    if fecha:
        data["fecha"] = [fecha]

    prediction = predictor.predict(data)[0]
    return int(round(prediction))


def predict_multiple_days(
    csv_file, model_path="restaurant_visit_predictor.pkl", output_file=None
):
    """
    Función para predecir múltiples días desde un archivo CSV

    Args:
        csv_file (str): Ruta al archivo CSV con los datos
        model_path (str): Ruta al modelo
        output_file (str): Ruta para guardar resultados (opcional)

    Returns:
        DataFrame: Resultados de las predicciones
    """
    # Cargar datos
    data = pd.read_csv(csv_file)

    # Crear predictor
    predictor = RestaurantPredictor(model_path)

    # Realizar predicciones
    results = predictor.predict_with_details(data)

    # Guardar resultados si se especifica
    if output_file:
        results.to_csv(output_file, index=False)
        print(f"Resultados guardados en: {output_file}")

    return results


# Funciones auxiliares
# Definir las fechas fijas de las festividades por año
def obtener_festividades_anuales(year):
    return {
        "AñoNuevo": datetime(year, 1, 1),
        "DíaConstitución": datetime(year, 2, 5),
        "BenitoJuárez": datetime(year, 3, 21),
        "DíaMadre": datetime(year, 5, 10),
        "DíaPadre": obtener_dia_padre(year),
        "DíaIndependencia": datetime(year, 9, 16),
        "DíaMuertos": datetime(year, 11, 2),
        "RevoluciónMexicana": datetime(year, 11, 20),
        "Navidad": datetime(year, 12, 25),
        "SemanaSanta": obtener_semana_santa(year),
    }


# Día del padre: tercer domingo de junio
def obtener_dia_padre(year):
    junio = datetime(year, 6, 1)
    tercer_domingo = junio + timedelta(days=(6 - junio.weekday() + 7) % 7 + 14)
    return tercer_domingo


# Semana Santa: aproximación del Viernes Santo (para efectos de simulación)
def obtener_semana_santa(year):
    # Algoritmo de computus (cálculo aproximado de Pascua)
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mes = (h + l - 7 * m + 114) // 31
    dia = ((h + l - 7 * m + 114) % 31) + 1
    pascua = datetime(year, mes, dia)
    viernes_santo = pascua - timedelta(days=2)
    return viernes_santo


def es_festivo(fecha, festividades):
    for nombre, dia in festividades.items():
        if isinstance(
            dia, list
        ):  # Por si Semana Santa fuera lista (ajústalo si es un solo día)
            if fecha.date() in [d.date() for d in dia]:
                return 1, "SemanaSanta"
        elif isinstance(dia, datetime) and fecha.date() == dia.date():
            return 1, nombre
    return (1 if fecha.weekday() >= 5 else 0), "Normal"


def generar_datos(fechas):
    datos = {"fecha": [], "busquedas": [], "festivo": [], "festividad": []}
    festividades = obtener_festividades_anuales(fechas[0].year)
    for fecha in fechas:
        festivo, festividad = es_festivo(fecha, festividades)
        datos["fecha"].append(fecha.strftime("%Y-%m-%d"))
        # datos["busquedas"].append(np.random.poisson(50))
        # Pq asi lo ando en el generador (y de esa manera se apega a los datos sinteticos)
        datos["busquedas"].append(
            np.random.poisson(50)
            + (np.random.randint(30, 100) if festividad != "Normal" else 0)
        )
        datos["festivo"].append(festivo)
        datos["festividad"].append(festividad)
    return datos

""" <=============== ponle un # al inicio para poder ejecutarlo y probarlo sin tener que levantar la app :P
# Uso del predictor
if __name__ == "__main__":
    print("=== MODULO DE PREDICCION PARA RESTAURANTE ===\n")
    predictor = RestaurantPredictor()
    hoy = datetime.today()

    try:
        # 1. Predicción para mañana
        print("1. Predicción para mañana:")
        fecha = hoy + timedelta(days=1)
        festivo, festividad = es_festivo(
            fecha, obtener_festividades_anuales(fecha.year)
        )
        visitas = predict_single_day(
            busquedas=100,
            festivo=festivo,
            festividad=festividad,
            fecha=fecha.strftime("%Y-%m-%d"),
        )
        generar_dato = generar_datos([fecha])
        print(predictor.predict_with_details(generar_dato).to_string(index=False))
        print("\n" + "-" * 50 + "\n")

        # 2. Predicción de lunes a domingo de la próxima semana
        print("2. Predicción de lunes a domingo de la próxima semana:")
        prox_lunes = hoy + timedelta(days=(7 - hoy.weekday()) % 7)
        semana = [prox_lunes + timedelta(days=i) for i in range(7)]
        datos_semana = generar_datos(semana)
        print(predictor.predict_with_details(datos_semana).to_string(index=False))
        print("\n" + "-" * 50 + "\n")

        # 3. Predicción desde mañana hasta el próximo domingo
        print("3. Predicción de mañana hasta el próximo domingo:")
        manana = hoy + timedelta(days=1)
        dias_restantes = 7 if manana.weekday() == 0 else 7 - manana.weekday()
        dias = [manana + timedelta(days=i) for i in range(dias_restantes)]
        datos_hasta_domingo = generar_datos(dias)
        print(
            predictor.predict_with_details(datos_hasta_domingo).to_string(index=False)
        )
        print("\n" + "-" * 50 + "\n")

        # 4. Predicción de la próxima quincena
        print("4. Predicción de la próxima quincena:")
        if hoy.day <= 15:
            # Segunda quincena del mes actual
            inicio_q = datetime(hoy.year, hoy.month, 16)
            mes = hoy.month
            año = hoy.year
        else:
            # Primera quincena del siguiente mes
            mes = hoy.month + 1 if hoy.month < 12 else 1
            año = hoy.year + 1 if mes == 1 else hoy.year
            inicio_q = datetime(año, mes, 1)

        if inicio_q.day == 16:
            # Fin de mes
            if mes == 12:
                fin_q = datetime(año + 1, 1, 1) - timedelta(days=1)
            else:
                fin_q = datetime(año, mes + 1, 1) - timedelta(days=1)
        else:
            # Primera quincena
            fin_q = datetime(año, mes, 15)

        dias_q = [
            inicio_q + timedelta(days=i) for i in range((fin_q - inicio_q).days + 1)
        ]
        datos_q = generar_datos(dias_q)
        print(predictor.predict_with_details(datos_q).to_string(index=False))
        print("\n" + "-" * 50 + "\n")

        # 5. Predicción del próximo mes completo
        print("5. Predicción del próximo mes completo:")
        mes_siguiente = hoy.month + 1 if hoy.month < 12 else 1
        año_siguiente = hoy.year + 1 if mes_siguiente == 1 else hoy.year
        inicio_m = datetime(año_siguiente, mes_siguiente, 1)
        if mes_siguiente == 12:
            fin_m = datetime(año_siguiente + 1, 1, 1) - timedelta(days=1)
        else:
            fin_m = datetime(año_siguiente, mes_siguiente + 1, 1) - timedelta(days=1)
        dias_m = [
            inicio_m + timedelta(days=i) for i in range((fin_m - inicio_m).days + 1)
        ]
        datos_m = generar_datos(dias_m)
        print(predictor.predict_with_details(datos_m).to_string(index=False))
        print("\n" + "-" * 50 + "\n")
        
        # 6. Predicción de un año completo
        print("6. Predicción de un año completo:")
        inicio_a = datetime(hoy.year, hoy.month, hoy.day)
        fin_a = hoy + timedelta(hoy.year % 4 == 0 and 366 or 365)
        dias_a = [inicio_a + timedelta(days=i) for i in range((fin_a - inicio_a).days + 1)]
        datos_a = generar_datos(dias_a)
        print(predictor.predict_with_details(datos_a).to_string(index=False))
        print("\n" + "-" * 50 + "\n")

        # Información del modelo
        print("Información del modelo:")
        info = predictor.get_model_info()
        if info:
            print(f"Grado del polinomio: {info['grado_polinomio']}")
            print(f"Características: {info['caracteristicas_originales']}")
            print(f"Festividades conocidas: {info['festividades_conocidas']}")
            print("Métricas del modelo:")
            for metrica, valor in info["metricas"].items():
                print(f"  {metrica}: {valor:.4f}")

    except Exception as e:
        print(f"Error: {e}")
        print("Asegúrate de que el modelo 'restaurant_visit_predictor.pkl' existe.")
        print("Ejecuta primero el script de entrenamiento.")
"""
# predictor = RestaurantPredictor()
predictor = None

def get_predictor():
    """Función para obtener el predictor, cargándolo solo cuando se necesita"""
    global predictor
    if predictor is None:
        predictor = RestaurantPredictor()
    return predictor

# ================================================================================= Estas son las funciones que se deben enrutar
bp_prediccion = Blueprint("prediccion", __name__)

@bp_prediccion.route("/predecir-manana", methods=["GET"])
def predecir_manana(busquedas=100):
    fecha = datetime.today() + timedelta(days=1)
    festivo, festividad = es_festivo(fecha, obtener_festividades_anuales(fecha.year))
    visitas = predict_single_day(
        busquedas=busquedas,
        festivo=festivo,
        festividad=festividad,
        fecha=fecha.strftime("%Y-%m-%d"),
    )
    datos = generar_datos([fecha])
    pred = predictor.predict_with_details(datos)
    return pred.to_dict(orient="records")

@bp_prediccion.route("/predecir-proxima-semana", methods=["GET"])
def predecir_proxima_semana():
    hoy = datetime.today()
    prox_lunes = hoy + timedelta(days=(7 - hoy.weekday()) % 7)
    semana = [prox_lunes + timedelta(days=i) for i in range(7)]
    datos = generar_datos(semana)
    pred = predictor.predict_with_details(datos)
    return pred.to_dict(orient="records")

@bp_prediccion.route("/predecir-hasta-domingo", methods=["GET"])
def predecir_hasta_domingo():
    hoy = datetime.today()
    manana = hoy + timedelta(days=1)
    dias_restantes = 7 if manana.weekday() == 0 else 7 - manana.weekday()
    dias = [manana + timedelta(days=i) for i in range(dias_restantes)]
    datos = generar_datos(dias)
    pred = predictor.predict_with_details(datos)
    return pred.to_dict(orient="records")

@bp_prediccion.route("/predecir-quincena", methods=["GET"])
def predecir_quincena():
    hoy = datetime.today()
    if hoy.day <= 15:
        inicio_q = datetime(hoy.year, hoy.month, 16)
        mes = hoy.month
        anio = hoy.year
    else:
        mes = hoy.month + 1 if hoy.month < 12 else 1
        anio = hoy.year + 1 if mes == 1 else hoy.year
        inicio_q = datetime(anio, mes, 1)

    if inicio_q.day == 16:
        fin_q = (
            datetime(anio + 1, 1, 1) - timedelta(days=1)
            if mes == 12
            else datetime(anio, mes + 1, 1) - timedelta(days=1)
        )
    else:
        fin_q = datetime(anio, mes, 15)

    dias = [inicio_q + timedelta(days=i) for i in range((fin_q - inicio_q).days + 1)]
    datos = generar_datos(dias)
    pred = predictor.predict_with_details(datos)
    return pred.to_dict(orient="records")

@bp_prediccion.route("/predecir-mes-completo", methods=["GET"])
def predecir_mes_completo():
    hoy = datetime.today()
    mes_siguiente = hoy.month + 1 if hoy.month < 12 else 1
    anio_siguiente = hoy.year + 1 if mes_siguiente == 1 else hoy.year
    inicio = datetime(anio_siguiente, mes_siguiente, 1)
    fin = (
        datetime(anio_siguiente + 1, 1, 1) - timedelta(days=1)
        if mes_siguiente == 12
        else datetime(anio_siguiente, mes_siguiente + 1, 1) - timedelta(days=1)
    )
    dias = [inicio + timedelta(days=i) for i in range((fin - inicio).days + 1)]
    datos = generar_datos(dias)
    pred = predictor.predict_with_details(datos)
    return pred.to_dict(orient="records")

@bp_prediccion.route("/predecir-anio-completo", methods=["GET"])
def predecir_anio_completo():
    hoy = datetime.today()
    inicio = datetime(hoy.year, hoy.month, hoy.day)
    fin = hoy + timedelta(hoy.year % 4 == 0 and 366 or 365)
    dias = [inicio + timedelta(days=i) for i in range((fin - inicio).days + 1)]
    datos = generar_datos(dias)
    pred = predictor.predict_with_details(datos)
    return pred.to_dict(orient="records")

@bp_prediccion.route("/obtener-info-modelo", methods=["GET"])
def obtener_info_modelo():
    return get_predictor().get_model_info()
# =================================================================================
# """

# Pa que cheques el retorno de las funciones
#if __name__ == "__main__":
#    print(predecir_manana()))