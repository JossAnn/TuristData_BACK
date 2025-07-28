# Este es para generar los datos sinteticos, hay que correrlo primero (Solo una vez para que te genere el csv)
# Despues de este, corre el archivo de regresionP.py para entrenar el modelo


import pandas as pd
import numpy as np
from datetime import datetime, timedelta


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


# Generar datos diarios
fechas = pd.date_range(start="2020-01-01", end="2025-12-31", freq="D")
datos = []

for fecha in fechas:
    año = fecha.year
    festividades = obtener_festividades_anuales(año)
    es_festividad = 0
    nombre_festividad = ""

    for nombre, fecha_festividad in festividades.items():
        if (
            isinstance(fecha_festividad, datetime)
            and fecha.date() == fecha_festividad.date()
        ):
            es_festividad = 1
            nombre_festividad = nombre
            break

    # Simulación de búsquedas (más en fechas festivas)
    base_busquedas = np.random.poisson(50)
    if es_festividad:
        base_busquedas += np.random.randint(30, 100)

    # Simulación de visitas al restaurante (relacionadas con búsquedas)
    visitas = max(0, int(base_busquedas * np.random.uniform(0.4, 0.8)))

    datos.append(
        {
            "fecha": fecha.date(),
            "busquedas": base_busquedas,
            "visitas": visitas,
            "festivo": es_festividad,
            "festividad": nombre_festividad,
        }
    )

df = pd.DataFrame(datos)
df.to_csv("data_restaurante_2020_2025.csv", index=False)
