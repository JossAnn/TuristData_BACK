import re
import numpy as np
import os
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from flask import request, jsonify, Blueprint
from src.Project.Infrastructure.Utils.MineriaTexto.transliterator import transliterator
#from transliterator import transliterator
from src.Project.Infrastructure.Utils.MineriaTexto.cleaner import clean_text
#from cleaner import clean_text
import warnings

# Configuración inicial
# nltk.download("punkt")
warnings.filterwarnings("ignore")

# Palabras clave para cada categoría con pesos de sentimiento
CATEGORY_KEYWORDS = {
    "limpieza": {
        "positivas": [
            "limpio", "limpia", "limpios", "limpias", "ordenado", "ordenada", "aseado", "aseada", 
            "higiene", "higiénico", "higiénica", "pulcro", "pulcra", "impecable", "reluciente",
            "brillante", "depurado", "depurada", "organizado", "organizada", "acicalado", 
            "pulido", "esterilizado", "esterilizada", "nítido", "nítida", "inmaculado", 
            "inmaculada", "lustroso", "desinfectado", "desinfectada", "fresco", "fresca"
        ],
        "negativas": [
            "sucio", "sucia", "desorden", "barro", "polvo", "manchado", "manchada", 
            "cochino", "cochina", "mugroso", "mugrosa", "desarreglado", "descuidado", 
            "descuidada", "asqueroso", "asquerosa", "repugnante", "fétido", "fétida", 
            "maloliente", "apestoso", "apestosa", "pestilente", "desaseado", "desaliñado", 
            "embarrado", "enlodado", "grasiento", "pegajoso", "hediondo", "maloliente", "mohoso", "inmundo"
        ]
    },
    "comida": {
        "positivas": [
            "delicioso", "deliciosa", "sabroso", "sabrosa", "rico", "rica", "gustó", "gusto", 
            "exquisito", "exquisita", "buenísimo", "espectacular", "apetitoso", "apetitosa", 
            "suculento", "suculenta", "tentador", "tentadora", "excelente", "maravilloso", 
            "maravillosa", "divino", "increíble", "jugoso", "jugosa", "tierno", "tierna", 
            "aromático", "aromática", "sabrosón", "sabrosísima", "deleitante", "palatable",
            "bien cocido", "en su punto", "crujiente", "recién hecho", "fresca", "saboreable"
        ],
        "negativas": [
            "asqueroso", "asquerosa", "malo", "mala", "horrible", "feo", "fea", "desagradable", 
            "repugnante", "insípido", "insípida", "pasado", "pasada", "quemado", "quemada", 
            "crudo", "cruda", "soso", "sosa", "podrido", "podrida", "rancio", "rancia", 
            "amargo", "amarga", "agrio", "agria", "indigesto", "descompuesto", "recalentado", 
            "viejo", "frío", "mal cocido", "pasado de sal", "salado", "ácido", "aceitoso", "grasoso",
            "desabrido", "insipidez", "poco apetitoso", "recocido", "tasteless", "repetitivo"
        ]
    },
    "atencion": {
        "positivas": [
            "amable", "educado", "educada", "excelente", "rápido", "rápida", "atento", "atenta", 
            "cordial", "profesional", "eficiente", "servicial", "respetuoso", "respetuosa", 
            "solícito", "solícita", "gentil", "agradable", "considerado", "considerada", 
            "eficaz", "competente", "proactivo", "proactiva", "empático", "empática", "cortés",
            "receptivo", "receptiva", "amigable", "dispuesto", "comprometido", "presente"
        ],
        "negativas": [
            "maleducado", "maleducada", "lento", "lenta", "grosero", "grosera", "descortés", 
            "antipático", "antipática", "déspota", "irrespetuoso", "irrespetuosa", "negligente", 
            "despistado", "brusco", "brusca", "rudo", "pasivo", "indiferente", "incompetente", 
            "desconsiderado", "impaciente", "malhumorado", "altanero", "desganado", "tardado",
            "desinteresado", "torpe", "poco profesional", "inexperto", "prepotente", "falto de tacto"
        ]
    },
    "precio": {
        "positivas": [
            "barato", "barata", "accesible", "económico", "económica", "justo", "justa", 
            "conveniente", "razonable", "asequible", "modesto", "competitivo", "ventajoso", 
            "apropiado", "buen precio", "adecuado", "moderado", "rebajado", "oferta", 
            "promoción", "ganga", "descuento", "precio bajo", "ajustado", "correcto", 
            "buena relación calidad-precio", "por menos", "dan mas"
        ],
        "negativas": [
            "caro", "cara", "costoso", "costosa", "excesivo", "excesiva", "abusivo", "abusiva", 
            "exagerado", "exagerada", "sobreprecio", "desorbitado", "inflado", "prohibitivo", 
            "injustificado", "disparatado", "desproporcionado", "elevado", "elevadísimo", 
            "engañoso", "pésima relación calidad-precio", "impagable", "precio ridículo", "por mas", "dan menos"
        ]
    }
}


NEUTRAL_INDICATORS = {
    "limpieza": [
        "limpieza", "orden", "aspecto", "lugar", "baño", "baños", "suelo", "mesa", 
        "cocina", "ambiente", "instalaciones", "salón", "mobiliario", "higiene", 
        "sanitario", "piso", "pared", "techos", "ventanas", "cubiertos", "servilleta", 
        "mantel", "trapeador", "escoba", "toalla"
    ],
    "comida": [
        "comida", "plato", "ingrediente", "cocina", "sabor", "comedor", "menú", "carta", 
        "restaurante", "bebida", "postre", "entrada", "desayuno", "almuerzo", "cena", 
        "porción", "ración", "receta", "chef", "cocinero", "chef ejecutivo", "buffet", 
        "platillo", "preparación", "ingredientes", "guarnición", "salsas", "pan", "vino"
    ],
    "atencion": [
        "atención", "servicio", "mesero", "camarero", "trato", "empleado", "personal", "staff",
        "camarera", "mesera", "gerente", "anfitrión", "hostess", "recepcionista", "garzón", 
        "mozos", "asistencia", "ayuda", "encargado", "equipo", "cliente", "usuarios", 
        "turno", "reserva", "host", "supervisor"
    ],
    "precio": [
        "precio", "valor", "cuenta", "factura", "costo", "tarifa", "gasto", "pago", 
        "desembolso", "presupuesto", "oferta", "promoción", "descuento", "cobro", "pagar", 
        "dinero", "efectivo", "tarjeta", "monto", "importe", "cuantía",  "pesos", "dolares", "euros", "centavos", "cambio", "recibo", "facturación",
    ]
}


BAD_WORDS = [
    # Español - Insultos comunes
    "maldito", "malditos", "maldita", "malditas", "idiota", "estúpido", "estúpida", "imbécil", "imbéciles", "tonto", "tonta",
    "pendejo", "pendeja", "mierda", "mierder", "joder", "jodido", "jodida", "jodiendo", "chingar",
    "chingado", "chingada", "chingón", "chingona", "verga", "vergas", "culero", "culera", "cabron", "cabrón",
    "cabrona", "carajo", "coño", "hostia", "gilipollas", "zorra", "zorro", "bastardo", "bastarda",
    "cagada", "cagar", "cagón", "cagona", "maricón", "marica", "maricones", "picha", "pendejez", 
    "polla", "pajero", "pajera", "mamón", "mamona", "panocha", "concha", "culito", "culo", "culazo",
    "pinche", "huevón", "huevona", "webon", "webona", "pelotudo", "pelotuda", "boludo", "boluda",
    "güey", "wey", "ñero", "naco", "choto", "chota", "mierdero", "mierdoso", "mierdosa", "puta", "puto", 
    "putita", "putito", "perra", "perro", "perris", "petardo", "imbecil", "pendejazo", "pendejita",
    "estupides", "zanguango", "vago", "bruto", "bruja", "brujo",

    # Palabras sexuales ofensivas o vulgares
    "follar", "fornicar", "penetrar", "coger", "sexo", "cojer", "tragar", "mamar", "chupar", 
    "lamer", "tirar", "garchar", "meter", "sacar", "montar", "encular", "cachonda", "cachondo", 
    "caliente", "arder", "templar", "trancar", "vergazo", "chingazo", "culear", "culeado", 
    "culeada", "culeando", "nalgas", "nalga", "trasero", "traserito", "ano", "anal", "clítoris", 
    "vagina", "pene", "genitales", "testículos", "tetas", "pechos", "chichis", "boobies", 
    "boobs", "trasero", "culo", "culo grande",

    # Palabras ofensivas en inglés (comunes en textos mixtos o redes sociales)
    "fuck", "fucking", "fucker", "motherfucker", "shit", "bullshit", "asshole", "bastard", 
    "bitch", "son of a bitch", "dick", "dickhead", "cock", "pussy", "slut", "whore", 
    "jerk", "suck", "sucker", "retard", "retarded", "damn", "crap", "wanker", "twat", 
    "cum", "nigger", "nigga", "spic", "fag", "faggot",

    # Variantes con números o símbolos usados para evadir filtros
    "m4ldito", "put@", "p3ndejo", "p3rra", "estup1do", "c4bron", "mierd@", "j0der", "ch1ngar", 
    "m4m0n", "c4g4r", "cul3ro", "g1lip0llas", "z0rr@", "v3rg@", "idi0ta", "imb3cil"
]

# Modelo de análisis de sentimientos
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    top_k=1,
)


def censor_text(text: str) -> tuple:
    """Censura groserías en el texto y cuenta ocurrencias"""
    censored = text
    count = 0
    for word in BAD_WORDS:
        pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
        matches = pattern.findall(text)
        if matches:
            censored = pattern.sub("#" * len(word), censored)
            count += len(matches)
    return censored, count

def extract_sentiment_score(result):
    """Extrae la puntuación de sentimiento del resultado del modelo"""
    label = result["label"]
    score = result["score"]
    
    # El modelo nlptown devuelve etiquetas como "1 star", "2 stars", etc.
    if "star" in label:
        return int(label.split()[0])
    else:
        # Mapeo alternativo si no usa el formato de estrellas
        if score > 0.8:
            return 5
        elif score > 0.6:
            return 4
        elif score > 0.4:
            return 3
        elif score > 0.2:
            return 2
        return 1

def analyze_category_sentiment(sentence: str, category: str) -> float:
    """Analiza el sentimiento específico de una categoría en una oración"""
    clean_sentence = clean_text(sentence)
    
    # Verificar palabras positivas y negativas
    positive_score = 0
    negative_score = 0
    
    for pos_word in CATEGORY_KEYWORDS[category]["positivas"]:
        if pos_word in clean_sentence:
            positive_score += 1
    
    for neg_word in CATEGORY_KEYWORDS[category]["negativas"]:
        if neg_word in clean_sentence:
            negative_score += 1
    
    # Si hay palabras específicas de sentimiento, usarlas
    if positive_score > 0 or negative_score > 0:
        if positive_score > negative_score:
            return 4 + min(positive_score * 0.5, 1)  # 4-5
        elif negative_score > positive_score:
            return max(1, 2 - negative_score * 0.5)  # 1-2
        else:
            return 3  # Neutro si hay igual cantidad
    
    # Si no hay palabras específicas, usar el modelo general
    try:
        result = sentiment_analyzer(sentence)[0][0]
        return extract_sentiment_score(result)
    except:
        return 3  # Neutro por defecto

def sentence_mentions_category(sentence: str, category: str) -> bool:
    """Verifica si una oración menciona una categoría específica"""
    clean_sentence = clean_text(sentence)
    
    # Verificar palabras clave específicas de la categoría
    all_keywords = (CATEGORY_KEYWORDS[category]["positivas"] + 
                   CATEGORY_KEYWORDS[category]["negativas"] + 
                   NEUTRAL_INDICATORS[category])
    
    return any(keyword in clean_sentence for keyword in all_keywords)

def analyze_sentiment(text: str) -> dict:
    """Función principal de análisis de sentimientos"""
    # Preprocesamiento
    transli_text = transliterator(text)
    cleaned_text = clean_text(transli_text)
    censored_text, censura_count = censor_text(cleaned_text)

    # Dividir en oraciones
    sentences = sent_tokenize(transli_text, language="spanish")

    # Analizar sentimientos por categoría
    category_scores = {category: [] for category in CATEGORY_KEYWORDS}

    for sentence in sentences:
        for category in CATEGORY_KEYWORDS:
            if sentence_mentions_category(sentence, category):
                score = analyze_category_sentiment(sentence, category)
                category_scores[category].append(score)

    # Calcular puntuaciones finales
    final_scores = {}
    for category, scores in category_scores.items():
        if scores:
            final_scores[category] = round(np.mean(scores))
        else:
            final_scores[category] = None

    # Calcular métricas
    active_scores = [s for s in final_scores.values() if s is not None]
    suma_activas = sum(active_scores) if active_scores else 0
    num_activas = len(active_scores)
    suma_max_activas = num_activas * 5
    suma_max_total = 4 * 5

    promedio_activas = round(suma_activas / num_activas, 2) if num_activas > 0 else 0
    promedio_total = round(suma_activas / 4, 2) if suma_activas > 0 else 0

    return {
        "categorias": final_scores,
        "puntuacion": [suma_activas, suma_max_activas, suma_max_total],
        "promedio": [promedio_activas, promedio_total],
        "censuras": censura_count,
        "textocensurado": censored_text,
        "detalle_oraciones": {
            category: len(scores) for category, scores in category_scores.items()
        },
    }


bp_calificacion = Blueprint("calificacion", __name__)
@bp_calificacion.route("/calificar-comentario", methods=["POST"])
def analizar_sentimiento():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "El JSON debe contener una clave 'text'"}), 400

    texto = data["text"]
    resultado = analyze_sentiment(texto)

    return jsonify({"resultado": resultado})

@bp_calificacion.route("/comentarios-categorizados", methods=["GET"])
def obtener_comentarios_categorizados():
    """
    Endpoint GET que lee comentarios desde un archivo .txt y los categoriza
    por sentimiento (de peores a mejores)
    """
    try:
        # Configuración del archivo
        archivo_comentarios = "opiniones.txt"  # Ajusta la ruta según tu estructura
        
        # Verificar si el archivo existe
        if not os.path.exists(archivo_comentarios):
            return jsonify({
                "success": False,
                "error": "Archivo de comentarios no encontrado",
                "archivo_buscado": archivo_comentarios
            }), 404
        
        # Leer comentarios del archivo
        comentarios_raw = leer_comentarios_desde_txt(archivo_comentarios)
        
        if not comentarios_raw:
            return jsonify({
                "success": False,
                "error": "No se encontraron comentarios en el archivo",
                "total_lineas": 0
            }), 404
        
        # Procesar cada comentario y categorizarlo
        comentarios_procesados = []
        
        for idx, comentario_texto in enumerate(comentarios_raw, 1):
            if comentario_texto.strip():  # Ignorar líneas vacías
                # Analizar sentimiento del comentario
                resultado_analisis = analyze_sentiment(comentario_texto)
                
                # Calcular puntuación general del comentario
                puntuacion_general = resultado_analisis["promedio"][0]
                
                # Clasificar en categoría de sentimiento
                categoria_sentimiento, nivel = clasificar_sentimiento(puntuacion_general)
                
                # Crear objeto de comentario procesado
                comentario_procesado = {
                    "id": idx,
                    "texto_original": comentario_texto.strip(),
                    "texto_censurado": resultado_analisis["textocensurado"],
                    "puntuacion_general": puntuacion_general,
                    "nivel": nivel,
                    "categoria_sentimiento": categoria_sentimiento,
                    "puntuaciones_detalladas": resultado_analisis["categorias"],
                    "censuras": resultado_analisis["censuras"],
                    "metricas": {
                        "suma_puntuacion": resultado_analisis["puntuacion"][0],
                        "suma_maxima": resultado_analisis["puntuacion"][1],
                        "promedio_activas": resultado_analisis["promedio"][0],
                        "promedio_total": resultado_analisis["promedio"][1]
                    },
                    "detalle_oraciones": resultado_analisis["detalle_oraciones"],
                    "categorias_detectadas": [
                        cat for cat, puntaje in resultado_analisis["categorias"].items() 
                        if puntaje is not None
                    ]
                }
                
                comentarios_procesados.append(comentario_procesado)
        
        # Ordenar comentarios de peor a mejor (1 a 5 estrellas)
        comentarios_ordenados = sorted(comentarios_procesados, key=lambda x: x["puntuacion_general"])
        
        # Agrupar por categorías de sentimiento
        comentarios_por_categoria = agrupar_por_categoria(comentarios_procesados)
        
        # Calcular estadísticas generales
        estadisticas = calcular_estadisticas(comentarios_procesados)
        
        # Respuesta estructurada
        respuesta = {
            "success": True,
            "archivo": archivo_comentarios,
            "fecha_procesamiento": datetime.now().isoformat(),
            "estadisticas": estadisticas,
            "comentarios_categorizados": comentarios_por_categoria,
            "todos_comentarios_ordenados": comentarios_ordenados,
            "metadata": {
                "ordenamiento": "de peor a mejor (1-5 estrellas)",
                "categorias": [
                    {"nombre": "pesimo", "rango": "1.0 - 1.4", "descripcion": "Muy insatisfecho", "color": "#ff4444"},
                    {"nombre": "malo", "rango": "1.5 - 2.4", "descripcion": "Insatisfecho", "color": "#ff8800"},
                    {"nombre": "regular", "rango": "2.5 - 3.4", "descripcion": "Neutral", "color": "#ffaa00"},
                    {"nombre": "bueno", "rango": "3.5 - 4.4", "descripcion": "Satisfecho", "color": "#88dd00"},
                    {"nombre": "excelente", "rango": "4.5 - 5.0", "descripcion": "Muy satisfecho", "color": "#00aa00"}
                ]
            }
        }
        
        return jsonify(respuesta), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "Error interno del servidor",
            "mensaje": str(e),
            "tipo_error": type(e).__name__
        }), 500
        
def leer_comentarios_desde_txt(archivo_path: str) -> list:
    """
    Lee comentarios desde un archivo .txt
    Soporta diferentes formatos:
    - Un comentario por línea
    - Comentarios separados por líneas vacías
    - Comentarios numerados (1. comentario, 2. comentario, etc.)
    """
    comentarios = []
    
    try:
        with open(archivo_path, 'r', encoding='utf-8') as archivo:
            contenido = archivo.read()
        
        # Detectar formato del archivo
        if '\n\n' in contenido:
            # Comentarios separados por líneas vacías
            comentarios = [c.strip() for c in contenido.split('\n\n') if c.strip()]
        else:
            # Un comentario por línea
            lineas = contenido.split('\n')
            for linea in lineas:
                linea = linea.strip()
                if linea:
                    # Quitar numeración si existe (1., 2., etc.)
                    import re
                    linea_limpia = re.sub(r'^\d+\.\s*', '', linea)
                    comentarios.append(linea_limpia)
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Archivo no encontrado: {archivo_path}")
    except UnicodeDecodeError:
        # Intentar con otra codificación
        try:
            with open(archivo_path, 'r', encoding='latin1') as archivo:
                contenido = archivo.read()
            comentarios = [c.strip() for c in contenido.split('\n') if c.strip()]
        except:
            raise UnicodeDecodeError("No se pudo leer el archivo con las codificaciones disponibles")
    
    return comentarios

def clasificar_sentimiento(puntuacion_general: float) -> tuple:
    """Clasifica la puntuación en categoría de sentimiento"""
    if puntuacion_general >= 4.5:
        return "excelente", 5
    elif puntuacion_general >= 3.5:
        return "bueno", 4
    elif puntuacion_general >= 2.5:
        return "regular", 3
    elif puntuacion_general >= 1.5:
        return "malo", 2
    else:
        return "pesimo", 1

def agrupar_por_categoria(comentarios_procesados: list) -> dict:
    """Agrupa los comentarios por categoría de sentimiento"""
    comentarios_por_categoria = {
        "pesimo": [],      # 1-1.4 estrellas
        "malo": [],        # 1.5-2.4 estrellas  
        "regular": [],     # 2.5-3.4 estrellas
        "bueno": [],       # 3.5-4.4 estrellas
        "excelente": []    # 4.5-5 estrellas
    }
    
    for comentario in comentarios_procesados:
        categoria = comentario["categoria_sentimiento"]
        comentarios_por_categoria[categoria].append(comentario)
    
    return comentarios_por_categoria


def calcular_estadisticas(comentarios_procesados: list) -> dict:
    """Calcula estadísticas generales de los comentarios"""
    total_comentarios = len(comentarios_procesados)
    
    if total_comentarios == 0:
        return {
            "total_comentarios": 0,
            "promedio_general": 0,
            "distribucion": {},
            "total_censuras": 0
        }
    
    promedio_general = sum(c["puntuacion_general"] for c in comentarios_procesados) / total_comentarios
    total_censuras = sum(c["censuras"] for c in comentarios_procesados)
    
    # Distribución por categorías
    distribucion = {}
    for comentario in comentarios_procesados:
        categoria = comentario["categoria_sentimiento"]
        distribucion[categoria] = distribucion.get(categoria, 0) + 1
    
    # Estadísticas por categoría temática (limpieza, comida, atención, precio)
    categorias_tematicas = {"limpieza": [], "comida": [], "atencion": [], "precio": []}
    
    for comentario in comentarios_procesados:
        for categoria, puntaje in comentario["puntuaciones_detalladas"].items():
            if puntaje is not None:
                categorias_tematicas[categoria].append(puntaje)
    
    promedios_tematicos = {}
    for categoria, puntajes in categorias_tematicas.items():
        if puntajes:
            promedios_tematicos[categoria] = round(sum(puntajes) / len(puntajes), 2)
        else:
            promedios_tematicos[categoria] = None
    
    return {
        "total_comentarios": total_comentarios,
        "promedio_general": round(promedio_general, 2),
        "distribucion": distribucion,
        "total_censuras": total_censuras,
        "promedios_por_categoria_tematica": promedios_tematicos,
        "comentarios_con_censura": sum(1 for c in comentarios_procesados if c["censuras"] > 0),
        "porcentaje_censuras": round((sum(1 for c in comentarios_procesados if c["censuras"] > 0) / total_comentarios) * 100, 2)
    }
def print_results_bonito(result):
    """Imprime los resultados de forma bonita en consola"""

    # Título principal
    print("=" * 60)
    print("ANÁLISIS DE SENTIMIENTOS - RESULTADOS".center(60))
    print()

    # Categorías reconocidas
    print("CATEGORÍAS RECONOCIDAS:")
    print("-" * 50)
    for categoria, cantidad in result["detalle_oraciones"].items():
        print(f"   • {categoria.capitalize():12} -> {cantidad} oraciones")
    print()

    # Estrellas por categoría
    print("PUNTUACIÓN POR CATEGORÍA:")
    print("-" * 50)
    for categoria, puntuacion in result["categorias"].items():
        # Verificar si la puntuación es None o un número válido
        if puntuacion is None:
            print(f"   • {categoria.capitalize():12} -> No puntuado  ──")
        else:
            # Asegurar que la puntuación esté entre 0 y 5
            puntuacion_int = max(0, min(5, int(puntuacion)))
            estrellas = "★" * puntuacion_int + "☆" * (5 - puntuacion_int)
            print(
                f"   • {categoria.capitalize():12} -> {puntuacion:.1f}/5.0  {estrellas}"
            )
    print()

    # Métricas principales
    print("MÉTRICAS PRINCIPALES:")
    print("-" * 50)
    print(f"   • Puntuación total:     {result['puntuacion'][0]:3d}")
    print(f"   • Puntaje promedio:     {result['promedio'][0]:5.2f}")
    print(f"   • Censuras realizadas:  {result['censuras']:3d}")
    print()

    # Estado del texto
    print("ESTADO DEL TEXTO:")
    print("-" * 50)
    if result["censuras"] > 0:
        print(f"   • Se realizaron {result['censuras']} censura(s)")
        print("   • Texto censurado disponible")
    else:
        print("   • Texto limpio, sin censuras")
    print()

    # Resumen final
    total_oraciones = sum(result["detalle_oraciones"].values())
    eficiencia = (result["puntuacion"][0] / result["puntuacion"][1]) * 100

    # Filtrar categorías con puntuación válida para mejor/peor
    categorias_validas = {
        k: v for k, v in result["categorias"].items() if v is not None
    }

    print("RESUMEN:")
    print("-" * 50)
    print(f"   • Total de oraciones:   {total_oraciones}")
    print(f"   • Eficiencia del texto: {eficiencia:.1f}%")

    if categorias_validas:
        mejor_categoria = max(categorias_validas, key=categorias_validas.get)
        peor_categoria = min(categorias_validas, key=categorias_validas.get)
        print(f"   • Mejor Categoría:      {mejor_categoria.capitalize()}")
        print(f"   • Peor Categoría:       {peor_categoria.capitalize()}")
    else:
        print("   • No hay categorías con puntuación válida")

    print()

"""
if __name__ == "__main__":
    # Datos de ejemplo
    data = ""Esos malditos tacos están buenísimos, 
    especialmente los de suadero y tripa, súper jugosos y bien servidos. 
    El sabor es callejero auténtico. Lo único malo es que las mesas estaban algo 
    sucias y no había servilletas a la mano. Aun así, por 3 tacos y un refresco 
    por menos de 70 pesos, no se puede pedir más.""

    # Ejecutar análisis
    result = analyze_sentiment(data)

    # Mostrar resultados bonitos
    print_results_bonito(result)

    # Si quieres ver el texto censurado también:
    if result["censuras"] > 0:
        print("TEXTO CENSURADO:")
        print("-" * 60)
        print(result["textocensurado"])
        print("=" * 60)
"""