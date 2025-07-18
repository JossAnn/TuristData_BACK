import requests
from src.Project.Infrastructure.Models.lugar_turistico import LugarTuristico
from src.DataBases.MySQL import SessionLocal
import os

class GeoapifyService:
    def __init__(self):
        # Opción 1: Variable de entorno
        self.api_key = os.getenv("GEOAPIFY_API_KEY")
        
        # Opción 2: Usar playground key (temporal para pruebas)
        if not self.api_key:
            # Ve a https://apidocs.geoapify.com/playground/places y copia la "Playground API Key"
            self.api_key = "121b5b5c7f534ce08d3d29e34e303420"
        
        print(f"🔑 API Key configurada: {self.api_key[:10]}..." if self.api_key else "❌ No hay API key")
        
        if not self.api_key or "TU_" in self.api_key:
            print("❌ Necesitas configurar una API key válida:")
            print("1. Ve a https://www.geoapify.com/ y crea una cuenta")
            print("2. Obtén tu API key del dashboard")
            print("3. O usa la Playground API Key de https://apidocs.geoapify.com/playground/places")
            raise ValueError("API Key no configurada correctamente")
        
        self.url = "https://api.geoapify.com/v2/places"
        
        # Coordenadas del playground de Geoapify para México
        # Bounding box amplio que cubre más territorio mexicano: [west, south, east, north]
        mexico_bbox = "-101.89850683646785,17.61191599483738,-97.3307428601189,21.51172980180362"
        
        self.params = {
            "categories": "tourism.sights,tourism.attraction,building.historic",
            "filter": f"rect:{mexico_bbox}",
            "lang": "es",
            "limit": 10,
            "apiKey": self.api_key
        }

    def obtener_y_guardar_lugares(self):
        try:
            # Primero ejecuta el test para encontrar parámetros que funcionen
            print("🔍 Buscando configuración óptima...")
            working_params = self.test_api_connection()
            
            if working_params:
                print(f"\n🎯 Usando configuración que funciona...")
                working_params['limit'] = 100  # Aumentar el límite
                response = requests.get(self.url, params=working_params)
            else:
                print("\n🔍 Realizando petición con parámetros por defecto...")
                response = requests.get(self.url, params=self.params)
            
            # Verificar si la respuesta es exitosa
            if response.status_code != 200:
                print(f"❌ Error en la API: {response.status_code}")
                print(f"Respuesta: {response.text}")
                return
            
            data = response.json()
            
            # Verificar si hay features
            features = data.get("features", [])
            # print(f"🔍 Lugares obtenidos: {len(features)}")
            
            if not features:
                print("⚠️  No se encontraron lugares. Intentando con parámetros alternativos...")
                
                # Intentar con parámetros más amplios
                alt_params = {
                    "categories": "tourism",
                    "text": "Mexico City tourism",
                    "limit": 50,
                    "apiKey": self.api_key
                }
                
                response = requests.get(self.url, params=alt_params)
                if response.status_code == 200:
                    data = response.json()
                    features = data.get("features", [])
                    print(f"🔍 Lugares obtenidos con búsqueda alternativa: {len(features)}")
                
                if not features:
                    print("❌ No se encontraron lugares con ninguna configuración")
                    return
            
            session = SessionLocal()
            lugares_guardados = 0
            
            for feature in features:
                props = feature.get("properties", {})
                nombre = props.get("name")
                estado = props.get("state") or props.get("country")
                ciudad = props.get("city")
                
                # Debug: imprimir información del lugar
                print(f"📍 Lugar: {nombre}")
                print(f"   Estado: {estado}")
                print(f"   Ciudad: {ciudad}")
                print(f"   Coordenadas: {feature.get('geometry', {}).get('coordinates', [])}")
                print("---")
                
                # Solo guardar si tiene nombre
                if nombre:
                    lugar = LugarTuristico(
                        nombre=nombre, 
                        estado=estado or ciudad or "México"
                    )
                    session.add(lugar)
                    lugares_guardados += 1
            
            if lugares_guardados > 0:
                print(f"💾 Guardando {lugares_guardados} lugares en la base de datos...")
                session.commit()
                print("✅ Lugares guardados exitosamente!")
            else:
                print("⚠️  No se guardaron lugares (todos sin nombre)")
                
            session.close()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error de conexión: {e}")
        except Exception as e:
            print(f"❌ Error inesperado: {e}")

    def test_api_connection(self):
        """Método para probar la conexión con la API"""
        try:
            # Prueba con diferentes configuraciones
            test_configs = [
                {
                    "name": "México - Bounding Box del Playground",
                    "params": {
                        "categories": "tourism.sights,tourism.attraction,building.historic",
                        "filter": "rect:-101.89850683646785,17.61191599483738,-97.3307428601189,21.51172980180362",
                        "lang": "es",
                        "limit": 10,
                        "apiKey": self.api_key
                    }
                },
                {
                    "name": "Ciudad de México - Círculo",
                    "params": {
                        "categories": "tourism.sights,tourism.attraction",
                        "filter": "circle:-99.1332,19.4326,15000",  # Centro de CDMX con radio 15km
                        "limit": 10,
                        "apiKey": self.api_key
                    }
                },
                {
                    "name": "Estado de México - Más amplio",
                    "params": {
                        "categories": "tourism",
                        "filter": "place:51b2b5e71159a359c05961094dd4d4e43840f00101f901da1e0400000000009203064d657869636f",
                        "limit": 10,
                        "apiKey": self.api_key
                    }
                },
                {
                    "name": "Sin filtro geográfico - Solo categorías",
                    "params": {
                        "categories": "tourism.sights,tourism.attraction",
                        "text": "Mexico City",
                        "limit": 10,
                        "apiKey": self.api_key
                    }
                }
            ]
            
            for config in test_configs:
                print(f"\n🧪 Probando: {config['name']}")
                response = requests.get(self.url, params=config['params'])
                
                print(f"📡 Status Code: {response.status_code}")
                print(f"📄 URL: {response.url}")
                
                if response.status_code == 200:
                    data = response.json()
                    features = data.get('features', [])
                    print(f"✅ Lugares encontrados: {len(features)}")
                    
                    # Mostrar algunos ejemplos
                    for i, feature in enumerate(features[:3]):
                        props = feature['properties']
                        name = props.get('name', 'Sin nombre')
                        city = props.get('city', '')
                        country = props.get('country', '')
                        print(f"  {i+1}. {name} - {city}, {country}")
                    
                    if features:
                        print(f"🎯 Esta configuración funciona!")
                        return config['params']  # Retorna los parámetros que funcionan
                else:
                    print(f"❌ Error: {response.text}")
            
            print("\n⚠️  Ninguna configuración encontró lugares")
            
        except Exception as e:
            print(f"❌ Error en test: {e}")
            
        return None

# Ejemplo de uso
if __name__ == "__main__":
    service = GeoapifyService()
    
    # Primero prueba la conexión
    service.test_api_connection()
    
    # Luego ejecuta el método principal
    # service.obtener_y_guardar_lugares()