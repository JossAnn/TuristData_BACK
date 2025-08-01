# app.py (o tu archivo principal)
from flask import Flask, jsonify
import os

# Importar el setup del modelo
from TuristData_BACK.setup_model import setup_model

# Importar el blueprint
from src.Project.Infrastructure.Utils.MineriaPrediccion.prediccionReutilizable import bp_prediccion

def create_app():
    app = Flask(__name__)
    
    # Configurar el modelo al iniciar la app
    try:
        print("🚀 Configurando modelo de predicción...")
        setup_model()
        print("✅ Modelo configurado correctamente")
    except Exception as e:
        print(f"❌ Error al configurar modelo: {e}")
        # La app puede continuar, pero las predicciones fallarán
    
    # Registrar el blueprint
    app.register_blueprint(bp_prediccion, url_prefix='/api/prediccion')
    
    # Ruta de prueba
    @app.route('/')
    def home():
        return jsonify({
            "message": "API de Predicción de Restaurante",
            "status": "active",
            "endpoints": [
                "/api/prediccion/predecirmanana",
                "/api/prediccion/predecir-proxima-semana",
                "/api/prediccion/predecir-hasta-domingo",
                "/api/prediccion/predecir-quincena", 
                "/api/prediccion/predecir-mes-completo",
                "/api/prediccion/predecir-anio-completo",
                "/api/prediccion/obtener-info-modelo"
            ]
        })
    
    # Ruta para verificar el estado del modelo
    @app.route('/api/status')
    def status():
        model_path = "src/Project/Infrastructure/Utils/MineriaPrediccion/restaurant_visit_predictor.pkl"
        model_exists = os.path.exists(model_path)
        
        return jsonify({
            "model_exists": model_exists,
            "model_path": model_path,
            "working_directory": os.getcwd()
        })
    
    return app

# Para desarrollo local
if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)

# Para producción (Render)
app = create_app()