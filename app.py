import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from src.Project.Infrastructure.Routes.register import register_blueprints
from src.Project.Aplication.LugarTuristicoUsesCases.LugarTuristicoUseCases import LugarTuristicoUseCases

load_dotenv()

def create_app():
    application = Flask(__name__)

    # Configuración CORS específica para TuristData
    CORS(application, 
         origins=[
             "https://turistdata-netlify.app",
             "https://*.netlify.app", 
             "http://localhost:3000",
             "http://localhost:3001"
         ],
         methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
         allow_headers=["Content-Type", "Authorization"],
         supports_credentials=True
    )

    # Manejar peticiones OPTIONS manualmente
    @application.before_request
    def handle_preflight():
        if request.method == "OPTIONS":
            response = jsonify({'status': 'OK'})
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
            response.headers.add('Access-Control-Max-Age', '86400')
            return response

    # Headers para todas las respuestas
    @application.after_request
    def after_request(response):
        origin = request.headers.get('Origin')
        if origin in ['https://turistdata-netlify.app', 'http://localhost:3000']:
            response.headers.add('Access-Control-Allow-Origin', origin)
        else:
            response.headers.add('Access-Control-Allow-Origin', '*')
        
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response

    # Logging para debug
    @application.before_request
    def log_request():
        print(f"🔍 {request.method} {request.path}")
        print(f"Origin: {request.headers.get('Origin')}")
        print(f"Content-Type: {request.headers.get('Content-Type')}")
        print(f"Authorization: {'✅' if request.headers.get('Authorization') else '❌'}")

    register_blueprints(application)
    return application

if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "0.0.0.0")
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    
    try:
        use_case = LugarTuristicoUseCases()
        use_case.ejecutar()
        print("✅ Lugares turísticos guardados con éxito.")
    except Exception as e:
        print(f"⚠️ Error al inicializar lugares: {e}")
    
    print(f"🚀 Servidor iniciando en {host}:{port}")
    print(f"🌐 CORS_ORIGIN: {os.getenv('CORS_ORIGIN')}")
    
    app.run(debug=debug_mode, port=port, host=host)