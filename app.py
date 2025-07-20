import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from src.Project.Infrastructure.Routes.register import register_blueprints
from src.Project.Aplication.LugarTuristicoUsesCases.LugarTuristicoUseCases import LugarTuristicoUseCases
load_dotenv()

def create_app():
    application = Flask(__name__)

    cors_origin = os.getenv("CORS_ORIGIN", "")
    allowed_origins = "*"

    CORS(
        application,
        supports_credentials=True,
        resources={r"/*": {"origins": "*"}}
    )

    

    register_blueprints(application)

    return application



if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("PORT"))  # Default 5000
    host = os.getenv("HOST", "0.0.0.0")
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    use_case = LugarTuristicoUseCases()
    use_case.ejecutar()
    print("✅ Lugares turísticos guardados con éxito.")
    print("CORS_ORIGIN-pao:", os.getenv("CORS_ORIGIN"))
    app.run(debug=debug_mode, port=port, host=host)
    

