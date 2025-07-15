import os
#import logging
from flask import Flask#, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
#from sqlalchemy import text
from src.Project.Infrastructure.Routes.register import register_blueprints
#from src.Project.Infrastructure.Repositories.AdministradoRepository import AdministradorRepository

"""
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
"""

load_dotenv()

"""
repository = AdministradorRepository()

def get_db_session():
    return repository.get_connect()
"""

def create_app():
    application = Flask(__name__)

    cors_origin = os.getenv("CORS_ORIGIN", "")
    allowed_origins = [
        origin.strip().rstrip("/")
        for origin in cors_origin.split(",")
        if origin.strip()
    ]

    CORS(
        application,
        supports_credentials=True,
        resources={
            r"/*": {
                "origins": allowed_origins,
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"],
            }
        },
    )

    register_blueprints(application)
    """
    @application.route("/health/database", methods=["GET"])
    def database_health():
        session = None
        try:
            session = get_db_session()
            result = session.execute(text("SELECT 1")).scalar()
            if result == 1:
                return (
                    jsonify(
                        {"status": "ok", "message": "Database connection successful"}
                    ),
                    200,
                )
        except Exception as e:
            logger.error("Error en health check de base de datos", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500
        finally:
            if session:
                session.close()
    """
    return application


if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("PORT"))  # Default 5000
    host = os.getenv("HOST", "0.0.0.0")
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    print("CORS_ORIGIN-pao:", os.getenv("CORS_ORIGIN"))
    app.run(debug=debug_mode, port=port, host=host)
