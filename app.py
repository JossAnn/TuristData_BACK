import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

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

    #Registrar blueprints
    #app.register_blueprint(api)
    #app.register_blueprint(auth)

    return application

if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("PORT"))  # ("PORT", 9000)
    host = os.getenv("HOST", "127.0.0.1")
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(debug=debug_mode, port=port, host=host)
