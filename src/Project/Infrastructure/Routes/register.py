from src.Project.Infrastructure.Controllers.EstablecimientoController import (
    bp_establecimiento,
)


def register_blueprints(app):
    app.register_blueprint(bp_establecimiento, url_prefix="/api")
