from src.Project.Infrastructure.Controllers.EstablecimientoController import (
    bp_establecimiento 
)

from src.Project.Infrastructure.Controllers.TuristController import (bp_turista)
from src.Project.Infrastructure.Controllers.AdministradorController import (
    bp_administrador
)


def register_blueprints(app):
    app.register_blueprint(bp_establecimiento, url_prefix="/api")
    app.register_blueprint(bp_turista, url_prefix="/api")
    app.register_blueprint(bp_administrador, url_prefix="/api")
    
