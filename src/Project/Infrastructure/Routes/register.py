from src.Project.Infrastructure.Controllers.EstablecimientoController import (bp_establecimiento)
from src.Project.Infrastructure.Controllers.TuristController import (bp_turista)
from src.Project.Infrastructure.Controllers.AdministradorController import (bp_administrador)
from src.Project.Infrastructure.Controllers.EventosEspecialesController import (
    bp_eventosespeciales
)
from src.Project.Infrastructure.Utils.upload import (bp_upload)
from src.Project.Infrastructure.Controllers.TemporadaController import (bp_temporadas)
from src.Project.Infrastructure.Controllers.DestinoController import (bp_destinos)

from flask import send_from_directory

def register_blueprints(app):
    app.register_blueprint(bp_establecimiento, url_prefix="/api")
    app.register_blueprint(bp_turista, url_prefix="/api")
    app.register_blueprint(bp_administrador, url_prefix="/api")
    app.register_blueprint(bp_upload, url_prefix="/api") 
    app.register_blueprint(bp_eventosespeciales, url_prefix="/api")
    app.register_blueprint(bp_temporadas, url_prefix="/api")
    app.register_blueprint(bp_destinos, url_prefix="/api")


@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory('uploads', filename)
