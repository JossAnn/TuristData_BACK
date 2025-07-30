from src.Project.Infrastructure.Controllers.EstablecimientoController import (bp_establecimiento)
from src.Project.Infrastructure.Controllers.TuristController import (bp_turista)
from src.Project.Infrastructure.Controllers.AdministradorController import (bp_administrador)
from src.Project.Infrastructure.Controllers.EventosEspecialesController import (
    bp_eventosespeciales
)
from src.Project.Infrastructure.Utils.upload import (bp_upload)
from src.Project.Infrastructure.Controllers.TemporadaController import (bp_temporadas)
from src.Project.Infrastructure.Controllers.DestinoController import (bp_destinos)
from src.Project.Infrastructure.Controllers.ComentarioController import (bp_comentario)
from src.Project.Infrastructure.Utils.MineriaPrediccion.prediccionReutilizable import (bp_prediccion)
from src.Project.Infrastructure.Utils.MineriaTexto.textminer_v11 import (bp_calificacion)

def register_blueprints(app):
    app.register_blueprint(bp_establecimiento, url_prefix="/api")
    app.register_blueprint(bp_turista, url_prefix="/api")
    app.register_blueprint(bp_administrador, url_prefix="/api")
    app.register_blueprint(bp_upload, url_prefix="/api") 
    app.register_blueprint(bp_eventosespeciales, url_prefix="/api")
    app.register_blueprint(bp_temporadas, url_prefix="/api")
    app.register_blueprint(bp_destinos, url_prefix="/api")
    app.register_blueprint(bp_comentario, url_prefix="/api")
    app.register_blueprint(bp_prediccion, url_prefix="/api")
    app.register_blueprint(bp_calificacion, url_prefix="/api")


