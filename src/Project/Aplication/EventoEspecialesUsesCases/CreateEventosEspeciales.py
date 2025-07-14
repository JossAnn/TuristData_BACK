class CreateEventosEspeciales:
    def __init__(self, repository):
        self.repository = repository

    def create_eventosespeciales(self, id_, nombre, fecha_inicio, fecha_final, descripcion, estado_afectado, id_destino, id_temporada):
        return self.repository.create(
            id_, nombre, fecha_inicio, fecha_final, descripcion, estado_afectado, id_destino, id_temporada
        )