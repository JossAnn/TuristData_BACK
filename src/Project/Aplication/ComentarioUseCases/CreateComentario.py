class CreateComentario:
    def __init__(self, repository):
        self.repository = repository

    def create_comentario(self, comentario_data: dict):
        return self.repository.create(comentario_data)
    