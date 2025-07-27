class GetComentario:
    def __init__(self, repository):
        self.repository = repository

    def listar(self):
        return self.repository.get_all()

    def get_by_establecimiento(self, id_establecimiento):
        return self.repository.get_by_establecimiento(id_establecimiento)
