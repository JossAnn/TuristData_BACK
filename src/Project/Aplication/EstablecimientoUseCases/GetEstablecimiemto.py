class GetEstablecimientos:
    def __init__(self, repository):
        self.repository = repository

    def execute_all(self):
        return self.repository.get_all()

    def execute_by_id(self, id_):
        return self.repository.get_by_id(id_)
    def obtener_por_administrador(self, id_administrador):
        return self.repository.get_by_administrador(id_administrador)
