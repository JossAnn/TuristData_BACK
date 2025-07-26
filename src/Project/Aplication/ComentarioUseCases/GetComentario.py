class GetComentario:
    def __init__(self, repository):
        self.repository = repository

    def execute_all(self):
        return self.repository.get_all()

