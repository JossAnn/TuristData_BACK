class DeleteEstablecimiento:
    def __init__(self, repository):
        self.repository = repository

    def delete(self, id_):
        return self.repository.delete(id_)