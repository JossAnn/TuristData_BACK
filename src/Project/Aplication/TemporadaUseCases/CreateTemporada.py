class CreateTemporada:
    def __init__(self, repository):
        self.repository = repository

    def create_temporada(self, data: dict):
        return self.repository.create(data)
