class CreateLugares:
    def __init__(self, repository):
        self.repository = repository

    def create_lugares(self, lugar_data: dict):
        return self.repository.create(lugar_data)
    