class CreateEventosEspeciales:
    def __init__(self, repository):
        self.repository = repository

    def create_eventosespeciales(self, evento_data: dict):
        return self.repository.create(evento_data)
    