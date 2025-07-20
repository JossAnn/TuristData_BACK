class CreateDestinos:
    def __init__(self, repository):
        self.repository = repository

    # def create_destinos(self, evento_data: dict):
    #     return self.repository.create(evento_data)
    
    def execute(self, evento_data: dict):
        return self.repository.create(evento_data)
    