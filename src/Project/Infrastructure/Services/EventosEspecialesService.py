class EventosEspecialesService:
    def __init__(self, creator):
        self.creator = creator

    def create(self, evento_data: dict):
        return self.repository.create(evento_data)

