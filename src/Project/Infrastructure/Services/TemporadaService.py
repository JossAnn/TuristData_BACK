class TemporadaService:
    def __init__(self, creator):
        self.creator = creator

    def register(self, data: dict):
        return self.creator.create_temporada(data)
