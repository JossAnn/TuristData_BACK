class EventosEspecialesService:
    def __init__(self, getter,creator):
        self.creator = creator
        self.getter = getter

    def listar(self):
        return self.getter.execute_all()
    def create(self, evento_data: dict):
        return self.creator.execute(evento_data)

