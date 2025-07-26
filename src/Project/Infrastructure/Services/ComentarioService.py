class ComentarioService:
    def __init__(self, getter, creator):
        self.getter = getter
        self.creator = creator

    def listar(self):
        return self.getter.get_all()

    def create(self, data):
        return self.creator.execute(data)

