class ComentarioService:
    def __init__(self, getter, creator):
        self.getter = getter
        self.creator = creator

    def listar(self):
        return self.getter.get_all()
    
    
    def get_by_establecimiento(self, id_establecimiento):
        return self.getter.get_by_establecimiento(id_establecimiento)

    def create(self, data):
        return self.creator.create_comentario(data)


