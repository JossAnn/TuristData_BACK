class EstablecimientoService:
    def __init__(self, use_case):
        self.use_case = use_case

    def listar(self):
        return self.use_case.execute_all()

    def obtener(self, id_):
        return self.use_case.execute_by_id(id_)
    
    def create(self, id_,nombre,direccion,ciudad,tipo,horario,precio,imagen,id_administrador):
        return self.use_case.create_establecimiento(id_,nombre,direccion,ciudad,tipo,horario,precio,imagen,id_administrador)  
