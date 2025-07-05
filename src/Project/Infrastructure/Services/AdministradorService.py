class AdministradorService:
    def __init__(self, use_case):
        self.use_case = use_case
    
    # def register(self, id_,nombre,correo,password):
    #     return self.use_case.registrar_administrador(id_,nombre,correo,password)  

    def register(self, data):  # solo un argumento
        return self.use_case.registrar_administrador(data)