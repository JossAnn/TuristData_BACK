class AdministradorService:
    def __init__(self, use_case,login_admin_use_case):
        self.use_case = use_case
        self.login_admin_use_case = login_admin_use_case 
    
    # def register(self, id_,nombre,correo,password):
    #     return self.use_case.registrar_administrador(id_,nombre,correo,password)  

    def register(self, data):  # solo un argumento
        return self.use_case.registrar_administrador(data)
    
    def login(self, correo, password):
        return self.login_admin_use_case.ejecutar(correo, password)