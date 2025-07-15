class TuristService:
    def __init__(self, get_turist_use_case, create_turist_use_case,login_turist_use_case):
        self.get_turist = get_turist_use_case
        self.create_turist = create_turist_use_case
        self.login_turist_use_case = login_turist_use_case 

    def obtener(self, id_):
        return self.get_turist.get_user_by_id(id_)

    def register(self, data):
        return self.create_turist.registrar_turist(data)

    def login(self, correo, password):
        return self.login_turist_use_case.ejecutar(correo, password)