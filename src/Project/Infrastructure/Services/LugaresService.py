class LugaresService:
    def __init__(self, getter,creator):
        self.creator = creator

    def create(self, lugar_data: dict):
        return self.creator.execute(lugar_data)

