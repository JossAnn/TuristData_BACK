class LugaresService:
    def __init__(self,creator):
        self.creator = creator

    def create(self, lugar_data: dict):
        return self.creator.create_lugares(lugar_data)

