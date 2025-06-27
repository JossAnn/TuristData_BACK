from src.Project.Infrastructure.Repositories.EstablecimientoRepository import (
    EstablecimientoRepository,
)
import uuid


def test_create_establecimiento(db):
    repo = EstablecimientoRepository()
    data = {
        "idalta_establecimiento": str(uuid.uuid4()),
        "nombre": "Hotel Central",
        "direccion": "Calle 1",
        "ciudad": "Ciudad MX",
        "id_tipo": "fonda",  # adapta si usas FK
        "horario": "10:00",
        "precio": "$100",
        "imagen": "url",
    }

    result = repo.create(data)
    assert result.nombre == "Hotel Central"
