def test_get_establecimientos(client):
    response = client.get("/api/establecimientos")
    assert response.status_code == 200
    assert isinstance(response.json, list)

"""
def test_post_establecimiento(client):
    payload = {
        "nombre": "Test Hotel",
        "direccion": "Dirección de prueba",
        "ciudad": "CDMX",
        "id_tipo": "hotel",
        "horario": "12:00",
        "precio": "150",
        "imagen": "imagen.png",
    }
    response = client.post("/api/establecimientos", json=payload)
    assert response.status_code == 201
    assert response.json["nombre"] == "Test Hotel"
"""