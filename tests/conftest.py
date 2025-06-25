import pytest
from app import create_app
from src.DataBases.MySQL import Base, engine, SessionLocal
from sqlalchemy.orm import sessionmaker
from flask import Flask


@pytest.fixture(scope="module")
def test_app() -> Flask:
    app = create_app()
    app.config.update(
        {
            "TESTING": True,
        }
    )
    yield app


@pytest.fixture(scope="module")
def client(test_app):
    return test_app.test_client()


@pytest.fixture(scope="module")
def db():
    # Crea tablas temporalmente
    Base.metadata.create_all(bind=engine)
    yield SessionLocal()
    Base.metadata.drop_all(bind=engine)
