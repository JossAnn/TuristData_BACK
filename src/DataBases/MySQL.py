from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os


load_dotenv()

db_url = os.getenv("DBURL")
db_name = os.getenv("DBNAME")
db_user = os.getenv("DBUSER")
db_password = os.getenv("DBPASSW")


DATABASE_URL = f"mysql+pymysql://{db_user}:{db_password}@{db_url}/{db_name}"
engine = create_engine(
    DATABASE_URL,
    pool_size=40,
    max_overflow=80,
    pool_timeout=30,
    # pool_pre_ping=True
)
Base = declarative_base()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
