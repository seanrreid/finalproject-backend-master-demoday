from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

from utils.config import settings  # Adjusted import based on your project structure

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./final-project.db')

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Define the get_db function
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
