from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd

# Cargar las variables del archivo .env
load_dotenv()

def db_connect():
    import os
    engine = create_engine(os.getenv('DATABASE_URL'))
    engine.connect()
    return engine
