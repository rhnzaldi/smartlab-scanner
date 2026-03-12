import pymysql
import os
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
DB_PORT = int(os.environ.get("DB_PORT", "3306"))
DB_USER = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_NAME = os.environ.get("DB_NAME", "smartlab_db")

def migrate():
    conn = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )
    
    try:
        with conn.cursor() as cursor:
            # Add status column to jadwal table
            cursor.execute("ALTER TABLE jadwal ADD COLUMN status VARCHAR(20) DEFAULT 'tersedia'")
            print("Successfully added 'status' column to 'jadwal' table.")
        conn.commit()
    except Exception as e:
        print(f"Error or column already exists: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
