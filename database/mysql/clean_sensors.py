import mysql.connector

# Database connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="95412313@Sara",
    database="hydro_db"
)
cursor = db.cursor()

# Delete duplicate sensors
try:
    cursor.execute("DELETE FROM sensors WHERE sensor_id IN (6, 7, 8, 9, 10)")
    db.commit()
    print("Removed duplicate sensors")
except Exception as e:
    print(f"Error removing sensors: {str(e)}")
    db.rollback()

cursor.close()
db.close() 