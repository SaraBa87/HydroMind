import mysql.connector

# Database connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="95412313@Sara",
    database="hydro_db"
)
cursor = db.cursor()

# Define new sensors
new_sensors = [
    ('flow_cfs', 0, 100000, 'cfs'),
    ('flow_m3s', 0, 2831.68, 'm³/s'),
    ('depth_ft', 0, 32.8084, 'ft'),
    ('depth_m', 0, 10, 'm'),
    ('precipitation_mm', 0, 500, 'mm')
]

# Add new sensors
for sensor_type, lower_limit, upper_limit, unit in new_sensors:
    try:
        sql = """INSERT INTO sensors 
                (sensor_type, lower_limit, upper_limit, unit) 
                VALUES (%s, %s, %s, %s)"""
        values = (sensor_type, lower_limit, upper_limit, unit)
        cursor.execute(sql, values)
        print(f"Added sensor: {sensor_type}")
    except mysql.connector.Error as err:
        if err.errno == 1062:  # Duplicate entry error
            print(f"Sensor {sensor_type} already exists")
        else:
            print(f"Error adding sensor {sensor_type}: {err}")

db.commit()
cursor.close()
db.close() 