import mysql.connector

# Database connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="95412313@Sara",
    database="hydro_db"
)
cursor = db.cursor()

# Create location_sensor_limits table
create_table_sql = """
CREATE TABLE IF NOT EXISTS location_sensor_limits (
    location_id INT,
    sensor_id INT,
    lower_limit FLOAT,
    upper_limit FLOAT,
    PRIMARY KEY (location_id, sensor_id),
    FOREIGN KEY (location_id) REFERENCES locations(location_id),
    FOREIGN KEY (sensor_id) REFERENCES sensors(sensor_id)
)
"""
cursor.execute(create_table_sql)

# Location-specific sensor limits data
location_limits = [
    # Blue River
    (1, 2, 0, 566),    # Flow
    (1, 1, 0, 7.6),    # Depth
    (1, 3, 0, 250),    # Rainfall
    
    # Brush Creek Rockhill
    (2, 2, 0, 200),    # Flow
    (2, 1, 0, 5.0),    # Depth
    (2, 3, 0, 250),    # Rainfall
    
    # Brush Creek Ward
    (3, 2, 0, 200),    # Flow
    (3, 1, 0, 5.0),    # Depth
    (3, 3, 0, 250),    # Rainfall
    
    # Missouri River
    (4, 2, 0, 2800),   # Flow
    (4, 1, 0, 10.0),   # Depth
    (4, 3, 0, 250),    # Rainfall
    
    # Weather Station
    (5, 3, 0, 250)     # Rainfall only
]

# Insert location-specific limits
insert_sql = """
INSERT INTO location_sensor_limits (location_id, sensor_id, lower_limit, upper_limit)
VALUES (%s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    lower_limit = VALUES(lower_limit),
    upper_limit = VALUES(upper_limit)
"""

for location_id, sensor_id, lower_limit, upper_limit in location_limits:
    try:
        cursor.execute(insert_sql, (location_id, sensor_id, lower_limit, upper_limit))
        print(f"Updated limits for location_id {location_id}, sensor_id {sensor_id}: {lower_limit} to {upper_limit}")
    except Exception as e:
        print(f"Error updating limits: {str(e)}")
        db.rollback()

db.commit()
cursor.close()
db.close() 