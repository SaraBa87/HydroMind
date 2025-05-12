import mysql.connector
import pandas as pd
from datetime import datetime
from weather_data import fetch_weather_data
from station_data import fetch_station_data, STATIONS

# Database connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="95412313@Sara",
    database="hydro_db"
)
cursor = db.cursor()

# Location mapping (using existing location_ids)
location_mapping = {
    '06893578': 11,  # Blue River
    '06893562': 12,  # Brush Creek Rockhill
    '06893557': 13,  # Brush Creek Ward
    '06893000': 14,  # Missouri River
    'weather': 15    # Weather Station
}

# Sensor mapping (using new sensor_ids)
sensor_mapping = {
    'flow': {
        11: '11_1',  # Blue River flow
        12: '12_1',  # Brush Creek Rockhill flow
        13: '13_1',  # Brush Creek Ward flow
        14: '14_1'   # Missouri River flow
    },
    'depth': {
        11: '11_2',  # Blue River depth
        12: '12_2',  # Brush Creek Rockhill depth
        13: '13_2',  # Brush Creek Ward depth
        14: '14_2'   # Missouri River depth
    },
    'rain': {
        11: '11_3',  # Blue River rain
        12: '12_3',  # Brush Creek Rockhill rain
        13: '13_3',  # Brush Creek Ward rain
        14: '14_3',  # Missouri River rain
        15: '15_3'   # Weather Station rain
    }
}

def cleanup_sensor_data():
    print("Cleaning up sensor_data table...")
    try:
        cursor.execute("TRUNCATE TABLE sensor_data")
        db.commit()
        print("Successfully cleaned up sensor_data table")
    except Exception as e:
        print(f"Error cleaning up sensor_data table: {str(e)}")
        db.rollback()

def load_flow_data():
    print("Loading flow data...")
    start_date = datetime(2025, 5, 2)
    end_date = datetime(2025, 5, 9)
    
    for station_id in STATIONS.keys():
        try:
            df = fetch_station_data(station_id, start_date, end_date, '00060')
            if df is not None:
                location_id = location_mapping[station_id]
                sensor_id = sensor_mapping['flow'][location_id]
                
                # Insert flow data into sensor_data table (only m³/s)
                for timestamp, row in df.iterrows():
                    sql = """INSERT INTO sensor_data 
                            (location_id, sensor_id, value, timestamp) 
                            VALUES (%s, %s, %s, %s)"""
                    values = (
                        location_id,
                        sensor_id,
                        row['flow_m3s'],
                        timestamp
                    )
                    cursor.execute(sql, values)
                
                db.commit()
                print(f"Successfully loaded flow data for {STATIONS[station_id]}")
        except Exception as e:
            print(f"Error loading flow data for {STATIONS[station_id]}: {str(e)}")
            db.rollback()

def load_depth_data():
    print("\nLoading depth data...")
    start_date = datetime(2025, 5, 2)
    end_date = datetime(2025, 5, 9)
    
    for station_id in STATIONS.keys():
        try:
            df = fetch_station_data(station_id, start_date, end_date, '00065')
            if df is not None:
                location_id = location_mapping[station_id]
                sensor_id = sensor_mapping['depth'][location_id]
                
                # Insert depth data into sensor_data table (only m)
                for timestamp, row in df.iterrows():
                    sql = """INSERT INTO sensor_data 
                            (location_id, sensor_id, value, timestamp) 
                            VALUES (%s, %s, %s, %s)"""
                    values = (
                        location_id,
                        sensor_id,
                        row['gage_height_m'],
                        timestamp
                    )
                    cursor.execute(sql, values)
                
                db.commit()
                print(f"Successfully loaded depth data for {STATIONS[station_id]}")
        except Exception as e:
            print(f"Error loading depth data for {STATIONS[station_id]}: {str(e)}")
            db.rollback()

def load_weather_data():
    print("\nLoading weather data...")
    try:
        df = fetch_weather_data()
        location_id = location_mapping['weather']
        sensor_id = sensor_mapping['rain'][location_id]
        
        # Insert weather data into sensor_data table
        for timestamp, row in df.iterrows():
            # Format timestamp as string
            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            sql = """INSERT INTO sensor_data 
                    (location_id, sensor_id, value, timestamp) 
                    VALUES (%s, %s, %s, %s)"""
            values = (
                location_id,
                sensor_id,
                float(row['prcp']),  # Ensure value is float
                timestamp_str
            )
            cursor.execute(sql, values)
        
        db.commit()
        print("Successfully loaded weather data")
    except Exception as e:
        print(f"Error loading weather data: {str(e)}")
        db.rollback()

# Execute the loading functions
try:
    cleanup_sensor_data()
    load_flow_data()
    load_depth_data()
    load_weather_data()
    print("\nAll data has been loaded successfully!")
except Exception as e:
    print(f"An error occurred: {str(e)}")
finally:
    cursor.close()
    db.close() 