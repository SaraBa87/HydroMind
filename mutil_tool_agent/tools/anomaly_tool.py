from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy import stats
import mysql.connector
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mutil_tool_agent.config.db_config import DB_CONFIG

class AnomalyTools:
    def __init__(self, db_connection: mysql.connector.MySQLConnection, 
                 flatline_duration_hours: float = 5.0,  # Duration in hours
                 gap_duration_hours: float = 1.0,       # Duration in hours
                 change_threshold: float = 3.0):        # Standard deviations for sudden changes
        self.db = db_connection
        self.cursor = self.db.cursor(dictionary=True)
        self.sensor_limits = self._fetch_sensor_limits()
        # Convert hours to number of 15-minute readings
        self.flatline_threshold = int(flatline_duration_hours * 4)  # 4 readings per hour
        self.gap_threshold = int(gap_duration_hours * 4)            # 4 readings per hour
        self.change_threshold = change_threshold

    def _fetch_sensor_limits(self) -> Dict[str, Dict[str, float]]:
        """Fetch sensor limits from the database."""
        try:
            self.cursor.execute("SELECT * FROM sensors")
            sensor_limits = self.cursor.fetchall()
            limits = {}
            for limit in sensor_limits:
                sensor_id = limit['sensor_id']
                limits[sensor_id] = {
                    'sensor_type': limit['sensor_type'],
                    'unit': limit['unit'],
                    'lower': float(limit['lower_limit']),
                    'upper': float(limit['upper_limit'])
                }
            return limits
        except mysql.connector.Error as err:
            return {}

    def detect_anomalies(self, query: str) -> Dict[str, Any]:
        """Detect anomalies in sensor data using the provided SQL query."""
        try:
            # Get data from database
            self.cursor.execute(query)
            data = pd.DataFrame(self.cursor.fetchall())
            
            if data.empty:
                return {"error": "No data found for the given query"}

            # Detect different types of anomalies
            anomalies = {
                "sensor_limits": self._detect_sensor_limits_anomalies(data),
                "sensor_type": self._detect_sensor_type_anomalies(data),
                "correlated": self._detect_correlated_anomalies(data)
            }

            # Convert DataFrames to dictionaries
            serializable_anomalies = {}
            for anomaly_type, type_anomalies in anomalies.items():
                serializable_anomalies[anomaly_type] = {}
                for sensor_id, df in type_anomalies.items():
                    # Convert timestamp columns to strings
                    df_copy = df.copy()
                    for col in df_copy.columns:
                        if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                            df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                    serializable_anomalies[anomaly_type][sensor_id] = df_copy.to_dict(orient='records')

            # Generate summary
            summary = self._generate_anomaly_summary(anomalies)
            return {
                "anomalies": serializable_anomalies,
                "summary": summary
            }

        except Exception as e:
            return {"error": str(e)}

    def _detect_sensor_limits_anomalies(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Detect anomalies based on sensor limits."""
        anomalies = {}
        for sensor_id in data['sensor_id'].unique():
            sensor_type = self._get_sensor_type(sensor_id)
            if sensor_type in self.sensor_limits:
                limits = self.sensor_limits[sensor_id]
                sensor_data = data[data['sensor_id'] == sensor_id]
                
                # Check for values outside limits
                out_of_bounds = sensor_data[
                    (sensor_data['value'] < limits['lower']) | 
                    (sensor_data['value'] > limits['upper'])
                ]
                
                if not out_of_bounds.empty:
                    anomalies[sensor_id] = out_of_bounds
        return anomalies

    def _detect_sensor_type_anomalies(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Detect anomalies specific to each sensor type."""
        anomalies = {}
        
        # Get rain data for correlation checks
        try:
            rain_query = "SELECT timestamp, value FROM sensor_data WHERE sensor_id = '15_3'"
            self.cursor.execute(rain_query)
            rain_data = pd.DataFrame(self.cursor.fetchall())
            has_rain_data = not rain_data.empty
        except Exception as e:
            rain_data = pd.DataFrame()
            has_rain_data = False
        
        for sensor_id in data['sensor_id'].unique():
            sensor_type = self._get_sensor_type(sensor_id)
            sensor_data = data[data['sensor_id'] == sensor_id].copy()
            
            if sensor_type in ['Flow', 'Depth']:
                # 1. Check for negative values
                negative_values = sensor_data[sensor_data['value'] < 0]
                if not negative_values.empty:
                    anomalies[f"{sensor_id}_negative"] = negative_values
                
                # 2. Check for flat lines
                sensor_data['timestamp'] = pd.to_datetime(sensor_data['timestamp'])
                sensor_data['value_diff'] = sensor_data['value'].diff().abs()
                flat_line = sensor_data['value_diff'] == 0
                flat_line_groups = (flat_line != flat_line.shift()).cumsum()
                flat_line_lengths = flat_line_groups.groupby(flat_line_groups).transform('count')
                flat_line_anomalies = sensor_data[flat_line & (flat_line_lengths >= self.flatline_threshold)]
                if not flat_line_anomalies.empty:
                    # Add duration information
                    flat_line_anomalies['duration_hours'] = flat_line_anomalies.groupby(flat_line_groups)['timestamp'].transform(
                        lambda x: (x.max() - x.min()).total_seconds() / 3600
                    )
                    anomalies[f"{sensor_id}_flat_line"] = flat_line_anomalies
                
                # 3. Check for gaps
                time_diffs = sensor_data['timestamp'].diff()
                expected_interval = timedelta(minutes=15)  # Assuming 15-minute intervals
                gaps = time_diffs > (expected_interval * self.gap_threshold)
                if gaps.any():
                    gap_anomalies = sensor_data[gaps].copy()
                    gap_anomalies['gap_duration_hours'] = time_diffs[gaps].dt.total_seconds() / 3600
                    anomalies[f"{sensor_id}_gaps"] = gap_anomalies
                
                # 4. Check for sudden changes
                if len(sensor_data) > 1:
                    # Calculate rate of change
                    sensor_data['rate_of_change'] = sensor_data['value'].diff()
                    mean_change = sensor_data['rate_of_change'].mean()
                    std_change = sensor_data['rate_of_change'].std()
                    
                    # Detect sudden rises and drops with absolute threshold
                    min_significant_change = 0.1  # Minimum change to be considered significant
                    sudden_rise = sensor_data[
                        (sensor_data['rate_of_change'] > (mean_change + self.change_threshold * std_change)) &
                        (sensor_data['rate_of_change'] > min_significant_change)
                    ]
                    sudden_drop = sensor_data[
                        (sensor_data['rate_of_change'] < (mean_change - self.change_threshold * std_change)) &
                        (sensor_data['rate_of_change'] < -min_significant_change)
                    ]
                    
                    # Filter out changes that correlate with rain events
                    if has_rain_data:
                        def is_rain_correlated(timestamp, window=timedelta(hours=2)):
                            rain_window = rain_data[
                                (rain_data['timestamp'] >= timestamp - window) & 
                                (rain_data['timestamp'] <= timestamp + window)  # Check both before and after
                            ]
                            return rain_window['value'].max() > 0.1  # Significant rain threshold
                        
                        # Filter rises
                        valid_rises = sudden_rise[~sudden_rise['timestamp'].apply(is_rain_correlated)]
                        if not valid_rises.empty:
                            anomalies[f"{sensor_id}_sudden_rise"] = valid_rises
                        
                        # Filter drops
                        valid_drops = sudden_drop[~sudden_drop['timestamp'].apply(is_rain_correlated)]
                        if not valid_drops.empty:
                            anomalies[f"{sensor_id}_sudden_drop"] = valid_drops
                    else:
                        if not sudden_rise.empty:
                            anomalies[f"{sensor_id}_sudden_rise"] = sudden_rise
                        if not sudden_drop.empty:
                            anomalies[f"{sensor_id}_sudden_drop"] = sudden_drop
                
                # 5. Special check for depth sensor blockage
                if sensor_type == 'Depth':
                    # Blockage is defined as a sudden rise followed by a flat line lasting more than 3 hours
                    # Use the sudden_rise detection we already have
                    if 'sudden_rise' in anomalies:
                        sudden_rises = anomalies['sudden_rise']
                        
                        # For each sudden rise, check if it's followed by a flat line
                        blockage_candidates = []
                        for idx in sudden_rises.index:
                            # Get the next 3 hours of data
                            next_3_hours = sensor_data.loc[idx:].head(12)  # Assuming 15-min intervals
                            
                            # Check if the line is flat (all value_diffs are close to 0)
                            if len(next_3_hours) >= 12 and all(abs(diff) < 0.01 for diff in next_3_hours['value_diff']):
                                blockage_candidates.append(idx)
                        
                        blockage_candidates = sensor_data.loc[blockage_candidates]
                        if not blockage_candidates.empty:
                            anomalies[f"{sensor_id}_potential_blockage"] = blockage_candidates

            elif sensor_type == 'Rain Gauge':
                # Check for unrealistic rainfall
                unrealistic_rain = sensor_data[sensor_data['value'] > 250]  # More than 250mm/day
                if not unrealistic_rain.empty:
                    anomalies[f"{sensor_id}_unrealistic_rain"] = unrealistic_rain

        return anomalies

    def _detect_correlated_anomalies(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Detect anomalies by correlating different sensor types."""
        anomalies = {}

        # Get data for each sensor type
        flow_data = data[data['sensor_id'].str.contains('flow', case=False)]
        depth_data = data[data['sensor_id'].str.contains('depth', case=False)]
        rain_data = data[data['sensor_id'].str.contains('rain', case=False)]
        
        if not rain_data.empty:
            # Define rainfall events
            rain_data['timestamp'] = pd.to_datetime(rain_data['timestamp'])
            rain_data['is_rain'] = rain_data['value'] > 0.01  # Dry weather threshold
            rain_data['event'] = (rain_data['is_rain'] != rain_data['is_rain'].shift()).cumsum()
            rain_events = rain_data.groupby('event').agg({
                'timestamp': ['min', 'max'],
                'value': 'sum'
            }).reset_index()
            rain_events.columns = ['event', 'start_time', 'end_time', 'total_rain']
            rain_events['duration'] = (rain_events['end_time'] - rain_events['start_time']).dt.total_seconds() / 3600
            rain_events = rain_events[(rain_events['total_rain'] >= 0.25) & (rain_events['duration'] >= 6)]

            # Check for missing sensor response
            for sensor_id in data['sensor_id'].unique():
                sensor_type = self._get_sensor_type(sensor_id)
                if sensor_type in ['Flow', 'Depth']:
                    sensor_data = data[data['sensor_id'] == sensor_id].copy()
                    sensor_data['timestamp'] = pd.to_datetime(sensor_data['timestamp'])
                    missing_response = []
                    for _, event in rain_events.iterrows():
                        window_start = event['start_time']
                        window_end = event['start_time'] + timedelta(hours=4)
                        sensor_window = sensor_data[(sensor_data['timestamp'] >= window_start) & (sensor_data['timestamp'] <= window_end)]
                        if sensor_window.empty or sensor_window['value'].max() - sensor_window['value'].min() < 0.1:
                            missing_response.append(event)
                    if missing_response:
                        anomalies[f"{sensor_id}_missing_response"] = pd.DataFrame(missing_response)

        return anomalies

    def _get_sensor_type(self, sensor_id: str) -> str:
        """Get sensor type from sensor_id."""
        try:
            self.cursor.execute("SELECT sensor_type FROM sensors WHERE sensor_id = %s", (sensor_id,))
            result = self.cursor.fetchone()
            return result['sensor_type'] if result else 'Unknown'
        except mysql.connector.Error:
            return 'Unknown'

    def _generate_anomaly_summary(self, anomalies: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """Generate a summary of detected anomalies."""
        summary = {
            "total_anomalies": 0,
            "by_type": {},
            "by_sensor": {}
        }
        
        # Count anomalies by type
        for anomaly_type, type_anomalies in anomalies.items():
            type_count = sum(len(df) for df in type_anomalies.values())
            summary["by_type"][anomaly_type] = type_count
            summary["total_anomalies"] += type_count
            
            # Count by sensor
            for sensor_id, df in type_anomalies.items():
                if sensor_id not in summary["by_sensor"]:
                    summary["by_sensor"][sensor_id] = 0
                summary["by_sensor"][sensor_id] += len(df)

        return summary

# Example usage
if __name__ == "__main__":
    try:
        db_connection = mysql.connector.connect(**DB_CONFIG)
        anomaly_tool = AnomalyTools(db_connection=db_connection)
        anomalies = anomaly_tool.detect_anomalies("SELECT * FROM sensor_data WHERE sensor_id = '13_1'")
        print("\nAnomaly Detection Summary:")
        print(f"Total Anomalies: {anomalies['summary']['total_anomalies']}")
        print("\nBy Type:")
        for type_name, count in anomalies['summary']['by_type'].items():
            print(f"  {type_name}: {count}")
        print("\nBy Sensor:")
        for sensor_id, count in anomalies['summary']['by_sensor'].items():
            print(f"  {sensor_id}: {count}")
        db_connection.close()
    except Exception as e:
        pass
