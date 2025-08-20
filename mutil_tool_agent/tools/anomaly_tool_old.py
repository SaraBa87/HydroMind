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
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# from mutil_tool_agent.config.db_config import DB_CONFIG

class AnomalyTools:
    def __init__(self, db_connection: mysql.connector.MySQLConnection, 
                 flatline_duration_hours: float = 5.0,  # Duration in hours
                 gap_duration_hours: float = 1.0,       # Duration in hours
                 change_threshold: float = 3.0,         # Standard deviations for sudden changes
                 spike_duration_hours: float = 1.0,     # Duration for spike detection
                 blockage_duration_hours: float = 3.0): # Duration for blockage detection
        self.db = db_connection
        self.cursor = self.db.cursor(dictionary=True)
        self.sensor_limits = self._fetch_sensor_limits()
        # Convert hours to number of 15-minute readings
        self.flatline_threshold = int(flatline_duration_hours * 4)  # 4 readings per hour
        self.gap_threshold = int(gap_duration_hours * 4)            # 4 readings per hour
        self.spike_threshold = int(spike_duration_hours * 4)        # 4 readings per hour
        self.blockage_threshold = int(blockage_duration_hours * 4)  # 4 readings per hour
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
                
                # 2. Check for flat lines with duration
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
                
                # 3. Check for gaps with duration
                time_diffs = sensor_data['timestamp'].diff()
                expected_interval = timedelta(minutes=15)  # Assuming 15-minute intervals
                gaps = time_diffs > (expected_interval * self.gap_threshold)
                if gaps.any():
                    gap_anomalies = sensor_data[gaps].copy()
                    gap_anomalies['gap_duration_hours'] = time_diffs[gaps].dt.total_seconds() / 3600
                    anomalies[f"{sensor_id}_gaps"] = gap_anomalies
                
                # 4. Check for sudden changes with duration
                if len(sensor_data) > 1:
                    # --- NEW: Calculate normal range from dry weather periods ---
                    try:
                        rain_query = "SELECT timestamp, value FROM sensor_data WHERE sensor_id = '15_3'"
                        self.cursor.execute(rain_query)
                        rain_data = pd.DataFrame(self.cursor.fetchall())
                        rain_data['timestamp'] = pd.to_datetime(rain_data['timestamp'])
                        # Identify dry weather periods (no rain for at least 24 hours)
                        rain_data['is_rain'] = rain_data['value'] > 0.01  # Dry weather threshold
                        rain_data['dry_period'] = (~rain_data['is_rain']).astype(int)
                        rain_data['dry_period_group'] = (rain_data['dry_period'] != rain_data['dry_period'].shift()).cumsum()
                        dry_periods = rain_data.groupby('dry_period_group').agg({
                            'timestamp': ['min', 'max'],
                            'dry_period': 'first'
                        })
                        dry_periods.columns = ['start_time', 'end_time', 'is_dry']
                        dry_periods['duration'] = (dry_periods['end_time'] - dry_periods['start_time']).dt.total_seconds() / 3600
                        long_dry_periods = dry_periods[(dry_periods['is_dry'] == 1) & (dry_periods['duration'] >= 24)]
                        dry_weather_data = pd.DataFrame()
                        for _, period in long_dry_periods.iterrows():
                            period_data = sensor_data[(sensor_data['timestamp'] >= period['start_time']) & (sensor_data['timestamp'] <= period['end_time'])]
                            dry_weather_data = pd.concat([dry_weather_data, period_data])
                        if not dry_weather_data.empty:
                            normal_mean = dry_weather_data['value'].mean()
                            normal_std = dry_weather_data['value'].std()
                            normal_lower = normal_mean - 2 * normal_std
                            normal_upper = normal_mean + 2 * normal_std
                        else:
                            # Fallback to sensor limits if no dry weather data available
                            sensor_limits = self.sensor_limits.get(sensor_id, {})
                            normal_lower = sensor_limits.get('lower', float('-inf'))
                            normal_upper = sensor_limits.get('upper', float('inf'))
                    except Exception as e:
                        # Fallback to sensor limits
                        sensor_limits = self.sensor_limits.get(sensor_id, {})
                        normal_lower = sensor_limits.get('lower', float('-inf'))
                        normal_upper = sensor_limits.get('upper', float('inf'))
                    # Calculate rate of change
                    sensor_data['rate_of_change'] = sensor_data['value'].diff()
                    mean_change = sensor_data['rate_of_change'].mean()
                    std_change = sensor_data['rate_of_change'].std()
                    min_significant_change = 0.1  # Minimum change to be considered significant
                    # Only report the first out-of-normal event (spike or negative spike) in the entire dataset, ignore all others
                    spike_indices = []
                    negative_spike_indices = []
                    found_anomaly = False
                    anomaly_type = None
                    for idx, row in sensor_data.iterrows():
                        if found_anomaly:
                            break
                        val = row['value']
                        roc = row['rate_of_change']
                        if val > normal_upper and roc > min_significant_change:
                            spike_indices.append(idx)
                            found_anomaly = True
                            anomaly_type = 'spike'
                        elif val < normal_lower and roc < -min_significant_change:
                            negative_spike_indices.append(idx)
                            found_anomaly = True
                            anomaly_type = 'negative_spike'
                    if anomaly_type == 'spike' and spike_indices:
                        spike = sensor_data.loc[spike_indices].copy()
                        spike['duration_hours'] = self.spike_threshold / 4
                        anomalies[f"{sensor_id}_spike"] = spike
                    elif anomaly_type == 'negative_spike' and negative_spike_indices:
                        negative_spike = sensor_data.loc[negative_spike_indices].copy()
                        negative_spike['duration_hours'] = self.spike_threshold / 4
                        anomalies[f"{sensor_id}_negative_spike"] = negative_spike

                    # 5. Special check for depth sensor blockage with duration
                    if sensor_type == 'Depth':
                        # Blockage is defined as a spike followed by a flat line lasting more than blockage_threshold
                        if 'spike' in anomalies or 'negative_spike' in anomalies:
                            spikes = anomalies[f"{sensor_id}_spike"] if 'spike' in anomalies else anomalies[f"{sensor_id}_negative_spike"]
                            
                            # For each spike, check if it's followed by a flat line
                            blockage_candidates = []
                            for idx in spikes.index:
                                # Get the next blockage_threshold hours of data
                                next_hours = sensor_data.loc[idx:].head(self.blockage_threshold)
                                
                                # Check if the line is flat (all value_diffs are close to 0)
                                if len(next_hours) >= self.blockage_threshold and all(abs(diff) < 0.01 for diff in next_hours['value_diff']):
                                    blockage_candidates.append(idx)
                            
                            blockage_candidates = sensor_data.loc[blockage_candidates]
                            if not blockage_candidates.empty:
                                blockage_candidates['duration_hours'] = self.blockage_threshold / 4  # Convert readings to hours
                                anomalies[f"{sensor_id}_potential_blockage"] = blockage_candidates

            elif sensor_type == 'Rain Gauge':
                # Check for unrealistic rainfall with duration
                unrealistic_rain = sensor_data[sensor_data['value'] > 250]  # More than 250mm/day
                if not unrealistic_rain.empty:
                    unrealistic_rain['duration_hours'] = 1.0  # Assuming 1-hour duration for unrealistic rain
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
            # Define rainfall events with duration
            rain_data['timestamp'] = pd.to_datetime(rain_data['timestamp'])
            rain_data['is_rain'] = rain_data['value'] > 0.01  # Dry weather threshold
            rain_data['event'] = (rain_data['is_rain'] != rain_data['is_rain'].shift()).cumsum()
            rain_events = rain_data.groupby('event').agg({
                'timestamp': ['min', 'max'],
                'value': 'sum'
            }).reset_index()
            rain_events.columns = ['event', 'start_time', 'end_time', 'total_rain']
            rain_events['duration'] = (rain_events['end_time'] - rain_events['start_time']).dt.total_seconds() / 3600
            rain_events = rain_events[(rain_events['total_rain'] >= 0.25) & (rain_events['duration'] >= 2)]

            # Check for missing sensor response with duration
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
                            missing_response.append({
                                'start_time': event['start_time'],
                                'end_time': event['end_time'],
                                'duration': event['duration'],
                                'total_rain': event['total_rain']
                            })
                    if missing_response:
                        missing_response_df = pd.DataFrame(missing_response)
                        anomalies[f"{sensor_id}_missing_response"] = missing_response_df

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
            "by_type": {}
        }
        
        # Count anomalies by type (global only)
        for anomaly_type, type_anomalies in anomalies.items():
            type_count = 0
            for sensor_anomaly_key, df in type_anomalies.items():
                # Remove sensor ID and any number prefix from anomaly key for summary
                type_label = sensor_anomaly_key
                if '_' in type_label:
                    parts = type_label.split('_')
                    if len(parts) > 2:
                        type_label = '_'.join(parts[2:])
                    else:
                        type_label = parts[1]
                # By type (global)
                if type_label not in summary["by_type"]:
                    summary["by_type"][type_label] = 0
                summary["by_type"][type_label] += len(df)
                type_count += len(df)
            summary["total_anomalies"] += type_count
        return summary

# # Example usage
# if __name__ == "__main__":
#     try:
#         db_connection = mysql.connector.connect(**DB_CONFIG)
#         anomaly_tool = AnomalyTools(db_connection=db_connection)
#         query = "SELECT * FROM sensor_data WHERE sensor_id = '13_2'"
#         print(f"Executing query: {query}")
#         anomalies = anomaly_tool.detect_anomalies(query)

#         # Print the full anomalies output for debugging
#         print("\nFull anomalies output:")
#         print(anomalies)

#         # Only try to print the summary if it exists
#         if 'summary' in anomalies:
#             print("\nAnomaly Detection Summary:")
#             print(f"Total Anomalies: {anomalies['summary']['total_anomalies']}")
#             print("\nBy Type:")
#             for type_name, count in anomalies['summary']['by_type'].items():
#                 print(f"  {type_name}: {count}")
#         else:
#             print("\nNo summary found in anomalies output.")
        
#         db_connection.close()
#     except Exception as e:
#         print(f"Error occurred: {str(e)}")
#         import traceback
#         traceback.print_exc()
