import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy import stats
from typing import Dict, Any, List, Optional
import mysql.connector
from datetime import datetime
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mutil_tool_agent.config.db_config import DB_CONFIG
from google.adk.tools import BaseTool
    
class AnomalyDetectionTool(BaseTool):
    def __init__(self, db_connection: mysql.connector.MySQLConnection, contamination: float = 0.01):
        super().__init__(
            name="anomaly_detection_tool",
            description="Detect anomalies in hydrological sensor data using a provided SQL query."
        )
        self.db = db_connection
        self.cursor = self.db.cursor(dictionary=True)
        self.contamination = contamination
        self.sensor_limits = self._fetch_sensor_limits()

    def _fetch_sensor_limits(self) -> Dict[str, Dict[str, float]]:
        try:
            self.cursor.execute("SELECT * FROM sensors")
            sensor_limits = self.cursor.fetchall()
            limits = {}
            for limit in sensor_limits:
                sensor_type = limit['sensor_type']
                limits[sensor_type] = {
                    'sensor_id': limit['sensor_id'],
                    'unit': limit['unit'],
                    'lower': float(limit['lower_limit']),
                    'upper': float(limit['upper_limit'])
                }
            return limits
        except mysql.connector.Error as err:
            print(f"Error fetching sensor limits: {err}")
            return {
                'Flow': {'sensor_id': None, 'unit': 'm3/s', 'lower': 0, 'upper': 500},
                'Depth': {'sensor_id': None, 'unit': 'm', 'lower': 0, 'upper': 10},
                'Rain Gauge': {'sensor_id': None, 'unit': 'mm/day', 'lower': 0, 'upper': 250}
            }

    def get_data(self, query: str) -> pd.DataFrame:
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        return pd.DataFrame(results)

    def _get_sensor_type(self, sensor_id: str) -> str:
        """
        Get sensor type from sensor_id.
        """
        try:
            self.cursor.execute("SELECT sensor_type FROM sensors WHERE sensor_id = %s", (sensor_id,))
            result = self.cursor.fetchone()
            return result['sensor_type'] if result else 'Unknown'
        except mysql.connector.Error:
            return 'Unknown'

    def _prepare_sensor_data(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for each sensor type by pivoting the data.
        """
        sensor_data = {}
        for sensor_id in data['sensor_id'].unique():
            sensor_type = self._get_sensor_type(sensor_id)
            if sensor_type != 'Unknown':
                sensor_values = data[data['sensor_id'] == sensor_id].set_index('timestamp')['value']
                sensor_data[sensor_type] = sensor_values
        return sensor_data

    def detect_sensor_type_anomalies(self, data: pd.DataFrame) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Detect anomalies for all sensor types (Flow, Depth, Rain).
        
        Args:
            data: DataFrame containing sensor data
            
        Returns:
            Dictionary containing anomalies for each sensor type
        """
        anomalies = {}
        sensor_data = self._prepare_sensor_data(data)
        
        # Define sensor-specific thresholds and checks
        sensor_configs = {
            'Flow': {
                'reversal_threshold': 0,  # Flow should not be negative
                'window_size': 15,        # Window for consistency checks
                'prolonged_window': None, # Not used for flow
                'min_value': 0,           # Minimum valid value
                'max_value': float('inf'), # Maximum valid value
                'gap_threshold': 4,       # Number of missing points to consider a gap
                'normal_fluctuation': 0.1  # 10% of mean value for normal fluctuations
            },
            'Depth': {
                'reversal_threshold': None,  # Not used for depth
                'window_size': 15,           # Window for consistency checks
                'prolonged_window': None,    # Not used for depth
                'min_value': 0,              # Minimum valid value
                'max_value': float('inf'),   # Maximum valid value
                'gap_threshold': 4,          # Number of missing points to consider a gap
                'normal_fluctuation': 0.05   # 5% of mean value for normal fluctuations
            },
            'Rain Gauge': {
                'reversal_threshold': None,  # Not used for rain
                'window_size': 15,           # Window for consistency checks
                'prolonged_window': 60,      # Window for prolonged rainfall
                'min_value': 0,              # Minimum valid value
                'max_value': float('inf'),   # Maximum valid value
                'gap_threshold': 4,          # Number of missing points to consider a gap
                'normal_fluctuation': 0.2    # 20% of mean value for normal fluctuations
            }
        }
        
        for sensor_type, values in sensor_data.items():
            if sensor_type not in sensor_configs:
                continue
                
            config = sensor_configs[sensor_type]
            sensor_anomalies = {}
            
            # 1. Check for flat lines (constant values)
            constant_std = values.rolling(window=config['window_size']).std() == 0
            constant_range = values.rolling(window=config['window_size']).max() - values.rolling(window=config['window_size']).min() == 0
            flat_line = constant_std | constant_range
            sensor_anomalies[f'{sensor_type.lower()}_flat_line'] = pd.DataFrame({
                'timestamp': values[flat_line].index,
                'value': values[flat_line].values
            })
            
            # 2. Check for negative values
            if config['min_value'] is not None:
                negative_values = values < config['min_value']
                sensor_anomalies[f'{sensor_type.lower()}_negative'] = pd.DataFrame({
                    'timestamp': values[negative_values].index,
                    'value': values[negative_values].values
                })
            
            # 3. Check for gaps in data
            time_diffs = values.index.to_series().diff()
            expected_readings = time_diffs.dt.total_seconds() / (15 * 60)
            gaps = expected_readings > config['gap_threshold']
            if gaps.any():
                sensor_anomalies[f'{sensor_type.lower()}_gaps'] = pd.DataFrame({
                    'timestamp': values.index[gaps],
                    'gap_size': expected_readings[gaps]
                })
            
            # 4. Check for missing data
            missing_values = values.isna()
            if missing_values.any():
                sensor_anomalies[f'{sensor_type.lower()}_missing'] = pd.DataFrame({
                    'timestamp': values.index[missing_values],
                    'value': values[missing_values]
                })
            
            # 5. Sudden spikes with rain event check
            if sensor_type in ['Flow', 'Depth'] and 'Rain Gauge' in sensor_data:
                rain_data = sensor_data['Rain Gauge']
                # Define spike threshold as 95th percentile
                spike_threshold = values.quantile(0.95)
                potential_spikes = values > spike_threshold
                
                # Check for rain events within 2 hours before each spike
                rain_window = pd.Timedelta(hours=2)
                valid_spikes = []
                
                for timestamp in values[potential_spikes].index:
                    # Check if there was rain in the window before the spike
                    rain_window_data = rain_data[timestamp - rain_window:timestamp]
                    if rain_window_data.empty or rain_window_data.max() < 0.1:  # No significant rain
                        valid_spikes.append(timestamp)
                
                if valid_spikes:
                    sensor_anomalies[f'{sensor_type.lower()}_spikes'] = pd.DataFrame({
                        'timestamp': valid_spikes,
                        'value': values[valid_spikes].values
                    })
            
            # 6. Sudden drops with rain event check
            if sensor_type in ['Flow', 'Depth'] and 'Rain Gauge' in sensor_data:
                rain_data = sensor_data['Rain Gauge']
                # Define drop threshold as 5th percentile
                drop_threshold = values.quantile(0.05)
                potential_drops = values < drop_threshold
                
                # Check for rain events and spikes before each drop
                rain_window = pd.Timedelta(hours=4)  # Longer window to catch preceding events
                valid_drops = []
                
                for timestamp in values[potential_drops].index:
                    # Check if there was rain and a spike in the window before the drop
                    window_data = values[timestamp - rain_window:timestamp]
                    rain_window_data = rain_data[timestamp - rain_window:timestamp]
                    
                    if not window_data.empty and not rain_window_data.empty:
                        had_rain = rain_window_data.max() > 0.1
                        had_spike = window_data.max() > values.quantile(0.95)
                        
                        # Only report as anomaly if there was no rain or no spike
                        if not (had_rain and had_spike):
                            valid_drops.append(timestamp)
                
                if valid_drops:
                    sensor_anomalies[f'{sensor_type.lower()}_drops'] = pd.DataFrame({
                        'timestamp': valid_drops,
                        'value': values[valid_drops].values
                    })
            
            # 7. Irregular fluctuations with reasonable range
            mean_value = values.mean()
            normal_range = mean_value * config['normal_fluctuation']
            rolling_std = values.rolling(window=config['window_size']).std()
            inconsistencies = rolling_std > normal_range
            sensor_anomalies[f'{sensor_type.lower()}_inconsistencies'] = pd.DataFrame({
                'timestamp': values[inconsistencies].index,
                'value': values[inconsistencies].values
            })
            
            # 8. Sensor-specific checks
            if sensor_type == 'Flow' and config['reversal_threshold'] is not None:
                reversal = values < config['reversal_threshold']
                sensor_anomalies['flow_reversal'] = pd.DataFrame({
                    'timestamp': values[reversal].index,
                    'value': values[reversal].values
                })
            
            if sensor_type == 'Rain Gauge' and config['prolonged_window'] is not None:
                prolonged = values.rolling(window=config['prolonged_window']).sum() > values.quantile(0.90)
                sensor_anomalies['rain_prolonged'] = pd.DataFrame({
                    'timestamp': values[prolonged].index,
                    'value': values[prolonged].values
                })
            
            anomalies[sensor_type] = sensor_anomalies
            
        return anomalies

    def detect_correlated_anomalies(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Detect anomalies based on correlations between variables.
        """
        anomalies = {}
        sensor_data = self._prepare_sensor_data(data)
        
        if 'Flow' in sensor_data and 'Depth' in sensor_data and 'Rain Gauge' in sensor_data:
            flow_data = sensor_data['Flow']
            depth_data = sensor_data['Depth']
            rain_data = sensor_data['Rain Gauge']

            # Calculate correlations
            flow_depth_corr = flow_data.corr(depth_data)
            flow_rain_corr = flow_data.corr(rain_data)

            # Mismatch between rain and flow/depth
            flow_depth_anomaly = (abs(flow_depth_corr) < 0.3) & (rain_data > rain_data.quantile(0.5))
            anomalies['flow_depth_anomaly'] = pd.DataFrame({
                'timestamp': flow_depth_anomaly.index,
                'flow': flow_data[flow_depth_anomaly],
                'depth': depth_data[flow_depth_anomaly],
                'rain': rain_data[flow_depth_anomaly]
            })

            # Inconsistent relationships
            flow_depth_rain_anomaly = (flow_depth_corr < 0.3) | (flow_rain_corr < 0.3)
            anomalies['flow_depth_rain_anomaly'] = pd.DataFrame({
                'timestamp': flow_depth_rain_anomaly.index,
                'flow': flow_data[flow_depth_rain_anomaly],
                'depth': depth_data[flow_depth_rain_anomaly],
                'rain': rain_data[flow_depth_rain_anomaly]
            })

        return anomalies

    def detect_sensor_anomalies(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Detect sensor-specific anomalies.
        """
        anomalies = {}
        sensor_data = self._prepare_sensor_data(data)
        
        for sensor_type, values in sensor_data.items():
            # Calculate rate of change
            rate_of_change = values.diff()
            
            # Define drift thresholds based on sensor type
            if sensor_type in ['Depth', 'Flow']:
                # Common parameters
                drift_window = pd.Timedelta(hours=4)  # 4-hour window
                
                # Sensor-specific thresholds
                if sensor_type == 'Depth':
                    drift_threshold = 0.1  # 10cm change
                    noise_threshold = 0.05  # 5cm
                else:  # Flow
                    drift_threshold = values.mean() * 0.2  # 20% of mean flow
                    noise_threshold = values.mean() * 0.1  # 10% of mean flow
                
                # Check for consistent direction
                def all_positive(x):
                    return all(x > 0)
                def all_negative(x):
                    return all(x < 0)
                
                consistent_up = rate_of_change.rolling(window=16).apply(all_positive)
                consistent_down = rate_of_change.rolling(window=16).apply(all_negative)
                
                # Calculate total change in window
                total_change = values.rolling(window=16).apply(lambda x: x.iloc[-1] - x.iloc[0])
                
                # Check for rain events if rain data is available
                if 'Rain Gauge' in sensor_data:
                    rain_data = sensor_data['Rain Gauge']
                    potential_drift = ((consistent_up == True) | (consistent_down == True)) & (abs(total_change) > drift_threshold)
                    
                    # Filter out changes that coincide with rain
                    valid_drift = []
                    for timestamp in values[potential_drift].index:
                        window_data = rain_data[timestamp - drift_window:timestamp]
                        if window_data.empty or window_data.max() < 0.1:  # No significant rain
                            valid_drift.append(timestamp)
                    
                    if valid_drift:
                        anomalies[f'{sensor_type}_drift'] = pd.DataFrame({
                            'timestamp': valid_drift,
                            'value': values[valid_drift].values,
                            'total_change': [total_change[t] for t in valid_drift],
                            'change_percentage': [(total_change[t] / values[t]) * 100 for t in valid_drift]
                        })
                else:
                    # If no rain data, use simpler drift detection
                    drift = ((consistent_up == True) | (consistent_down == True)) & (abs(total_change) > drift_threshold)
                    if drift.any():
                        anomalies[f'{sensor_type}_drift'] = pd.DataFrame({
                            'timestamp': values[drift].index,
                            'value': values[drift].values,
                            'total_change': total_change[drift],
                            'change_percentage': (total_change[drift] / values[drift]) * 100
                        })
            
            # Noise detection - less sensitive
            if sensor_type in ['Depth', 'Flow']:
                # Calculate local variance with a larger window
                local_std = values.rolling(window=20).std()
                # Calculate global variance
                global_std = values.std()
                
                # Set higher noise threshold
                if sensor_type == 'Depth':
                    noise_threshold = 0.10  # 10cm
                else:  # Flow
                    noise_threshold = values.mean() * 0.2  # 20% of mean flow
                
                # Detect noise as high local variance relative to global variance
                noise = (local_std > noise_threshold) & (local_std > 1.5 * global_std)
                if noise.any():
                    anomalies[f'{sensor_type}_noise'] = pd.DataFrame({
                        'timestamp': values[noise].index,
                        'value': values[noise].values,
                        'local_std': local_std[noise],
                        'std_percentage': (local_std[noise] / values[noise]) * 100
                    })

        # Missing data
        missing_data = data[data.isnull().any(axis=1)]
        anomalies['missing_data'] = missing_data

        # Sensor limits
        sensor_anomalies = self.detect_sensor_limits_anomalies(data)
        anomalies.update(sensor_anomalies)

        return anomalies

    def detect_sensor_limits_anomalies(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Detect anomalies based on sensor limits from the database.
        """
        anomalies = {}
        sensor_data = self._prepare_sensor_data(data)
        
        for sensor_type, values in sensor_data.items():
            if sensor_type in self.sensor_limits:
                limits = self.sensor_limits[sensor_type]
                sensor_anomalies = (values < limits['lower']) | (values > limits['upper'])
                anomalies[f'{sensor_type}_limits'] = pd.DataFrame({'timestamp': values[sensor_anomalies].index, 'value': values[sensor_anomalies].values})

        return anomalies

    def generate_anomaly_summary(self, anomalies: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate summary statistics for detected anomalies."""
        summary = {
            "total_anomalies": 0,
            "anomaly_types": {},
            "sensor_specific": {},
            "missing_data": False
        }
        
        # Count anomalies by type
        for anomaly_type, anomaly_list in anomalies.items():
            if anomaly_list:  # Only count if there are anomalies of this type
                count = len(anomaly_list)
                summary["anomaly_types"][anomaly_type] = count
                summary["total_anomalies"] += count
        
        # Add sensor-specific information
        summary["sensor_specific"] = {
            "drift": False,
            "noise": False,
            "out_of_range": False
        }
        
        # Check for missing data
        summary["missing_data"] = any(
            anomaly_type.startswith("missing_") for anomaly_type in anomalies.keys()
        )
        
        return summary

    def __call__(self, query: str) -> Dict[str, Any]:
        """
        Detect anomalies using the provided SQL query.
        
        Args:
            query: SQL query to fetch the data for anomaly detection
        """
        data = self.get_data(query)

        # Run all anomaly detection methods
        sensor_type_anomalies = self.detect_sensor_type_anomalies(data)
        correlated_anomalies = self.detect_correlated_anomalies(data)
        sensor_anomalies = self.detect_sensor_anomalies(data)

        # Combine all anomalies
        all_anomalies = {
            'sensor_type_anomalies': sensor_type_anomalies,
            'correlated_anomalies': correlated_anomalies,
            'sensor_anomalies': sensor_anomalies
        }

        summary = self.generate_anomaly_summary(all_anomalies)
        return {
            'anomalies': all_anomalies,
            'summary': summary,
            'total_records': len(data),
            'timestamp': datetime.now().isoformat()
        }

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL query to fetch the data for anomaly detection."
                }
            },
            "required": ["query"]
        }

# Example usage for direct testing
if __name__ == "__main__":
    try:
        db_connection = mysql.connector.connect(**DB_CONFIG)
        anomaly_tool = AnomalyDetectionTool(db_connection=db_connection)
        # Example: detect anomalies using a query
        query = "SELECT * FROM sensor_data WHERE sensor_id = '11_2'"
        result = anomaly_tool(query=query)
        print("Anomaly Detection Result:")
        print(result)
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'db_connection' in locals() and db_connection.is_connected():
            db_connection.close()
            print("\nDatabase connection closed.")
