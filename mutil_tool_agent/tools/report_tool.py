from typing import Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import mysql.connector
from mutil_tool_agent.config.db_config import DB_CONFIG
from mutil_tool_agent.tools.anomaly_tool import AnomalyTools
from matplotlib.patches import Rectangle

class ReportTools:
    def __init__(self, db_connection: mysql.connector.MySQLConnection, output_dir: str = "reports"):
        self.db = db_connection
        self.output_dir = output_dir
        self.cursor = self.db.cursor(dictionary=True)
        # Create reports directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Define colors for different types of anomalies
        self.anomaly_colors = {
            'negative': 'red',              # For negative values
            'flat_line': 'limegreen',      # For flat line anomalies (valid color)
            'abnormal_increase': 'orange',  # For abnormal increases
            'abnormal_decrease': 'purple',  # For abnormal decreases
            'gaps': 'gray',                # For data gaps
            'potential_blockage': 'brown',  # For potential blockages
            'spike': 'orange',             # For sudden spikes
            'negative_spike': 'purple',     # For sudden drops
            'unrealistic_rain': 'navy',     # For unrealistic rainfall (valid color)
            'missing_response': 'pink'      # For missing sensor responses
        }

    def plot_anomalies(self, sensor_id: str, anomalies: Dict[str, Any], save_path: Optional[str] = None):
        """Plot sensor data with rain from top border and highlighted anomalies."""
        try:
            print(f"Fetching data for sensor {sensor_id}...")
            # Get sensor data
            query = f"SELECT timestamp, value FROM sensor_data WHERE sensor_id = '{sensor_id}'"
            self.cursor.execute(query)
            sensor_data = pd.DataFrame(self.cursor.fetchall())
            if sensor_data.empty:
                print(f"No data found for sensor {sensor_id}")
                return
            print(f"Found {len(sensor_data)} sensor readings")
            sensor_data['timestamp'] = pd.to_datetime(sensor_data['timestamp'])

            print("Fetching rain data...")
            # Get rain data for correlation
            rain_query = "SELECT timestamp, value FROM sensor_data WHERE sensor_id = '15_3'"
            self.cursor.execute(rain_query)
            rain_data = pd.DataFrame(self.cursor.fetchall())
            if rain_data.empty:
                print("No rain data found")
                return
            print(f"Found {len(rain_data)} rain readings")
            rain_data['timestamp'] = pd.to_datetime(rain_data['timestamp'])

            print("Creating plot...")
            fig, ax = plt.subplots(figsize=(15, 7))
            ax.set_title(f'Anomaly Analysis for Sensor {sensor_id}', fontsize=16)

            # Plot sensor data on primary y-axis
            ax.plot(sensor_data['timestamp'], sensor_data['value'], label='Sensor Value', color='red', zorder=2)
            ax.set_ylabel('Sensor Value', color='red')
            ax.tick_params(axis='y', labelcolor='red')
            ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)

            # Create secondary y-axis for rainfall
            ax2 = ax.twinx()
            # Plot rainfall as positive bars
            ax2.bar(rain_data['timestamp'], rain_data['value'], width=0.01, color='blue', alpha=0.6, label='Rainfall', zorder=1)
            ax2.set_ylabel('Rainfall (mm)', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            # Invert the y-axis so that 0 is at the top and higher rainfall is at the bottom
            rain_max = rain_data['value'].max() if not rain_data.empty else 1
            ax2.set_ylim(0, rain_max * 1.1)
            ax2.invert_yaxis()

            # Highlight anomalies
            print("Highlighting anomalies...")
            detected_types = []
            for anomaly_type, anomaly_data in anomalies['anomalies']['sensor_type'].items():
                if sensor_id in anomaly_type and anomaly_data:
                    type_label = anomaly_type[len(sensor_id)+1:] if anomaly_type.startswith(sensor_id + '_') else anomaly_type
                    print(f"Processing {type_label} anomalies...")
                    anomaly_df = pd.DataFrame(anomaly_data)
                    if not anomaly_df.empty:
                        detected_types.append(type_label)
                        anomaly_df['timestamp'] = pd.to_datetime(anomaly_df['timestamp'])
                        color = self.anomaly_colors.get(type_label, 'gray')
                        for _, row in anomaly_df.iterrows():
                            ax.axvspan(row['timestamp'], row['timestamp'] + timedelta(hours=1), color=color, alpha=0.3, label=type_label)

            # Build legend only for detected types
            handles, labels = [], []
            for type_label in detected_types:
                color = self.anomaly_colors.get(type_label, 'gray')
                handles.append(plt.Rectangle((0, 0), 1, 1, alpha=0.3, color=color))
                labels.append(type_label.replace('_', ' ').title())
            line1, = ax.plot([], [], color='red', label='Sensor Value')
            bar1 = plt.Rectangle((0, 0), 1, 1, color='blue', alpha=0.6, label='Rainfall')
            handles = [line1, bar1] + handles
            labels = ['Sensor Value', 'Rainfall'] + labels
            ax.legend(handles, labels, loc='upper right')

            fig.tight_layout()
            print("Saving/displaying plot...")
            if save_path:
                plt.savefig(save_path)
                plt.close()
                print(f"Plot saved to {save_path}")
            else:
                plt.show()

        except Exception as e:
            print(f"Error plotting anomalies: {str(e)}")
            import traceback
            traceback.print_exc()

# Example usage
if __name__ == "__main__":
    try:
        print("Initializing database connection...")
        db_connection = mysql.connector.connect(**DB_CONFIG)
        print("Database connection successful")
        
        print("Initializing tools...")
        report_tool = ReportTools(db_connection=db_connection)
        anomaly_tool = AnomalyTools(db_connection=db_connection)
        
        # Example: Plot anomalies for sensor 13_2
        print("Detecting anomalies...")
        anomalies = anomaly_tool.detect_anomalies("SELECT * FROM sensor_data WHERE sensor_id = '13_2'")
        if 'error' in anomalies:
            print(f"Error detecting anomalies: {anomalies['error']}")
        else:
            print("Anomalies detected successfully")
            print("Plotting anomalies...")
            # Save plot to reports directory
            save_path = os.path.join("reports", f"anomalies_sensor_13_2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            report_tool.plot_anomalies('13_2', anomalies, save_path=save_path)
        
        db_connection.close()
        print("Done!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc() 