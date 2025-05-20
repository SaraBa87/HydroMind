from typing import Dict, Any, Optional, Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import mysql.connector
from mutil_tool_agent.config.db_config import DB_CONFIG
from mutil_tool_agent.tools.anomaly_tool import AnomalyTools
from matplotlib.patches import Rectangle
from fpdf import FPDF
import time
import json

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


    def plot_sensor_with_anomalies(self, sensor_id: str, anomalies: Optional[dict[str, Any]] = None, save_path: Optional[str] = None):
        """
        Plot sensor data with optional anomalies.
        
        Args:
            sensor_id: ID of the sensor
            anomalies: Optional dictionary containing anomaly data. Can be in two formats:
                      1. Original format: {'detect_anomalies_response': {'anomalies': {'sensor_type': {...}}}}
                      2. Simplified format: {'anomalies': [{'anomaly_type': str, 'start_time': str, 'end_time': str, 'duration_hours': float}]}
            save_path: Optional path to save the plot
        """
        try:
            # Debug print
            print("[DEBUG] Anomalies data received for plotting:")
            print(json.dumps(anomalies, indent=2, default=str))

            # Get sensor data
            query = f"SELECT timestamp, value FROM sensor_data WHERE sensor_id = '{sensor_id}'"
            self.cursor.execute(query)
            sensor_data = pd.DataFrame(self.cursor.fetchall())
            if sensor_data.empty:
                print(f"No data found for sensor {sensor_id}")
                return
            sensor_data['timestamp'] = pd.to_datetime(sensor_data['timestamp'])

            # Create the plot
            fig, ax = plt.subplots(figsize=(15, 7))
            ax.set_title(f'Sensor Data for {sensor_id}', fontsize=16)
            
            # Plot sensor data
            ax.plot(sensor_data['timestamp'], sensor_data['value'], 
                   label='Sensor Value', color='blue', zorder=1)
            
            # Plot anomalies if provided
            if anomalies is not None:
                # Handle original format
                if 'detect_anomalies_response' in anomalies:
                    anomaly_data = anomalies['detect_anomalies_response']['anomalies']['sensor_type']
                    for anomaly_type, anomaly_list in anomaly_data.items():
                        if not anomaly_list:
                            continue
                            
                        # Get color for this anomaly type
                        color = self.anomaly_colors.get(anomaly_type.split('_')[-1], 'gray')
                        
                        # Plot each anomaly
                        for anomaly in anomaly_list:
                            timestamp = pd.to_datetime(anomaly['timestamp'])
                            duration = pd.Timedelta(hours=anomaly['duration_hours'])
                            ax.axvspan(timestamp, timestamp + duration, 
                                     color=color, alpha=0.3, 
                                     label=f'{anomaly_type.split("_")[-1]} Anomaly')
                
                # Handle simplified format
                elif 'anomalies' in anomalies and isinstance(anomalies['anomalies'], list):
                    for anomaly in anomalies['anomalies']:
                        anomaly_type = anomaly['anomaly_type']
                        start_time = pd.to_datetime(anomaly['start_time'])
                        end_time = pd.to_datetime(anomaly['end_time'])
                        
                        # Get color for this anomaly type
                        color = self.anomaly_colors.get(anomaly_type, 'gray')
                        
                        # Plot the anomaly
                        ax.axvspan(start_time, end_time, 
                                 color=color, alpha=0.3, 
                                 label=f'{anomaly_type} Anomaly')

            # Customize the plot
            ax.set_xlabel('Time')
            ax.set_ylabel('Sensor Value')
            ax.grid(True, alpha=0.3)
            
            # Add legend (avoid duplicates)
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or show the plot
            if save_path:
                plt.savefig(save_path)
                print(f"Plot saved to {save_path}")
            plt.show()
            plt.close()
            
        except Exception as e:
            print(f"Error plotting sensor data: {str(e)}")
            import traceback
            traceback.print_exc()

    def plot_sensor_vs_rainfall(self, sensor_id: str, save_path: Optional[str] = None):
        """
        Plot sensor data against rainfall with a secondary y-axis.
        
        Args:
            sensor_id: ID of the sensor
            save_path: Optional path to save the plot
        """
        try:
            # Get sensor data
            sensor_query = f"SELECT timestamp, value FROM sensor_data WHERE sensor_id = '{sensor_id}'"
            self.cursor.execute(sensor_query)
            sensor_data = pd.DataFrame(self.cursor.fetchall())
            if sensor_data.empty:
                print(f"No data found for sensor {sensor_id}")
                return
            sensor_data['timestamp'] = pd.to_datetime(sensor_data['timestamp'])

            # Get rainfall data for the same time period
            rainfall_query = f"""
                SELECT timestamp, value 
                FROM sensor_data 
                WHERE sensor_id = '15_3' 
                AND timestamp BETWEEN '{sensor_data['timestamp'].min()}' AND '{sensor_data['timestamp'].max()}'
            """
            self.cursor.execute(rainfall_query)
            rainfall_data = pd.DataFrame(self.cursor.fetchall())
            if not rainfall_data.empty:
                rainfall_data['timestamp'] = pd.to_datetime(rainfall_data['timestamp'])

            # Create the plot
            fig, ax1 = plt.subplots(figsize=(15, 7))
            ax1.set_title(f'Sensor Data vs Rainfall for {sensor_id}', fontsize=16)
            
            # Plot sensor data on primary y-axis
            ax1.plot(sensor_data['timestamp'], sensor_data['value'], 
                    label='Sensor Value', color='blue', zorder=1)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Sensor Value', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            # Create secondary y-axis for rainfall
            ax2 = ax1.twinx()
            if not rainfall_data.empty:
                ax2.plot(rainfall_data['timestamp'], rainfall_data['value'], 
                        label='Rainfall', color='green', alpha=0.7, zorder=2)
                ax2.set_ylabel('Rainfall', color='green')
                ax2.tick_params(axis='y', labelcolor='green')
                # Invert the rainfall axis
                ax2.invert_yaxis()
            
            # Add grid
            ax1.grid(True, alpha=0.3)
            
            # Add legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or show the plot
            if save_path:
                plt.savefig(save_path)
                print(f"Plot saved to {save_path}")
            plt.show()
            plt.close()
            
        except Exception as e:
            print(f"Error plotting sensor vs rainfall data: {str(e)}")
            import traceback
            traceback.print_exc()

    def generate_llm_report_with_retry(self, generate_func, max_retries=3, delay=10):
        """Retry LLM report generation up to max_retries times if a 503 error occurs."""
        for attempt in range(max_retries):
            try:
                return generate_func()
            except Exception as e:
                if ("503" in str(e) or "UNAVAILABLE" in str(e)) and attempt < max_retries - 1:
                    print(f"LLM service overloaded, retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise

    def save_anomalies_to_json(self, data: Dict[str, Any]) -> None:
        """
        Save anomaly results as a text file.
        
        Args:
            data: Dictionary containing anomaly data in the format:
                 {
                     "anomalies": [
                         {
                             "anomaly_type": str,  # e.g., "flat_line", "negative_spike"
                             "start_time": str,    # e.g., "2025-05-02 00:15:00"
                             "end_time": str,      # e.g., "2025-05-02 09:15:00"
                             "duration_hours": float  # e.g., 9.0
                         },
                         ...
                     ]
                 }
        """
        try:
            # Create timestamp for unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            txt_path = os.path.join(self.output_dir, f"anomalies_{timestamp}.txt")
            
            # Format the anomalies as a readable table
            with open(txt_path, 'w') as f:
                f.write("Anomaly Report\n")
                f.write("=============\n\n")
                
                if isinstance(data, dict) and "anomalies" in data:
                    anomalies = data["anomalies"]
                    if isinstance(anomalies, list):
                        for i, anomaly in enumerate(anomalies, 1):
                            f.write(f"Anomaly {i}:\n")
                            f.write(f"  Type: {anomaly.get('anomaly_type', 'Unknown')}\n")
                            f.write(f"  Start: {anomaly.get('start_time', 'Unknown')}\n")
                            f.write(f"  End: {anomaly.get('end_time', 'Unknown')}\n")
                            f.write(f"  Duration: {anomaly.get('duration_hours', 'Unknown')} hours\n")
                            f.write("\n")
                    else:
                        f.write("No anomalies found in the expected format.\n")
                else:
                    f.write("Invalid data format. Expected a dictionary with 'anomalies' key.\n")
            
            print(f"Anomalies saved to {txt_path}")
            
        except Exception as e:
            print(f"Error saving anomalies: {str(e)}")
            import traceback
            traceback.print_exc()

# # Example usage
# if __name__ == "__main__":
#     try:
#         print("Initializing database connection...")
#         db_connection = mysql.connector.connect(**DB_CONFIG)
#         print("Database connection successful")
        
#         print("Initializing tools...")
#         report_tool = ReportTools(db_connection=db_connection)
#         anomaly_tool = AnomalyTools(db_connection=db_connection)
        
#         # Example: Plot anomalies for sensor 13_2
#         print("Detecting anomalies...")
#         anomalies = anomaly_tool.detect_anomalies("SELECT * FROM sensor_data WHERE sensor_id = '13_2'")
#         if 'error' in anomalies:
#             print(f"Error detecting anomalies: {anomalies['error']}")
#         else:
#             print("Anomalies detected successfully")
            
#             # Save anomalies as JSON
#             print("Saving anomalies as JSON...")
#             data = {
#                 'anomalies': anomalies,
#                 'sensor_id': {'value': '13_2'}
#             }
#             report_tool.save_anomalies_to_json(data)
            
#             print("Plotting anomalies...")
#             # Save plot to reports directory
#             save_path = os.path.join("reports", f"anomalies_sensor_13_2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
#             report_tool.plot_sensor_with_anomalies('13_2', anomalies, save_path=save_path)
            
#             # Example: Plot sensor data vs rainfall
#             print("\nPlotting sensor data vs rainfall...")
#             rainfall_plot_path = os.path.join("reports", f"sensor_vs_rainfall_13_2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
#             report_tool.plot_sensor_vs_rainfall('13_2', save_path=rainfall_plot_path)
        
#         db_connection.close()
#         print("Done!")
        
#     except Exception as e:
#         print(f"Error occurred: {str(e)}")
#         import traceback
#         traceback.print_exc() 


