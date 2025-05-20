from typing import Dict, Any
import pandas as pd
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import numpy as np
import mysql.connector
from mutil_tool_agent.config.db_config import DB_CONFIG
from mutil_tool_agent.tools.anomaly_tool import AnomalyTools

class ReportTools:
    def __init__(self, db_connection: mysql.connector.MySQLConnection, output_dir: str = "reports"):
        self.db = db_connection
        self.output_dir = output_dir
        # Create reports directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def generate_anomaly_report(self, anomalies: Dict[str, Any], title: str = "Anomaly Detection Report") -> str:
        """Generate a PDF report for anomaly detection results."""
        pdf = FPDF()
        pdf.add_page()
        
        # Add title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, title, ln=True, align='C')
        pdf.ln(10)
        
        # Add timestamp
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(10)
        
        # Add summary
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Summary", ln=True)
        pdf.set_font('Arial', '', 10)
        
        summary = anomalies.get('summary', {})
        pdf.cell(0, 10, f"Total Anomalies: {summary.get('total_anomalies', 0)}", ln=True)
        
        # Add details by type
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Anomalies by Type", ln=True)
        pdf.set_font('Arial', '', 10)
        
        for anomaly_type, count in summary.get('by_type', {}).items():
            pdf.cell(0, 10, f"{anomaly_type}: {count}", ln=True)
        
        # Add details by sensor
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Anomalies by Sensor", ln=True)
        pdf.set_font('Arial', '', 10)
        
        for sensor, count in summary.get('by_sensor', {}).items():
            pdf.cell(0, 10, f"{sensor}: {count}", ln=True)
        
        # Add detailed anomaly information
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Detailed Anomaly Information", ln=True)
        pdf.set_font('Arial', '', 10)
        
        for anomaly_type, type_anomalies in anomalies.get('anomalies', {}).items():
            if type_anomalies:
                pdf.ln(5)
                pdf.set_font('Arial', 'B', 11)
                pdf.cell(0, 10, f"{anomaly_type.upper()}", ln=True)
                pdf.set_font('Arial', '', 10)
                
                for sensor_id, sensor_anomalies in type_anomalies.items():
                    if sensor_anomalies:
                        pdf.cell(0, 10, f"Sensor: {sensor_id}", ln=True)
                        # Convert anomaly data to DataFrame for better formatting
                        df = pd.DataFrame(sensor_anomalies)
                        for _, row in df.iterrows():
                            pdf.cell(0, 10, f"  Timestamp: {row.get('timestamp', 'N/A')}", ln=True)
                            pdf.cell(0, 10, f"  Value: {row.get('value', 'N/A')}", ln=True)
                            if 'duration_hours' in row:
                                pdf.cell(0, 10, f"  Duration: {row['duration_hours']} hours", ln=True)
                            pdf.ln(2)
        
        # Save the PDF
        filename = f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        pdf.output(filepath)
        
        return filepath

    def generate_sensor_report(self, sensor_data: pd.DataFrame, sensor_id: str) -> str:
        """Generate a PDF report for sensor data with visualizations."""
        pdf = FPDF()
        pdf.add_page()
        
        # Add title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, f"Sensor Report: {sensor_id}", ln=True, align='C')
        pdf.ln(10)
        
        # Add timestamp
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(10)
        
        # Query rain gauge data for 15_3
        rain_query = "SELECT timestamp, value FROM sensor_data WHERE sensor_id = '15_3'"
        cursor = self.db.cursor(dictionary=True)
        cursor.execute(rain_query)
        rain_data = pd.DataFrame(cursor.fetchall())
        cursor.close()
        
        # Create and save combined plot of sensor and rain data with dual y-axes
        merged = pd.merge(sensor_data, rain_data, on="timestamp", suffixes=('_sensor', '_rain'))
        fig, ax1 = plt.subplots(figsize=(10, 6))
        color1 = 'tab:green'
        ax1.set_xlabel('Timestamp')
        ax1.set_ylabel('Depth (m)', color=color1)
        ax1.plot(merged['timestamp'], merged['value_sensor'], color=color1, label=f'Sensor {sensor_id}')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim(bottom=0)
        
        ax2 = ax1.twinx()
        color2 = 'tab:blue'
        ax2.set_ylabel('Rainfall (mm)', color=color2)
        ax2.plot(merged['timestamp'], merged['value_rain'], color=color2, label='Rain Gauge (15_3)', alpha=0.7)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(bottom=0)
        
        fig.tight_layout()
        fig.autofmt_xdate()
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')
        
        plot_filename = f"sensor_{sensor_id}_vs_rain_plot.png"
        plot_path = os.path.join(self.output_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        
        # Add plot to PDF
        pdf.image(plot_path, x=10, y=30, w=190)
        pdf.ln(150)  # Move down after the plot
        
        # Add statistics
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Statistics", ln=True)
        pdf.set_font('Arial', '', 10)
        
        stats = sensor_data['value'].describe()
        pdf.cell(0, 10, f"Mean: {stats['mean']:.2f}", ln=True)
        pdf.cell(0, 10, f"Std Dev: {stats['std']:.2f}", ln=True)
        pdf.cell(0, 10, f"Min: {stats['min']:.2f}", ln=True)
        pdf.cell(0, 10, f"Max: {stats['max']:.2f}", ln=True)
        
        # Save the PDF
        filename = f"sensor_report_{sensor_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        pdf.output(filepath)
        
        # Clean up the plot file
        os.remove(plot_path)
        
        return filepath

# Example usage
if __name__ == "__main__":
    try:
        print("Initializing database connection...")
        db_connection = mysql.connector.connect(**DB_CONFIG)
        cursor = db_connection.cursor(dictionary=True)
        
        print("Initializing tools...")
        report_tool = ReportTools(db_connection=db_connection)
        anomaly_tool = AnomalyTools(db_connection=db_connection)
        
        # Query sensor data for 13_2
        print("Querying sensor data...")
        sensor_query = "SELECT timestamp, value FROM sensor_data WHERE sensor_id = '13_2'"
        cursor.execute(sensor_query)
        sensor_data = pd.DataFrame(cursor.fetchall())
        print(f"Found {len(sensor_data)} sensor readings")
        
        # Query rain gauge data for 15_3
        print("Querying rain gauge data...")
        rain_query = "SELECT timestamp, value FROM sensor_data WHERE sensor_id = '15_3'"
        cursor.execute(rain_query)
        rain_data = pd.DataFrame(cursor.fetchall())
        print(f"Found {len(rain_data)} rain gauge readings")
        
        # Example 1: Generate a sensor report for 13_2
        if not sensor_data.empty:
            print("Generating sensor report...")
            sensor_report_path = report_tool.generate_sensor_report(sensor_data, "13_2")
            print(f"Sensor report generated at: {sensor_report_path}")
        else:
            print("No sensor data found for sensor 13_2")
        
        # Example 2: Generate an anomaly report
        print("Detecting anomalies...")
        anomalies = anomaly_tool.detect_anomalies("SELECT * FROM sensor_data WHERE sensor_id = '13_2'")
        print("Generating anomaly report...")
        anomaly_report_path = report_tool.generate_anomaly_report(anomalies)
        print(f"Anomaly report generated at: {anomaly_report_path}")
        
        cursor.close()
        db_connection.close()
        print("Done!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc() 