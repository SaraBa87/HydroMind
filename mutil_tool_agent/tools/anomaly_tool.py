import sys
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


# Add the parent directory to Python path to make mutil_tool_agent importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import datetime

from mutil_tool_agent.tools.sql_tool import (
    get_mysql_client,
    run_mysql_validation,
)
import faiss


class AnomalyTools:
    def __init__(self,
                 flatline_duration_hours: float = 5.0,
                 gap_duration_hours: float = 1.0,
                 change_threshold: float = 3.0,
                 spike_duration_hours: float = 1.0,
                 blockage_duration_hours: float = 3.0):
        self.db = get_mysql_client()
        self.cursor = self.db.cursor(dictionary=True)
        self.sensor_limits = self._fetch_sensor_limits()

        # thresholds
        self.flatline_threshold = int(flatline_duration_hours * 4)
        self.gap_threshold = int(gap_duration_hours * 4)
        self.spike_threshold = int(spike_duration_hours * 4)
        self.blockage_threshold = int(blockage_duration_hours * 4)
        self.change_threshold = change_threshold

    def _fetch_sensor_limits(self) -> Dict[str, Dict[str, float]]:
        """Fetch sensor thresholds from DB."""
        query = "SELECT * FROM sensors"
        result = run_mysql_validation(query)
        rows = result.get("query_result") or []
        limits = {}
        for limit in rows:
            limits[limit['sensor_id']] = {
                'sensor_type': limit['sensor_type'],
                'unit': limit['unit'],
                'lower': float(limit['lower_limit']),
                'upper': float(limit['upper_limit']),
            }
        return limits

    def detect_anomalies(self, query: str) -> Dict[str, Any]:
        """Detect anomalies based on SQL query results and generate summary + table."""
        result = run_mysql_validation(query)
        rows = result.get("query_result")
        if not rows:
            return {"error": result.get("error_message") or "No data found"}

        data = pd.DataFrame(rows)
        data['timestamp'] = pd.to_datetime(data['timestamp'])

        anomalies = {
            "sensor_limits": self._detect_sensor_limits_anomalies(data),
            "sensor_type": self._detect_sensor_type_anomalies(data),
            "correlated": self._detect_correlated_anomalies(data),
        }

        # Prepare serializable anomalies for JSON
        serializable_anomalies = {}
        for anomaly_type, type_anomalies in anomalies.items():
            serializable_anomalies[anomaly_type] = {}
            for sensor_id, df in type_anomalies.items():
                df_copy = df.copy()
                for col in df_copy.columns:
                    if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                        df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                serializable_anomalies[anomaly_type][sensor_id] = df_copy.to_dict(orient='records')

        # Generate summary and detailed table
        summary_result = self._generate_anomaly_summary_table(anomalies)
        summary_text = self.display_anomaly_summary(summary_result)
        table_text = self.display_anomaly_table(summary_result)

        return {
            "anomalies": serializable_anomalies,
            "summary": summary_result,
            "summary_text": summary_text,
            "table_text": table_text
        }


    def _detect_sensor_limits_anomalies(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        anomalies = {}
        for sensor_id in data['sensor_id'].unique():
            if sensor_id in self.sensor_limits:
                limits = self.sensor_limits[sensor_id]
                sensor_data = data[data['sensor_id'] == sensor_id]
                out_of_bounds = sensor_data[
                    (sensor_data['value'] < limits['lower']) |
                    (sensor_data['value'] > limits['upper'])
                ]
                if not out_of_bounds.empty:
                    anomalies[sensor_id] = out_of_bounds
        return anomalies

    def _detect_sensor_type_anomalies(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        anomalies = {}
        for sensor_id in data['sensor_id'].unique():
            sensor_data = data[data['sensor_id'] == sensor_id].copy()
            sensor_type = self._get_sensor_type(sensor_id)

            if sensor_type in ['Flow', 'Depth']:
                # --- Negative values ---
                negative_values = sensor_data[sensor_data['value'] < 0]
                if not negative_values.empty:
                    negative_values['detected_by'] = 'negative_value_rule'
                    anomalies[f"{sensor_id}_negative_values"] = negative_values

                # --- Sudden drops (tunable threshold) ---
                sensor_data['delta'] = sensor_data['value'].diff()
                drop_anomalies = sensor_data[sensor_data['delta'] < -1.0]  # e.g., >1m sudden drop
                if not drop_anomalies.empty:
                    drop_anomalies['detected_by'] = 'sudden_drop_rule'
                    anomalies[f"{sensor_id}_sudden_drop"] = drop_anomalies

                # --- Isolation Forest for jumps/drops ---
                try:
                    iso_forest = IsolationForest(contamination=0.01, random_state=42)
                    X = sensor_data[['value']].values
                    sensor_data['anomaly_score'] = iso_forest.fit_predict(X)
                    iso_anomalies = sensor_data[sensor_data['anomaly_score'] == -1]
                    if not iso_anomalies.empty:
                        iso_anomalies['detected_by'] = 'isolation_forest_jump_drop'
                        anomalies[f"{sensor_id}_iso_forest_jump_drop"] = iso_anomalies
                except Exception:
                    pass

                # --- Flatline detection (low variance, relaxed threshold) ---
                try:
                    sensor_data['rolling_std'] = sensor_data['value'].rolling(window=4).std()
                    low_variance = sensor_data[sensor_data['rolling_std'] < 0.002]  # relaxed
                    if not low_variance.empty:
                        low_variance['detected_by'] = 'flatline_rule'
                        anomalies[f"{sensor_id}_flatline"] = low_variance
                except Exception:
                    pass

                # --- LSTM sequence model (abnormal patterns) ---
                try:
                    if len(sensor_data) >= 50:
                        values = sensor_data[['value']].values
                        scaler = StandardScaler()
                        values_scaled = scaler.fit_transform(values)
                        seq_length = 10
                        X, y = [], []
                        for i in range(len(values_scaled) - seq_length):
                            X.append(values_scaled[i:i+seq_length])
                            y.append(values_scaled[i+seq_length])
                        X, y = np.array(X), np.array(y)

                        model = Sequential()
                        model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
                        model.add(Dense(1))
                        model.compile(optimizer='adam', loss='mse')
                        model.fit(X, y, epochs=5, batch_size=16, verbose=0)

                        y_pred = model.predict(X, verbose=0)
                        mse = np.mean(np.power(y - y_pred, 2), axis=1)
                        threshold = np.percentile(mse, 95)
                        anomalies_idx = np.where(mse > threshold)[0]
                        detected = sensor_data.iloc[seq_length + anomalies_idx]
                        if not detected.empty:
                            detected['detected_by'] = 'lstm_sequence_post_rain'
                            anomalies[f"{sensor_id}_post_rain_lstm"] = detected
                except Exception:
                    pass

            elif sensor_type == 'Rain Gauge':
                unrealistic_rain = sensor_data[sensor_data['value'] > 250]
                if not unrealistic_rain.empty:
                    unrealistic_rain['detected_by'] = 'rain_gauge_rule'
                    anomalies[f"{sensor_id}_unrealistic_rain"] = unrealistic_rain

        return anomalies

    def _detect_correlated_anomalies(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        return {}  # TODO: implement correlation-based anomaly detection

    def _get_sensor_type(self, sensor_id: str) -> str:
        query = f"SELECT sensor_type FROM sensors WHERE sensor_id = '{sensor_id}'"
        result = run_mysql_validation(query)
        rows = result.get("query_result") or []
        return rows[0]['sensor_type'] if rows else 'Unknown'

    def _generate_anomaly_summary_table(self, anomalies: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """Generate summary counts and detailed anomaly rows."""
        summary = {"total_anomalies": 0, "by_type": {}, "details": []}
        for anomaly_type, type_anomalies in anomalies.items():
            type_count = 0
            for sensor_anomaly_key, df in type_anomalies.items():
                type_label = sensor_anomaly_key.split('_')[-1]
                summary["by_type"].setdefault(type_label, 0)
                summary["by_type"][type_label] += len(df)
                type_count += len(df)
                for _, row in df.iterrows():
                    summary["details"].append({
                        "sensor_id": row.get("sensor_id"),
                        "start_time": row.get("timestamp"),
                        "end_time": row.get("timestamp"),  # or row.get("end_time") if available
                        "value": row.get("value"),
                        "anomaly_type": type_label
                    })
            summary["total_anomalies"] += type_count
        return summary

    @staticmethod
    def display_anomaly_table(summary_result: Dict[str, Any], relevant_columns: Optional[List[str]] = None) -> str:
        rows = summary_result.get("details", [])
        if not rows:
            return "No anomalies detected"

        all_headers = list(rows[0].keys())
        headers = relevant_columns or all_headers
        col_widths = {h: max(len(str(h)), *(len(str(r.get(h, ""))) for r in rows)) for h in headers}
        separator = "+" + "+".join("-" * (w + 2) for w in col_widths.values()) + "+"
        lines = [separator, "|" + "|".join(f" {h:<{col_widths[h]}} " for h in headers) + "|", separator]
        for row in rows:
            lines.append("|" + "|".join(f" {str(row.get(h,'')):<{col_widths[h]}} " for h in headers) + "|")
        lines.append(separator)
        lines.append(f"\nTotal anomalies: {len(rows)}")
        return "\n".join(lines)

    @staticmethod
    def display_anomaly_summary(summary_result: Dict[str, Any]) -> str:
        lines = [
            "📊 ANOMALY SUMMARY",
            "="*40,
            f"Total anomalies: {summary_result.get('total_anomalies', 0)}",
            "By type:"
        ]
        for anomaly_type, count in summary_result.get("by_type", {}).items():
            lines.append(f"  {anomaly_type}: {count}")
        return "\n".join(lines)

# Standalone functions for agent use
def detect_anomalies_standalone(query: str) -> Dict[str, Any]:
    """
    Standalone function to detect anomalies based on SQL query results.
    This function can be used by the Google ADK agent framework.
    
    Args:
        query: SQL query string to fetch data for anomaly detection
        
    Returns:
        Dict containing anomaly detection results with summary and table
    """
    anomaly_tool = AnomalyTools()
    return anomaly_tool.detect_anomalies(query)


def display_anomaly_summary_standalone(summary_result: Dict[str, Any]) -> str:
    """
    Standalone function to display anomaly summary.
    
    Args:
        summary_result: Summary result from anomaly detection
        
    Returns:
        Formatted summary string
    """
    return AnomalyTools.display_anomaly_summary(summary_result)


def display_anomaly_table_standalone(summary_result: Dict[str, Any], relevant_columns: Optional[List[str]] = None) -> str:
    """
    Standalone function to display anomaly table.
    
    Args:
        summary_result: Summary result from anomaly detection
        relevant_columns: Optional list of columns to display
        
    Returns:
        Formatted table string
    """
    return AnomalyTools.display_anomaly_table(summary_result, relevant_columns)

def retrive_info_from_doc(query: str) -> list:
    """
    Retrieve sensor-specific information from the Kansas City sensor document using TF-IDF + FAISS.
    """
    # Load document with proper encoding
    document_path = 'C:/Users/Sara/Documents/GenAI/HydroMind/mutil_tool_agent/sensors_document.txt'
    with open(document_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split document by sensor entries (using regex to match IDs like 13_2, 12_2)
    import re
    chunks = re.split(r'\n(?=\d+_\d+ – )', content)  # splits at lines starting with "number_number – "
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(chunks).toarray().astype('float32')

    # FAISS index
    index = faiss.IndexFlatIP(doc_vectors.shape[1])
    index.add(doc_vectors)

    # Encode query
    query_vector = vectorizer.transform([query]).toarray().astype('float32')

    # Retrieve top-k similar chunks
    k = min(2, len(chunks))
    distances, indices = index.search(query_vector, k)

    retrived_chunks = [chunks[i] for i in indices[0]]

    return retrived_chunks