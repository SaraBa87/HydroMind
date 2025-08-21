# HydroMind

HydroMind is an AI agent for anomaly detection and analysis of hydrological sensor data, with a focus on Kansas City’s sensor networks. It supports anomaly detection, sensor documentation retrieval, and interaction with a MySQL database backend. 

The data used in HydroMind comes from:
- USGS – stream stage and flow measurements.
- Meteostat – historical and real-time precipitation data from local weather stations in the Kansas City area.

## Features
- **Anomaly Detection**: Detects outliers, flatlines, sudden drops, and other anomalies in sensor data using statistical rules, Isolation Forest, and LSTM models.
- **Sensor Documentation Retrieval**: Uses TF-IDF and FAISS to retrieve relevant sensor documentation based on queries.
- **Database Integration**: Connects to a MySQL database to fetch sensor data and metadata.
- **Extensible Tools**: Modular design allows for easy extension and integration of new tools and models.

## Agent Architecture

HydroMind uses a multi-agent architecture for data analysis and anomaly detection:

- **Root Agent (`data_analysis_agent`)**: 
    - Orchestrates SQL query generation, validation, and reporting for the MySQL database. 
    - Delegates anomaly detection tasks to a specialized sub-agent when needed.
- **Anomaly Detection Agent (`anomaly_detection_agent`)**: 
    - Focused on detecting anomalies in sensor data.
    - Generates anomaly detection results from natural language questions.
    - Ensures required columns are included in SQL queries for anomaly analysis.
    - Uses:
        - `detect_anomalies_standalone` → anomaly detection pipeline.
        - `retrive_info_from_doc` → RAG-based system for sensor-specific context.
    - Outputs results in a structured format:
        - Sensor explanation
        - Summary statistics
        - Detailed anomaly tables
        - Recommendations

Agents communicate and pass tool results between each other as needed, providing a seamless experience for both SQL querying and anomaly detection.