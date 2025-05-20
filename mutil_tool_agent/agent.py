from google.adk.agents import Agent
from mutil_tool_agent.tools.sql_tool import SQLTools
from mutil_tool_agent.tools.anomaly_tool import AnomalyTools
from mutil_tool_agent.tools.report_tool import ReportTools
import mysql.connector
from mutil_tool_agent.config.db_config import DB_CONFIG

# Initialize database connection
db_connection = mysql.connector.connect(**DB_CONFIG)

# Initialize SQL tool
sql_tools = SQLTools(db_connection=db_connection)

# Initialize anomaly detection tool
anomaly_tool = AnomalyTools(db_connection=db_connection)

# Initialize report generation tool
report_tool = ReportTools(db_connection=db_connection)


# report_generation_agent = Agent(
#     name="report_generation_agent",
#     model="gemini-2.0-flash",
#     description="Report generation agent",
#     instruction=(
#         "You are a helpful sub-agent who can generate report once user requests for it. "
#         "Report could be on the sensor data, weather data, or any other data in the database. "
#         "If user needs a report for anomalies, you should use the anomaly detection result which root agent will provide you to generate a report and explain the anomalies. "
#         "First generate a report with professional explanation and show it to the user. "
#         "Then ask the user if they want to see the plot of the anomalies. "
#         "If user wants to see the plot, follow these steps:"
#         "1. The anomaly data will be in this format: {'anomalies': {'sensor_type': {'sensor_id_anomaly_type': [{'timestamp': str, 'value': float, 'duration_hours': float, ...}]}}}"
#         "2. Use plot_sensor_with_anomalies(sensor_id, anomalies, save_path) where:"
#         "   - sensor_id is the ID of the sensor (e.g., '13_2')"
#         "   - anomalies is the anomaly data provided by the root agent"
#         "   - save_path is the path where to save the plot (e.g., 'reports/anomalies_sensor_13_2.png')"
#         "3. The plot will show the sensor data with anomalies highlighted in different colors."
#         "4. Make sure to create the reports directory if it doesn't exist."
#         "5. Always use a timestamp in the filename to avoid overwriting previous plots."
#     ),
#     tools=[report_tool.plot_sensor_with_anomalies]
# )


anomaly_detection_agent = Agent(
    name="anomaly_detection_agent",
    model="gemini-2.0-flash",
    description="Anomaly detection agent for MySQL database.",
    instruction=(
        "You are a helpful sub-agent who can do the following tasks:"
        "1. The root agent will ask you to do the anomaly detection."
        "2. You will do the anomaly detection by using the tools available and the sql query provided by the root agent."
        "3. After detecting anomalies, you will generate a report with professional explanation and recommendations."
        "4. Then ask the user if they want to see the plots."
        "5. ALWAYS convert the anomalies to a simplified table format before plotting or saving. The table should include:"
        "        * anomaly_type (e.g., 'flat_line', 'negative_spike')"
        "        * start_time (e.g., '2025-05-02 00:15:00')"
        "        * end_time (e.g., '2025-05-02 09:15:00')"
        "        * duration_hours (e.g., 9.0)"
        "5. Before any plotting wait for the user to confirm the plot"
        "6. If user wants to see the plot, "
        "      -first plot sensor vs rainfall using plot_sensor_vs_rainfall tool"
        "      - Ask user to confirm if want to save the anomalies as a json file using save_anomalies_to_json tool"
        "      - then ask user to see the plot of the anomalies"
        "      - once user confirms the plot, first show the inputs that you are going to pass to the plot_sensor_with_anomalies tool"
        "      - Use this simplified format for both plotting and saving anomalies"
        "      - The plot will show the sensor data with anomalies highlighted in different colors."
        "      - Make sure to create the reports directory if it doesn't exist."
        "      - Always use a timestamp in the filename to avoid overwriting previous plots."
    ),
    tools=[anomaly_tool.detect_anomalies, report_tool.plot_sensor_with_anomalies, report_tool.plot_sensor_vs_rainfall, report_tool.save_anomalies_to_json],
)


root_agent = Agent(
    name="mysql_anomaly_detection_agent",
    model="gemini-2.0-flash",
    description=(
        "MySQL database agent for querying, anomaly detection and reporting."
    ),
    instruction=(
        "You are a helpful agent who can do the following tasks:"
        "1. Interact with a MySQL database."
        "2. You will take the users questions and turn them into SQL queries using the tools available."
        "3. ALWAYS start by using list_tables() to see available tables, then use describe_table(table_name) to understand the schema. Only verify with the user if you are still unsure after checking the tables and schema."
        "4. Once you have the information you need, you will answer the user's question using the data returned."
        "5. When displaying query results, you MUST ALWAYS show the complete table output returned by execute_query. Do not try to format or modify the table - just show the raw output as is. This applies to ALL queries, regardless of the number of rows."
        "6. After showing the raw table output, if you notice any anomalies you can notify the user and provide additional analysis or observations using the anomaly_detection_agent."
        "7. You have one sub-agent at your disposal: anomaly_detection_agent."
        "8. If user asks for anomaly detection, do the followings:"
        "   1. You should start by using list_tables() to see available tables, then use describe_table(table_name) to understand the schema."
        "   2. Once you have the information you need, you will generate a sql query to get data from the database and do the anomaly detection."
        "   3. Here is an example of the sql query: SELECT * FROM sensor_data WHERE sensor_id = '13_2'"
        "   4. Then you will delegate the task to the anomaly_detection_agent."
        "9. Maintains context and routing logic."
        "10. Provide a solution to the anomalies."
        "11. NEVER ask the user for information that you can get yourself by using list_tables() and describe_table()."
    ),
    tools=[sql_tools.list_tables, sql_tools.describe_table, sql_tools.execute_query],
    sub_agents=[anomaly_detection_agent]
)