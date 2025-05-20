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

# # Initialize report generation tool
# report_tool = ReportTools(db_connection=db_connection)

anomaly_detection_agent = Agent(
    name="anomaly_detection_agent",
    model="gemini-2.0-flash",
    description="Anomaly detection agent for MySQL database.",
    instruction=(
        "You are a helpful sub-agent who can do the following tasks:"
        "1. You will take the users questions and turn them into SQL queries."
        "2. Use the sql query and do the anomaly detection."
        "3. Use the tools available to you to do the anomaly detection."),
    tools=[anomaly_tool.detect_anomalies]
)

report_generation_agent = Agent(
    name="report_generation_agent",
    model="gemini-2.0-flash",
    description="Report generation agent for MySQL database.",
    instruction=(
        "You are a helpful sub-agent who can generate report once user requests for it."
        "report could be on the sensor data, weather data, or any other data in the database."
        # "you can generate a pdf report and save it to the local directory."
        "If user needs a report for anomalies, you should use the anomaly detection result to generate a report."),
    # tools=[report_tool.generate_anomaly_report, report_tool.generate_sensor_report]
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
        "7. You have two sub-agents at your disposal: anomaly_detection_agent and report_generation_agent."
        "8. If user asks for anomaly detection, you will delegate the task to the anomaly_detection_agent."
        "9. If user asks for report generation, you will delegate the task to the report_generation_agent."
        "10. Maintains context and routing logic."
        "11. Provide a solution to the anomalies."
        "12. NEVER ask the user for information that you can get yourself by using list_tables() and describe_table()."
    ),
    tools=[sql_tools.list_tables, sql_tools.describe_table, sql_tools.execute_query],
    sub_agents=[anomaly_detection_agent, report_generation_agent]
)