from google.adk.agents import Agent
from mutil_tool_agent.tools.sql_tool import (
    initial_mysql_nl2sql, 
    run_mysql_validation, 
    display_results_as_table, 
    display_results_summary,
    get_env_var
)
# from mutil_tool_agent.tools.anomaly_tool import AnomalyTools
# from mutil_tool_agent.tools.report_tool import ReportTools
# import mysql.connector

# # Initialize database connection
# db_connection = mysql.connector.connect(
#     host=get_env_var("MYSQL_HOST"),
#     user=get_env_var("MYSQL_USER"),
#     password=get_env_var("MYSQL_PASSWORD"),
#     database=get_env_var("MYSQL_DATABASE"),
# )

# # Initialize anomaly detection tool
# anomaly_tool = AnomalyTools(db_connection=db_connection)

# # Initialize report generation tool
# report_tool = ReportTools(db_connection=db_connection)


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


# anomaly_detection_agent = Agent(
#     name="anomaly_detection_agent",
#     model="gemini-2.0-flash",
#     description="Anomaly detection agent for MySQL database.",
#     instruction=(
#         "You are a helpful sub-agent who can do the following tasks:"
#         "1. The root agent will ask you to do the anomaly detection."
#         "2. You will do the anomaly detection by using the tools available and the sql query provided by the root agent."
#         "3. After detecting anomalies, you will generate a report with professional explanation and recommendations."
#         "4. Then ask the user if they want to see the plots."
#         "5. ALWAYS convert the anomalies to a simplified table format before plotting or saving. The table should include:"
#         "        * anomaly_type (e.g., 'flat_line', 'negative_spike')"
#         "        * start_time (e.g., '2025-05-02 00:15:00')"
#         "        * end_time (e.g., '2025-05-02 09:15:00')"
#         "        * duration_hours (e.g., 9.0)"
#         "5. Before any plotting wait for the user to confirm the plot"
#         "6. If user wants to see the plot, "
#         "      -first plot sensor vs rainfall using plot_sensor_vs_rainfall tool"
#         "      - Ask user to confirm if want to save the anomalies as a json file using save_anomalies_to_json tool"
#         "      - then ask user to see the plot of the anomalies"
#         "      - once user confirms the plot, first show the inputs that you are going to pass to the plot_sensor_with_anomalies tool"
#         "      - Use this simplified format for both plotting and saving anomalies"
#         "      - The plot will show the sensor data with anomalies highlighted in different colors."
#         "      - Make sure to create the reports directory if it doesn't exist."
#         "      - Always use a timestamp in the filename to avoid overwriting previous plots."
#     ),
#     tools=[anomaly_tool.detect_anomalies, report_tool.plot_sensor_with_anomalies, report_tool.plot_sensor_vs_rainfall, report_tool.save_anomalies_to_json],
# )

def return_instructions_mysql() -> str:

    
    instruction_prompt_mysql_v1 = f"""
      You are an AI assistant serving as a SQL expert for MySQL.
      Your job is to help users generate SQL answers from natural language questions (inside Nl2sqlInput).
      You should proeuce the result as NL2SQLOutput.

      Use the provided tools to help generate the most accurate SQL:
      1. First, use initial_mysql_nl2sql tool to generate initial SQL from the question.
      2. You should also validate the SQL you have created for syntax and function errors (Use run_mysql_validation tool). If there are any errors, you should go back and address the error in the SQL. Recreate the SQL based by addressing the error.
      3. Generate the final result in JSON format with four keys: "explain", "sql", "sql_results", "nl_results".
          "explain": "write out step-by-step reasoning to explain how you are generating the query based on the schema, example, and question.",
          "sql": "Output your generated SQL!",
          "sql_results": "raw sql execution query_result from run_mysql_validation if it's available, otherwise None",
          "nl_results": "Natural language about results, otherwise it's None if generated SQL is invalid"
      4. Ask user if they want to see the results summary or table.
      5. If they want to see the results summary, use display_results_summary tool with the result from run_mysql_validation.
      6. If they want to see the results as a table, use display_results_as_table tool with the result from run_mysql_validation.
      7. IMPORTANT: When using display_results_as_table or display_results_summary, pass the COMPLETE result object from run_mysql_validation, not individual rows.
      ```
      You should pass one tool call to another tool call as needed!

      

      NOTE: you should ALWAYS USE THE TOOLS (initial_mysql_nl2sql AND run_mysql_validation) to generate SQL, not make up SQL WITHOUT CALLING TOOLS.
      Keep in mind that you are an orchestration agent, not a SQL expert, so use the tools to help you generate SQL, but do not make up SQL.

    """

    return instruction_prompt_mysql_v1

root_agent = Agent(
    name="data_analysis_agent",
    model="gemini-2.0-flash",
    description=(
        "MySQL database agent for querying, anomaly detection and reporting."
    ),
    instruction=return_instructions_mysql()
    ,
    tools=[initial_mysql_nl2sql, run_mysql_validation, display_results_as_table, display_results_summary],
    # sub_agents=[anomaly_detection_agent]
)