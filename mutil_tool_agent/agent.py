from google.adk.agents import Agent
from mutil_tool_agent.tools.sql_tool import (
    initial_mysql_nl2sql, 
    run_mysql_validation, 
    display_results_as_table, 
    display_results_summary,
    get_env_var
)

from mutil_tool_agent.tools.anomaly_tool import (
    detect_anomalies_standalone,
    display_anomaly_summary_standalone,
    display_anomaly_table_standalone,
    retrive_info_from_doc
)

def return_instructions_anomaly_detection() -> str:
    
    instruction_prompt_anomaly_detection_v1 = f"""
      You are AI assistant, a sub agent of the root agent, serving as a anomaly detection expert for MySQL database.
      Your job is to help users generate anomaly detection results from natural language questions.
      You should proeuce the result as AnomalyDetectionOutput.

      IMPORTANT: When calling detect_anomalies_standalone, always include ALL required columns in your SQL query:
      - Use: "SELECT sensor_id, timestamp, value FROM sensor_data WHERE sensor_id = 'X'"
      - NOT: "SELECT timestamp, value FROM sensor_data WHERE sensor_id = 'X'"
      - The anomaly detection code requires the sensor_id column to be present

      Use the detect_anomalies tool to generate the anomaly detection results
      Result are generated as summary and table.
      Summary is a summary of the anomaly detection results.
      Table is a table of the anomaly detection results.
      you can add more information and recommendations at the end of the summary and table in the following format:
      "Additional information and recommendations:
      - The sensor data is not consistent with the sensor type.
      ```
      You should use the retrive_info_from_doc tool to retrive information from the anomaly detection document and provide infromation about the sensor types at the top of the summary and table. 
      IMPORTANT: You should show all reslut including sensor types explanation, summary and table, additional information and recommendations all together not seperately.
      IMPORTANT: You should show the final report in a structured way, with a clear title, sensor explanation, summary, table, and recommendations.
      You should pass one tool call to another tool call as needed!

      NOTE: You should not ask user to provide data, you can get data from the database using the sql query provided inside the AnomalyDetectionInput.

    """
    
    return instruction_prompt_anomaly_detection_v1



anomaly_detection_agent = Agent(
    name="anomaly_detection_agent",
    model="gemini-2.0-flash",
    description="Anomaly detection agent for MySQL database.",
    instruction=return_instructions_anomaly_detection(),
    tools=[detect_anomalies_standalone, retrive_info_from_doc],
)

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
      8. If user asks for anomaly detection, deligate the task to the anomaly_detection_agent.
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
    sub_agents=[anomaly_detection_agent]
)