from google.adk.agents import Agent
from .tools import SQLTools

# Initialize SQL tools
sql_tools = SQLTools()

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
        "3. Once you have the information you need, you will answer the user's question using the data returned."
        "4. Use list_tables to see what tables are present, describe_table to understand the schema, and execute_query to issue an SQL SELECT query." 
        "5. If you don't have information about tabels or columns, you do not need to ask user for those infor,ation, you can use the list_tables or describe_tables tools to get those information and answer uesr's query."
        "6. you can use print_query_results to show query result"
        "6. Delegates tasks like anomaly detection or report generation to SubAgents."
        "7. Maintains context and routing logic."
        "8. Provide a solution to the anomalies."
    ),
    tools=[sql_tools.list_tables, sql_tools.describe_table, sql_tools.execute_query, sql_tools.print_query_results],
)