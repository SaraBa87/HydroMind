# mysql_agent_tool.py

import os
import logging
import re
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import mysql.connector
from mysql.connector import connect, Error
from google.genai import Client # LLM client for NL→SQL

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Singleton variables
mysql_client = None
database_settings = None
# Get API key from environment variable
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

# Initialize the Google AI client with the API key
llm_client = Client(api_key=google_api_key)

MAX_NUM_ROWS = 80

from google.adk.tools import ToolContext  # Optional: For session state

def get_env_var(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"Environment variable '{var_name}' not set")
    return value


# def get_llm_client() -> Client:
#     global llm_client
#     if llm_client is None:
#         google_api_key = get_env_var("GOOGLE_API_KEY")
#         llm_client = Client(api_key=google_api_key)
#     return llm_client


def get_mysql_client():
    global mysql_client
    if mysql_client is None:
        mysql_client = connect(
            host=get_env_var("MYSQL_HOST"),
            user=get_env_var("MYSQL_USER"),
            password=get_env_var("MYSQL_PASSWORD"),
            database=get_env_var("MYSQL_DATABASE"),
        )
    return mysql_client


def get_database_settings() -> Dict[str, Any]:
    global database_settings
    if database_settings is None:
        database_settings = update_database_settings()
    return database_settings


def update_database_settings() -> Dict[str, Any]:
    global database_settings
    ddl_schema = get_mysql_schema()
    database_settings = {"mysql_ddl_schema": ddl_schema}
    return database_settings


def get_mysql_schema() -> str:
    client = get_mysql_client()
    cursor = client.cursor()
    ddl_statements = ""

    try:
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]
    except Error as err:
        logging.error(f"Error listing tables: {err}")
        return ""

    for table in tables:
        try:
            cursor.execute(f"DESCRIBE {table}")
            columns = cursor.fetchall()
        except Error as err:
            logging.error(f"Error describing table {table}: {err}")
            continue

        ddl_statement = f"CREATE TABLE `{table}` (\n"
        for col in columns:
            name, col_type, null, key, default, extra = col
            ddl_statement += f"  `{name}` {col_type}"
            if null == "NO":
                ddl_statement += " NOT NULL"
            if default is not None:
                ddl_statement += f" DEFAULT '{default}'"
            ddl_statement += ",\n"
        ddl_statement = ddl_statement[:-2] + "\n);\n\n"

        try:
            cursor.execute(f"SELECT * FROM {table} LIMIT 5")
            rows = cursor.fetchall()
            if rows:
                ddl_statement += f"-- Example values for table `{table}`:\n"
                for row in rows:
                    values = []
                    for val in row:
                        if val is None:
                            values.append("NULL")
                        elif isinstance(val, str):
                            values.append(f"'{val}'")
                        else:
                            values.append(str(val))
                    ddl_statement += f"INSERT INTO `{table}` VALUES ({', '.join(values)});\n"
        except Error as err:
            logging.warning(f"Error fetching sample rows for {table}: {err}")

        ddl_statements += ddl_statement

    cursor.close()
    return ddl_statements


def initial_mysql_nl2sql(question: str) -> Dict[str, Any]:
    prompt_template = """
You are a MySQL expert tasked with generating SQL queries from natural language questions.
Use the following schema strictly and do not reference any other columns:

Schema:
```
{SCHEMA}
```

Question:
```
{QUESTION}
```

Write safe, read-only SQL (no UPDATE, DELETE, INSERT, DROP). Limit rows to {MAX_NUM_ROWS}.
"""

    db_settings = get_database_settings()
    ddl_schema = db_settings["mysql_ddl_schema"]

    prompt = prompt_template.format(SCHEMA=ddl_schema, QUESTION=question, MAX_NUM_ROWS=MAX_NUM_ROWS)

    response = llm_client.models.generate_content(
        model=os.getenv("MYSQL_AGENT_MODEL", "text-bison-001"),
        contents=prompt,
        config={"temperature": 0.1},
    )

    sql = response.text or ""
    sql = sql.replace("```sql", "").replace("```", "").strip()

    return {
        "explain": f"Generated SQL for question: {question}",
        "sql": sql,
        "sql_results": None,
        "nl_results": None
    }


def run_mysql_validation(sql_string: str, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    final_result = {"query_result": None, "error_message": None}

    if re.search(r"(?i)\b(update|delete|drop|insert|create|alter|truncate|merge)\b", sql_string):
        final_result["error_message"] = "Invalid SQL: Contains disallowed DML/DDL operations."
        return final_result

    if "limit" not in sql_string.lower():
        sql_string += f" LIMIT {MAX_NUM_ROWS}"

    client = get_mysql_client()
    cursor = client.cursor(dictionary=True)

    try:
        cursor.execute(sql_string)
        rows = cursor.fetchall()
        final_result["query_result"] = rows
        if tool_context is not None:
            tool_context.state["query_result"] = rows
    except Error as e:
        final_result["error_message"] = f"Invalid SQL: {e}"

    cursor.close()
    return final_result


def display_results_as_table(final_result: Dict[str, Any], relevant_columns: Optional[List[str]] = None) -> str:
    """
    Display MySQL results as a formatted table.
    
    Args:
        final_result: The result dictionary from run_mysql_validation (must contain 'query_result' key with list of rows)
        relevant_columns: List of column names to display. If None, shows all columns.
        
    Returns:
        str: Formatted table string
    """
    rows = final_result.get("query_result")
    if not rows:
        return final_result.get("error_message") or "No results to display"

    all_headers = list(rows[0].keys())
    headers = relevant_columns or all_headers

    col_widths = {h: max(len(str(h)), *(len(str(r.get(h, ""))) for r in rows)) for h in headers}

    separator = "+" + "+".join("-" * (w + 2) for w in col_widths.values()) + "+"
    table_lines = [separator]
    table_lines.append("|" + "|".join(f" {h:<{col_widths[h]}} " for h in headers) + "|")
    table_lines.append(separator)
    for row in rows:
        table_lines.append("|" + "|".join(f" {str(row.get(h,'')):<{col_widths[h]}} " for h in headers) + "|")
    table_lines.append(separator)
    table_lines.append(f"\nTotal rows: {len(rows)}")
    return "\n".join(table_lines)


def display_results_summary(final_result: Dict[str, Any]) -> str:
    rows = final_result.get("query_result")
    if not rows:
        return final_result.get("error_message") or "No results to summarize"

    summary_lines = [
        "📊 QUERY RESULTS SUMMARY",
        "="*40,
        f"Total rows returned: {len(rows)}",
        f"Total columns: {len(rows[0].keys())}",
        f"Columns: {', '.join(list(rows[0].keys())[:5])}{'...' if len(rows[0].keys()) > 5 else ''}",
        "Sample data types:"
    ]

    sample_row = rows[0]
    for col, val in sample_row.items():
        dtype = type(val).__name__ if val is not None else "NULL"
        summary_lines.append(f"  {col}: {dtype}")

    return "\n".join(summary_lines)