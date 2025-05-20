from typing import List, Tuple
import mysql.connector
from tabulate import tabulate
from datetime import datetime
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mutil_tool_agent.config.db_config import DB_CONFIG

class SQLTools:
    def __init__(self, db_connection: mysql.connector.MySQLConnection):
        self.db = db_connection
        self.cursor = self.db.cursor()

    def list_tables(self) -> list[str]:
        """Retrieve the names of all tables in the database."""
        print(' - DB CALL: list_tables()')
        try:
            self.cursor.execute("SHOW TABLES")
            tables = self.cursor.fetchall()
            return [table[0] for table in tables]
        except mysql.connector.Error as err:
            print(f"Database error: {err}")
            return []

    def describe_table(self, table_name: str) -> list[tuple]:
        """Look up the table schema in MySQL database."""
        print(f' - DB CALL: describe_table({table_name})')
        try:
            self.cursor.execute(f"DESCRIBE {table_name}")
            return self.cursor.fetchall()
        except mysql.connector.Error as err:
            print(f"Database error: {err}")
            return []

    def execute_query(self, sql: str) -> str:
        """Execute an SQL statement and return formatted results."""
        print(f' - DB CALL: execute_query({sql})')
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            column_names = [desc[0] for desc in self.cursor.description]
            
            if not results:
                return "No results found."
            
            # Convert datetime objects to strings and format results
            formatted_results = []
            for row in results:
                formatted_row = []
                for value in row:
                    if isinstance(value, datetime):
                        formatted_row.append(value.isoformat())
                    else:
                        formatted_row.append(str(value))
                formatted_results.append(formatted_row)
            
            # Create and return formatted table with a header
            table = tabulate(formatted_results, headers=column_names, tablefmt="grid")
            return f"\n{table}\n"
            
        except mysql.connector.Error as err:
            print(f"Database error: {err}")
            return f"Error executing query: {err}"

# Example usage
if __name__ == "__main__":
    # Create database connection
    db = mysql.connector.connect(**DB_CONFIG)
    
    # Initialize SQL tool
    sql = SQLTools(db)
    
    # List all tables
    tables = sql.list_tables()
    print("\nAvailable tables:", tables)
    
    # Describe sensor_data table
    columns = sql.describe_table("sensor_data")
    print("\nTable structure for sensor_data:")
    for col in columns:
        print(f"- {col[0]}: {col[1]}")
    
    # Execute a query for sensor 11_2
    print("\nQuery Results for sensor 11_2:")
    result = sql.execute_query("SELECT * FROM sensor_data WHERE c")
    print(result)