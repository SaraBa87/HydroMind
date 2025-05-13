from typing import List, Tuple
import mysql.connector
from tabulate import tabulate
from datetime import datetime

class SQLTools:
    def __init__(self):
        self.db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="95412313@Sara",
            database="hydro_db"
        )
        self.cursor = self.db.cursor()

    def close(self):
        """Explicitly close database connection"""
        if hasattr(self, 'cursor') and self.cursor:
            self.cursor.close()
        if hasattr(self, 'db') and self.db:
            self.db.close()

    def list_tables(self) -> List[str]:
        """Retrieve the names of all tables in the database."""
        print(' - DB CALL: list_tables()')
        self.cursor.execute("SHOW TABLES")
        return [table[0] for table in self.cursor.fetchall()]

    def describe_table(self, table_name: str) -> List[Tuple[str, str]]:
        """
        Look up the table schema in MySQL database.
        
        Args:
            table_name: Name of the table to describe
            
        Returns:
            List of columns, where each entry is a tuple of (column, type)
        """
        print(f' - DB CALL: describe_table({table_name})')
        self.cursor.execute(f"DESCRIBE {table_name}")
        columns = self.cursor.fetchall()
        return [(col[0], col[1]) for col in columns]

    def execute_query(self, sql: str) -> List[List]:
        """
        Execute an SQL statement, returning the results.
        
        Args:
            sql: SQL query to execute
            
        Returns:
            List of rows, where each row is a list of values
        """
        print(f' - DB CALL: execute_query({sql})')
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            
            # Convert datetime objects to strings
            serialized_results = []
            for row in results:
                serialized_row = []
                for value in row:
                    if isinstance(value, datetime):
                        serialized_row.append(value.isoformat())
                    else:
                        serialized_row.append(value)
                serialized_results.append(serialized_row)
            
            return serialized_results
        except mysql.connector.Error as err:
            print(f"Error executing query: {err}")
            return []

    def print_query_results(self, sql: str):
        """
        Execute a query and print results in a nicely formatted table.
        
        Args:
            sql: SQL query to execute
        """
        print(f' - DB CALL: print_query_results({sql})')
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            
            if not results:
                print("No results found.")
                return
                
            # Get column names
            column_names = [desc[0] for desc in self.cursor.description]
            
            # Convert datetime objects to strings for display
            serialized_results = []
            for row in results:
                serialized_row = []
                for value in row:
                    if isinstance(value, datetime):
                        serialized_row.append(value.isoformat())
                    else:
                        serialized_row.append(value)
                serialized_results.append(serialized_row)
            
            # Print results using tabulate
            print("\nQuery Results:")
            print(tabulate(serialized_results, headers=column_names, tablefmt="grid"))
            
        except mysql.connector.Error as err:
            print(f"Error executing query: {err}")

    # def get_sensor_data(self, location_name: str, sensor_type: str, start_time: str = None, end_time: str = None) -> List[List]:
    #     """
    #     Get sensor data for a specific location and sensor type.
        
    #     Args:
    #         location_name: Name of the location
    #         sensor_type: Type of sensor
    #         start_time: Optional start time filter
    #         end_time: Optional end time filter
            
    #     Returns:
    #         List of sensor data rows
    #     """
    #     print(f' - DB CALL: get_sensor_data({location_name}, {sensor_type})')
        
    #     query = """
    #     SELECT sd.timestamp, sd.value, s.unit
    #     FROM sensor_data sd
    #     JOIN locations l ON sd.location_id = l.location_id
    #     JOIN sensors s ON sd.sensor_id = s.sensor_id
    #     WHERE l.location_name = %s AND s.sensor_type = %s
    #     """
    #     params = [location_name, sensor_type]
        
    #     if start_time:
    #         query += " AND sd.timestamp >= %s"
    #         params.append(start_time)
    #     if end_time:
    #         query += " AND sd.timestamp <= %s"
    #         params.append(end_time)
            
    #     query += " ORDER BY sd.timestamp"
        
    #     self.cursor.execute(query, params)
    #     return self.cursor.fetchall()

# db = SQLTools()

# # List all tables
# tables = db.list_tables()
# print("Tables:", tables)

# # Get table structure
# columns = db.describe_table("sensor_data")
# for column_name, column_type in columns:
#     print(f"{column_name}: {column_type}")

# # Execute a query
# results = db.execute_query("SELECT * FROM sensor_data LIMIT 5")

# # Print query results
# db.print_query_results("SELECT * FROM sensor_data WHERE sensor_id = '11_1' AND timestamp BETWEEN '2025-05-02 00:00:00' AND '2025-05-02 04:00:00'")