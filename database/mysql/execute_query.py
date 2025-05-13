import mysql.connector
from typing import List, List

def execute_query(sql: str) -> List[List]:
    """
    Execute an SQL statement, returning the results.
    
    Args:
        sql: SQL query to execute
        
    Returns:
        List of rows, where each row is a list of values
    """
    print(f' - DB CALL: execute_query({sql})')
    
    # Database connection
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="95412313@Sara",
        database="hydro_db"
    )
    cursor = db.cursor()
    
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        return results
    finally:
        cursor.close()
        db.close() 