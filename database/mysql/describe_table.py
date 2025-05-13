import mysql.connector
from typing import List, Tuple

def describe_table(table_name: str) -> List[Tuple[str, str]]:
    """
    Look up the table schema in MySQL database.
    
    Args:
        table_name: Name of the table to describe
        
    Returns:
        List of columns, where each entry is a tuple of (column, type)
    """
    print(f' - DB CALL: describe_table({table_name})')
    
    # Database connection
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="95412313@Sara",
        database="hydro_db"
    )
    cursor = db.cursor()
    
    # Get table structure
    cursor.execute(f"DESCRIBE {table_name}")
    
    # Fetch all columns
    columns = cursor.fetchall()
    
    # Close connection
    cursor.close()
    db.close()
    
    # Return list of (column_name, column_type) tuples
    return [(col[0], col[1]) for col in columns] 