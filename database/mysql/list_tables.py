import mysql.connector

# Database connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="95412313@Sara",
    database="hydro_db"
)
cursor = db.cursor()

# Get list of tables
cursor.execute("SHOW TABLES")

# Print all tables
print("\nTables in hydro_db database:")
print("-" * 30)
for table in cursor:
    print(table[0])

cursor.close()
db.close() 