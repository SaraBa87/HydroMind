import mysql.connector
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Connect to MySQL
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='95412313@Sara',
    database='hydro_db'
)

# Step 2: Query the flow and rain data
query = """
    SELECT timestamp, flow, rain
    FROM flow_rain_data
    ORDER BY timestamp ASC
"""

df = pd.read_sql(query, conn, parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)

conn.close()

# Step 3: Create lag features for NARX model

def create_lag_features(df, output_col="flow", input_col="rain", n_lags=3):
    data = pd.DataFrame(index=df.index)
    for i in range(1, n_lags + 1):
        data[f"{output_col}_lag{i}"] = df[output_col].shift(i)
        data[f"{input_col}_lag{i}"] = df[input_col].shift(i)
    data["target"] = df[output_col]
    return data.dropna()

lagged_df = create_lag_features(df, output_col="flow", input_col="rain", n_lags=3)


# Step 4: Train the NARX model
# Split into features and target
X = lagged_df.drop(columns=["target"])
y = lagged_df["target"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train a model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
