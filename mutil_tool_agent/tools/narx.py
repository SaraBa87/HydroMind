"""
NARX Neural Network for Flow Prediction
Author: [Your Name]
Description: Predicts river flow using rainfall and past flow data with a PyTorch-based NARX model.
"""

import pandas as pd
import numpy as np
import sys
import os
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim

# Add the parent directory to Python path to make mutil_tool_agent importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mutil_tool_agent.tools.sql_tool import (
    get_mysql_client,
)

def fetch_sensor_data(conn, sensor_id, start, end):
    """Fetch sensor data from the database for a given sensor_id and time range."""
    query = f"""
        SELECT timestamp, value
        FROM sensor_data
        WHERE sensor_id = '{sensor_id}'
        AND timestamp BETWEEN '{start}' AND '{end}'
        ORDER BY timestamp ASC
    """
    df = pd.read_sql(query, conn, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df

def merge_and_prepare_data(df_flow, df_rain):
    """Merge flow and rain DataFrames and rename columns."""
    df = pd.merge(df_flow, df_rain, on="timestamp", how="inner")
    df = df.rename(columns={'value_x': 'flow', 'value_y': 'rain'})
    return df

def detect_largest_rain_event(df, rain_threshold=0.1):
    """Detect the largest rain event in the DataFrame."""
    df_reset = df.reset_index()
    df_reset['is_rain_event'] = df_reset['rain'] > rain_threshold
    df_reset['rain_event_group'] = (df_reset['is_rain_event'] != df_reset['is_rain_event'].shift()).cumsum()
    rain_events = df_reset[df_reset['is_rain_event']].groupby('rain_event_group').agg({
        'rain': 'sum',
        'timestamp': ['min', 'max']
    }).reset_index()
    rain_events.columns = ['event_group', 'total_rain', 'start_time', 'end_time']
    rain_events['duration'] = (rain_events['end_time'] - rain_events['start_time']).dt.total_seconds() / 3600
    largest_event = rain_events.loc[rain_events['total_rain'].idxmax()]
    print(f"\nLargest rain event:")
    print(f"Start: {largest_event['start_time']}")
    print(f"End: {largest_event['end_time']}")
    print(f"Duration: {largest_event['duration']:.2f} hours")
    print(f"Total rainfall: {largest_event['total_rain']:.2f} mm")
    return largest_event

def create_lag_features(df, output_col="flow", input_col="rain", n_lags=3):
    """Create lag features for NARX model."""
    data = pd.DataFrame(index=df.index)
    for i in range(1, n_lags + 1):
        data[f"{output_col}_t{i-1}"] = df[output_col].shift(i)
        data[f"{input_col}_t{i-1}"] = df[input_col].shift(i)
    data["target"] = df[output_col]
    return data.dropna()

class NARXNet(nn.Module):
    """PyTorch neural network for NARX modeling."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

def plot_results(df, train_index, test_index, y_train_tensor, y_test_tensor, y_pred, test_start, test_end, y_scaler):
    """Plot flow, rainfall, predictions, and errors."""
    # Rescale the training and test tensors back to original scale
    y_train = y_scaler.inverse_transform(y_train_tensor.numpy())
    y_test = y_scaler.inverse_transform(y_test_tensor.numpy())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[2, 1])
    fig.suptitle('Flow Prediction Analysis', fontsize=16)

    # Plot 1: Flow and Rainfall
    # ax1.set_title('Flow and Rainfall with Predictions', fontsize=14)
    ax1.plot(df.index, df['flow'], label='Flow (Full)', color='red', zorder=1)
    ax1.plot(test_index, y_pred.ravel(), label='Predictions', color='purple', linestyle='--')
    ax1.axvspan(test_start, test_end, alpha=0.2, color='blue', label='Rain Event Period')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Flow (m³/s)', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.set_ylim(0, 200)
    ax2_rain = ax1.twinx()
    ax2_rain.plot(df.index, df['rain'], label='Rainfall', color='blue', alpha=0.6, zorder=2)
    ax2_rain.set_ylabel('Rainfall (mm)', color='blue')
    ax2_rain.tick_params(axis='y', labelcolor='blue')
    ax2_rain.set_ylim(0, 50)
    ax2_rain.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2_rain.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Plot 2: Prediction Errors
    ax2.set_title('Prediction Errors', fontsize=14)
    errors = y_test.ravel() - y_pred.ravel()
    ax2.plot(test_index, errors, color='red', label='Prediction Error')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.fill_between(test_index, errors, 0, alpha=0.2, color='red')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Error (m³/s)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    print("\nError Statistics:")
    print(f"Mean Error: {errors.mean():.2f} m³/s")
    print(f"Std Error: {errors.std():.2f} m³/s")
    print(f"Max Error: {errors.max():.2f} m³/s")
    print(f"Min Error: {errors.min():.2f} m³/s")

def prepare_data_for_training(df, largest_event, n_lags=3):
    """Prepare data by creating lag features, splitting, scaling, and converting to PyTorch tensors."""
    lagged_df = create_lag_features(df, output_col="flow", input_col="rain", n_lags=n_lags)
    X = lagged_df.drop(columns=["target"]).values
    y = lagged_df["target"].values.reshape(-1, 1)
    test_start = largest_event['start_time']
    test_end = largest_event['end_time'] + pd.Timedelta(hours=12)
    target_index = lagged_df.index[-len(y):]
    train_mask = (lagged_df.index < test_start) | (lagged_df.index > test_end)
    test_mask = (lagged_df.index >= test_start) & (lagged_df.index <= test_end)
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, target_index, train_mask, test_mask, x_scaler, y_scaler

def train_model(X_train_tensor, y_train_tensor, input_dim, n_epochs=100):
    """Train the NARX model."""
    model = NARXNet(input_dim=input_dim)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred_train = model(X_train_tensor)
        loss = loss_fn(y_pred_train, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}")
    return model

def evaluate_model(model, X_test_tensor, y_test, y_scaler):
    """Evaluate the model and return predictions and metrics."""
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).numpy()
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print("\nModel Performance Metrics:")
    print(f"RMSE: {rmse:.2f} m³/s")
    print(f"Mean Flow: {y_test.mean():.2f} m³/s")
    print(f"Relative Error: {(rmse/y_test.mean()*100):.2f}%")
    return y_pred

def main():
    """Main function to run the NARX model pipeline."""
    # Step 1: Connect to MySQL
    conn = get_mysql_client()
    # Step 2: Query the flow and rain data
    df_flow = fetch_sensor_data(conn, '13_1', '2025-05-17 00:00:00', '2025-05-22 23:59:59')
    df_rain = fetch_sensor_data(conn, '15_3', '2025-05-17 00:00:00', '2025-05-22 23:59:59')
    df = merge_and_prepare_data(df_flow, df_rain)
    # Step 3: Detect largest rain event
    largest_event = detect_largest_rain_event(df)
    # Step 4: Prepare data for training
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, target_index, train_mask, test_mask, x_scaler, y_scaler = prepare_data_for_training(df, largest_event)
    print(f"\nTraining set size: {len(X_train_tensor)}")
    print(f"Test set size: {len(X_test_tensor)}")
    # Step 5: Train the model
    model = train_model(X_train_tensor, y_train_tensor, X_train_tensor.shape[1])
    # Step 6: Evaluate the model
    y_pred = evaluate_model(model, X_test_tensor, y_test_tensor, y_scaler)
    # Step 7: Plot results
    train_index = target_index[train_mask]
    test_index = target_index[test_mask]
    plot_results(df, train_index, test_index, y_train_tensor, y_test_tensor, y_pred, largest_event['start_time'], largest_event['end_time'] + pd.Timedelta(hours=12), y_scaler)
    conn.close()

if __name__ == "__main__":
    main()
