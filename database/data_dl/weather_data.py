from meteostat import Hourly, Stations
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

def fetch_weather_data(start_date=None, end_date=None):
    """
    Fetch weather data for Kansas City area and return as a DataFrame with 15-minute intervals.
    Adjusts UTC timestamps to match local time of flow/depth data.
    
    Args:
        start_date (datetime, optional): Start date. Defaults to May 2, 2025.
        end_date (datetime, optional): End date. Defaults to May 8, 2025.
    
    Returns:
        pandas.DataFrame: Weather data with 15-minute intervals in local time
    """
    if start_date is None:
        start_date = datetime(2025, 5, 2)
    if end_date is None:
        end_date = datetime(2025, 5, 8)
    
    # Add 37 hours to compensate for the delay between rain and flow
    start_date = start_date + timedelta(hours=37)
    end_date = end_date + timedelta(hours=37)
    
    # Search for nearest station to Kansas City
    stations = Stations()
    station = stations.nearby(39.0997, -94.5786).fetch(1)
    station_id = station.index[0]
    print(f"Using weather station: {station_id}")
    
    # Fetch hourly data
    data = Hourly(station_id, start_date, end_date).fetch()
    data = data.reset_index()
    
    # Subtract 37 hours to convert back to local time
    data['time'] = pd.to_datetime(data['time']) + timedelta(hours=11)
    
    # Set datetime index for resampling
    data.set_index("time", inplace=True)
    
    # Resample to 15-minute intervals with linear interpolation
    data_15min = data.resample("15min").interpolate(method="linear")
    
    return data_15min

def save_weather_data(df, save_csv=True, save_plot=True):
    """
    Save weather data to CSV and create a plot.
    
    Args:
        df (pandas.DataFrame): Weather data DataFrame
        save_csv (bool): Whether to save to CSV
        save_plot (bool): Whether to save plot
    """
    if save_csv:
        df.to_csv("kc_15min_weather.csv")
        print("Data has been saved to 'kc_15min_weather.csv'")
    
    if save_plot:
        df["prcp"].plot(title="Estimated 15-Minute Precipitation (mm)", figsize=(12, 5))
        plt.ylabel("Precipitation (mm)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('precipitation_15min.png')
        print("Plot has been saved to 'precipitation_15min.png'")

if __name__ == "__main__":
    # Example usage
    data = fetch_weather_data()
    save_weather_data(data)
