import requests
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Define stations and their names
STATIONS = {
    '06893578': 'Blue River at Stadium Drive',
    '06893562': 'Brush Creek at Rockhill Road',
    '06893557': 'Brush Creek at Ward Parkway',
    '06893000': 'Missouri River'
}

def fetch_station_data(station_id, start_date, end_date, parameter_code):
    """
    Fetch data from USGS NWIS API for a specific station and parameter.
    
    Args:
        station_id (str): USGS station ID
        start_date (datetime): Start date
        end_date (datetime): End date
        parameter_code (str): Parameter code ('00060' for flow, '00065' for gage height)
    
    Returns:
        pandas.DataFrame: Processed data with appropriate columns
    """
    # Format the start and end dates in YYYY-MM-DD format
    start = start_date.strftime('%Y-%m-%d')
    end = end_date.strftime('%Y-%m-%d')
    
    # USGS NWIS API URL
    url = f'https://waterservices.usgs.gov/nwis/iv/?site={station_id}&startDT={start}&endDT={end}&parameterCd={parameter_code}&format=rdb'
    
    # Send a GET request to the API
    response = requests.get(url)
    
    if response.status_code == 200:
        # Process the response
        data = response.text.splitlines()
        # Skip metadata lines and header rows
        data = [line for line in data if not line.startswith('#') and not line.startswith('agency_cd') and not line.startswith('5s')]
        df = pd.DataFrame([line.split('\t') for line in data])
        
        if df.shape[1] >= 6:
            # Extract relevant columns
            df = df.iloc[:, [1, 2, 4]]
            
            if parameter_code == '00060':  # Flow data
                df.columns = ['site_no', 'datetime', 'flow_cfs']
                df['flow_cfs'] = pd.to_numeric(df['flow_cfs'], errors='coerce')
                df['flow_m3s'] = df['flow_cfs'] * 0.028316847
            else:  # Gage height data
                df.columns = ['site_no', 'datetime', 'gage_height_ft']
                df['gage_height_ft'] = pd.to_numeric(df['gage_height_ft'], errors='coerce')
                df['gage_height_m'] = df['gage_height_ft'] * 0.3048
            
            # Clean datetime
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df.dropna(subset=['datetime'], inplace=True)
            df.set_index('datetime', inplace=True)
            
            return df
    return None

def save_station_data(df, station_id, station_name, data_type, save_csv=True, save_plot=True):
    """
    Save station data to CSV and create a plot.
    
    Args:
        df (pandas.DataFrame): Station data DataFrame
        station_id (str): Station ID
        station_name (str): Station name
        data_type (str): Type of data ('flow' or 'depth')
        save_csv (bool): Whether to save to CSV
        save_plot (bool): Whether to save plot
    """
    if df is not None:
        if save_plot:
            plt.figure(figsize=(12, 6))
            
            if data_type == 'flow':
                plt.plot(df.index, df['flow_m3s'])
                plt.ylabel('Discharge (m³/s)')
                title = f'Flow Data for {station_name}'
                filename = f'flow_{station_id}.png'
            else:
                plt.plot(df.index, df['gage_height_m'])
                plt.ylabel('Gage Height (m)')
                title = f'Gage Height Data for {station_name}'
                filename = f'depth_{station_id}.png'
            
            plt.title(f'{title} (May 2 - May 9, 2025)')
            plt.xlabel('Date')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(filename)
            print(f"Saved {data_type} plot for {station_name} to {filename}")
        
        if save_csv:
            csv_file = f'{data_type}_{station_id}.csv'
            df.to_csv(csv_file)
            print(f"Saved {data_type} data for {station_name} to {csv_file}")

if __name__ == "__main__":
    # Example usage
    start_date = datetime(2025, 5, 15)
    end_date = datetime(2025, 5, 22)
    
    for station_id, station_name in STATIONS.items():
        print(f"\nProcessing {station_name} (Station ID: {station_id})")
        
        # Fetch and process flow data
        flow_data = fetch_station_data(station_id, start_date, end_date, '00060')
        save_station_data(flow_data, station_id, station_name, 'flow')
        
        # Fetch and process gage height data
        depth_data = fetch_station_data(station_id, start_date, end_date, '00065')
        save_station_data(depth_data, station_id, station_name, 'depth') 