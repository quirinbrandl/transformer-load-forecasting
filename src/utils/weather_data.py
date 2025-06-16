from datetime import datetime

import maya
import pandas as pd
import requests

OPEN_METEO_URL_FORMAT = "https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lng}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,cloudcover,shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance"


def get_weather_data(lat, lng, start_date, end_date):
    start_date_parsed = datetime.strftime(maya.when(start_date).datetime(), "%Y-%m-%d")
    end_date_parsed = datetime.strftime(maya.when(end_date).datetime(), "%Y-%m-%d")
    url = OPEN_METEO_URL_FORMAT.format(lat=lat, lng=lng, start_date=start_date_parsed, end_date=end_date_parsed)
    data = requests.get(url)
    try:
        data = data.json()["hourly"]
    except:
        raise ValueError(
            f'Failed to parse weather data from response for lat: {lat}, lng: {lng}, start: {start_date}, end: {end_date}')
    data = pd.DataFrame(data)
    data["timestamp"] = data["time"].apply(parse_datetime)
    data = data.drop(columns=['time'])
    return data


def parse_datetime(value):
    return datetime.strftime(maya.parse(value).datetime(), "%Y-%m-%d %H:%M:%S")
