import requests

def get_weather(city):
    api_key = '3844a920fee106a82eb2def1a9505b1f'  # Replace 'YOUR_API_KENaY' with your actual OpenWeatherMap API key
    base_url = f'http://api.openweathermap.org/data/3.0/weather?q={city}&appid={api_key}'

    response = requests.get(base_url)
    data = response.json()

    if data['cod'] == 200:
        weather_data = {
            'temperature': data['main']['temp'],
            'wind_speed': data['wind']['speed'],
            'description': data['weather'][0]['description']
        }
        return weather_data
    else:
        return None

# Example usage
city_name = input("Enter city name: ")
weather_info = get_weather(city_name)

if weather_info:
    print(f'Temperature: {weather_info["temperature"]} K')
    print(f'Wind Speed: {weather_info["wind_speed"]} m/s')
    print(f'Description: {weather_info["description"]}')
else:
    print('City not found or weather data unavailable.')