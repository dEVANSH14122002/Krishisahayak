from flask import Flask, render_template, request
import serial
import re
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

app = Flask(__name__, static_url_path='/assets')

# Load the trained model
model = joblib.load('xgboost_model.pkl')

# Define label mapping based on the training phase
label_map = {
    0: 'Healthy',
    1: 'Stressed',
    2: 'Angular leaf spot',
    3: 'Rust',
    4: 'Unknown'
}

# Define status thresholds based on predicted health status
status_thresholds = {
    'Rust': {'temperature': 27, 'humidity': 60, 'moisture_percentage': 60},
    'Angular leaf spot': {'temperature': 25, 'humidity': 50, 'moisture_percentage': 80}
}

# Function to calculate features from sensor values
def calculate_features(sensor_values):
    # Calculate mean, median, and interquartile range
    mean_value = np.mean(sensor_values)
    median_value = np.median(sensor_values)
    iqr_value = np.percentile(sensor_values, 75) - np.percentile(sensor_values, 25)

    # Calculate noise features (assuming sensor_values contain noise data)
    white_noise = np.random.normal(0, 1, len(sensor_values))
    purple_noise = np.random.normal(0, 1, len(sensor_values))
    pink_noise = np.random.normal(0, 1, len(sensor_values))
    brown_noise = np.random.normal(0, 1, len(sensor_values))

    # Return the calculated features
    return [mean_value, median_value, iqr_value, white_noise.mean(), purple_noise.mean(), pink_noise.mean(), brown_noise.mean()]

# Function to predict health status based on sensor values
def predict_health_status(sensor_values):
    # Calculate features from sensor values
    features = calculate_features(sensor_values)

    # Predict health status using the trained model
    predicted_status = model.predict([features])

    return predicted_status

# Function to map numerical predictions to labels
def map_to_label(prediction):
    return label_map.get(prediction, 'Unknown')

# Read sensor values from the file
def read_sensor_values_from_file(file_path):
    with open(file_path, 'r') as file:
        # Read the content of the file
        content = file.read()
        # Extract sensor values from the content
        sensor_values = [int(value) for value in content.strip('[]').split(',')]
    return sensor_values

# Function to determine status based on predicted health status and sensor values
# Define solutions for each predicted health status
solutions = {
    'Angular leaf spot': [
        "Maintain the temperature above 25 degrees Celsius",
        "Maintain low moisture below 80%",
        "Avoid leaves from getting wet more than 12 hours",
        "Seed treatment with Carbendazim at a concentration of 0.1%, followed by three foliar sprays of Carbendazim at a concentration of 0.05% each, applied at 15-day intervals."
    ],
    'Rust': [
        "Maintain the temperature above 27 degrees Celsius",
        "Maintain low humidity below 60%, moisture below 50%",
        "Use of Triazole fungicides with a protective effect of over 90% when applied preventively",
        "Use of Azoxystrobin (0.1%) to reduce rust severity by up to 93% when applied thrice at 10-day intervals"
    ]
}

# Function to determine status based on predicted health status and sensor values
def determine_status(predicted_status, temperature, humidity, moisture_percentage):
    if predicted_status in status_thresholds:
        thresholds = status_thresholds[predicted_status]
        status = {
            'temperature_status': 'Normal' if temperature > thresholds['temperature'] else 'Low',
            'humidity_status': 'Normal' if humidity < thresholds['humidity'] else 'High',
            'moisture_percentage_status': 'Normal' if moisture_percentage < thresholds['moisture_percentage'] else 'High',
            'solutions': solutions.get(predicted_status, [])
        }
    else:
        status = {
            'temperature_status': 'Unknown',
            'humidity_status': 'Unknown',
            'moisture_percentage_status': 'Unknown',
            'solutions': []
        }
    return status


@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/read_sensor', methods=['POST'])
def read_sensor():
    ser = serial.Serial('COM7', 9600)  # Adjust port and baud rate if needed
    readings = []  # Initialize a list to store sensor readings
    temperature = None
    humidity = None
    moisture_percentage = None

    try:
        max_iterations = 100  # Maximum number of iterations to read data
        count = 0
        while count < max_iterations:  # Continue until maximum iterations reached
            data = ser.readline()  # Read bytes from serial port
            try:
                data_str = data.decode('utf-8').strip()  # Attempt to decode bytes as UTF-8
                print("Received data:", data_str)

                # Check if the data contains sensor readings
                if data_str.isdigit():
                    readings.append(int(data_str))  # Append the reading to the list
                    count += 1
                else:
                    # Parse temperature, humidity, and moisture percentage
                    if data_str.startswith("Temperature:"):
                        temperature = float(re.search(r'\d+\.\d+', data_str).group())
                    elif data_str.startswith("Humidity:"):
                        humidity = float(re.search(r'\d+\.\d+', data_str).group())
                    elif data_str.startswith("Moisture Percentage ="):
                        moisture_percentage = float(re.search(r'\d+\.\d+', data_str).group())

                    # If all data is available, break the loop
                    if all([temperature, humidity, moisture_percentage]):
                        ()
            except UnicodeDecodeError as e:
                print("Error decoding data:", e)
                print("Skipping problematic data...")
                continue  # Skip to the next iteration of the loop if decoding error occurs
    except Exception as e:
        print("Error reading sensor data:", e)
    finally:
        ser.close()  # Close the serial port when done

    # Store temperature, humidity, and moisture percentage values
    try:
        with open('sensor_data.txt', 'w') as file:
            file.write(f"Temperature: {temperature}\n")
            file.write(f"Humidity: {humidity}\n")
            file.write(f"Moisture Percentage: {moisture_percentage}\n")
    except Exception as e:
        print("Error writing sensor data to file:", e)

    # Format sensor readings as a list string
    readings_str = '[' + ', '.join(map(str, readings)) + ']'

    # Write the sensor readings to a file
    try:
        with open('sensor_readings.txt', 'w') as file:
            file.write(readings_str)
    except Exception as e:
        print("Error writing sensor readings to file:", e)

    user_input_sensor_values = read_sensor_values_from_file('rust.txt')

    plt.plot(user_input_sensor_values)
    plt.xlabel('Time')
    plt.ylabel('Sensor Reading')
    plt.title('Sensor Readings Over Time')
    plt.savefig('static/sensor_plot.png')  # Save the plot as a PNG file
    plt.close()  # Close the plot to release resources

    # Predict health status based on user input sensor values
    predicted_health_status = predict_health_status(user_input_sensor_values)

    # Map numerical prediction to label
    predicted_health_status_label = map_to_label(predicted_health_status[0])

    # Determine status based on predicted health status and sensor values
    status = determine_status(predicted_health_status_label, temperature, humidity, moisture_percentage)

    return render_template('index1.html', predicted_status=predicted_health_status_label,
                       temperature=temperature, humidity=humidity, moisture_percentage=moisture_percentage,
                       temperature_status=status['temperature_status'],
                       humidity_status=status['humidity_status'],
                       moisture_percentage_status=status['moisture_percentage_status'],
                       solutions=status['solutions'])

if __name__ == '__main__':
    app.run(debug=True)
