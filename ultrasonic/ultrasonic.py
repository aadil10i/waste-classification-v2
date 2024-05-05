import RPi.GPIO as GPIO # type: ignore
import time
from supabase import create_client, Client # type: ignore

# Supabase Connection Details (replace with your own)
SUPABASE_URL = 'SUPABASE_URL'
SUPABASE_KEY = 'SUPABASE_KEY'

# Ultrasonic Sensor Pins (adjust as needed)
TRIGGER_PIN = 18
ECHO_PIN = 16

# Function to measure distance
def measure_distance():
    GPIO.output(TRIGGER_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIGGER_PIN, False)
    start_time = time.time()
    stop_time = time.time()

    while GPIO.input(ECHO_PIN) == 0:
        start_time = time.time()

    while GPIO.input(ECHO_PIN) == 1:
        stop_time = time.time()

    elapsed_time = stop_time - start_time
    distance = (elapsed_time * 34300) / 2
    return distance

# Function to update Supabase
def update_bin_status(is_full):
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    data = supabase.table('bin_status').update({'is_full': is_full}).execute()
    print(f"Updated bin status to: {is_full}")

# Main program
if __name__ == '__main__':
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(TRIGGER_PIN, GPIO.OUT)
        GPIO.setup(ECHO_PIN, GPIO.IN)

        while True:
            distance = measure_distance()
            print(f"Distance: {distance:.1f} cm")

            # Set threshold distance for bin full detection (adjust as needed)
            if distance < 10:
                update_bin_status(True)
            else:
                update_bin_status(False)

            time.sleep(1) # Adjust delay between measurements as needed

    except KeyboardInterrupt:
        print("Measurement stopped by user")
        GPIO.cleanup()