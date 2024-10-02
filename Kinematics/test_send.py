import serial
import time

# Initialize serial communication
def init_serial(port='/dev/ttyUSB0', baudrate=9600):
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"Connected to {port} at {baudrate} baudrate.")
        return ser
    except serial.SerialException as e:
        print(f"Error: {e}")
        return None

# Send string data
def send_string(ser, data):
    if ser is not None:
        try:
            # Send string data as byte stream
            ser.write(data.encode('utf-8'))
            print(f"Sent: {data}")
        except serial.SerialTimeoutException as e:
            print(f"Timeout Error: {e}")
    else:
        print("Serial port not initialized.")

# Main function
if __name__ == "__main__":
    # Set the serial port and baudrate
    port = '/dev/ttyACM0'  # Change the port number as per your system
    baudrate = 57600        # Adjust the baudrate as per your settings

    # Initialize the serial port
    ser = init_serial(port, baudrate)

    try:
        while True:
            # Get string input from the user
            mood_string = input("Enter the mood to Hubert: ")
            
            # Send the string to Hubert
            send_string(ser, mood_string)

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        if ser is not None:
            ser.close()  # Close the serial connection before exiting
