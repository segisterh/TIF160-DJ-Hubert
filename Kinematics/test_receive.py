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

# Receive string data
def receive_string(ser):
    if ser is not None:
        try:
            # Read a line from the serial port
            if ser.in_waiting > 0:  # Check if there is data waiting to be read
                data = ser.readline().decode('utf-8').rstrip()  # Decode and remove newline characters
                return data
        except Exception as e:
            print(f"Error while receiving data: {e}")
    else:
        print("Serial port not initialized.")
    return None

# Main function
if __name__ == "__main__":
    # Set the serial port and baudrate
    port = '/dev/ttyUSB0'  # Change the port number as per your system
    baudrate = 9600        # Adjust the baudrate as per your settings

    # Initialize the serial port
    ser = init_serial(port, baudrate)

    try:
        while True:
            # Receive string from Hubert
            received_string = receive_string(ser)
            if received_string:
                print(f"Received: {received_string}")

            time.sleep(1)  # Adjust the delay as needed

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        if ser is not None:
            ser.close()  # Close the serial connection before exiting