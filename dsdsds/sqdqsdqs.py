import threading
import time
import sys
import traceback

# Global variable to store error details
error_details = None

def custom_excepthook(type, value, tb):
    global error_details
    error_details = (type, value, tb)
    print("An error occurred in the main code:", ''.join(traceback.format_exception(type, value, tb)))

# Set the custom exception hook
sys.excepthook = custom_excepthook

def monitor():
    start_time = time.time()
    while time.time() - start_time < 10:  # Monitor for 10 seconds
        if error_details:
            print(f"Monitor detected an error: {error_details[1]}")
            return
        time.sleep(1)
    print("Monitoring complete: No errors detected within the time frame.")

def main_code():
    print("Main code is running...")
    time.sleep(5)
    # This will raise an unhandled exception
    raise ValueError("An error occurred in the main code")

if __name__ == "__main__":
    # Start the main code in a separate thread
    main_thread = threading.Thread(target=main_code)
    main_thread.start()

    # Start the monitoring thread
    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()

    # Wait for both threads to complete
    main_thread.join()
    monitor_thread.join()

    print("Main code and monitoring have finished.")
