import logging 
import os
from datetime import datetime

# Log file name based on the current date
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d')}.log"

# Path to the folder for saving logs
logs_path = os.path.join(os.getcwd(), "logs")

# Create the logs folder if it does not exist
os.makedirs(logs_path, exist_ok=True)

# Full path to the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure the basic settings for logging
logging.basicConfig(
    filename=LOG_FILE_PATH,  # File where logs will be written
    format='[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s',  # Message format
    level=logging.INFO,  # Logging level
) 
