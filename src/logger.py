import logging 
import os
from datetime import datetime

# Ім'я файлу для логів, яке базується на поточній даті
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d')}.log"

# Шлях до папки для збереження логів
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Створення папки для логів, якщо вона не існує
os.makedirs(logs_path, exist_ok=True)

# Повний шлях до файлу логів
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Налаштування базового конфігуратора для логування
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Файл, куди будуть записуватись логи
    format='[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s',  # Формат повідомлень
    level=logging.INFO,  # Рівень логування
)   
