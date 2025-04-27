import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    """
    Ця функція приймає помилку та її деталі і повертає відформатоване повідомлення про помилку.
    Вона витягує ім'я файлу та номер рядка, де сталася помилка, і включає повідомлення про помилку.
    """
    # Отримуємо об'єкт traceback з деталей помилки
    _, _, exc_tb = error_detail.exc_info()
    
    # Отримуємо ім'я файлу, в якому сталася помилка
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Форматуємо повідомлення про помилку з ім'ям файлу, номером рядка та описом помилки
    error_message = f"Error occurred in script: [{file_name}] at line number: [{exc_tb.tb_lineno}] error message: [{str(error)}]"
        
    return error_message


class CustomException(Exception):
    """
    Клас користувацького виключення, який розширює базовий клас Exception.
    Надає детальне повідомлення про помилку, включаючи ім'я файлу та номер рядка.
    """
    def __init__(self, error_message, error_detail: sys):
        """
        Ініціалізуємо CustomException з повідомленням про помилку та деталями помилки.
        """
        super().__init__(error_message)
        # Генеруємо детальне повідомлення про помилку за допомогою допоміжної функції
        self.error_message = error_message_detail(error_message, error_detail)
        
    def __str__(self):
        """
        Повертає детальне повідомлення про помилку, коли виключення перетворюється на рядок.
        """
        return self.error_message
