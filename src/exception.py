import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    """
    This function takes an error and its details and returns a formatted error message.
    It extracts the filename and line number where the error occurred and includes the error message.
    """
    # Get the traceback object from the error details
    _, _, exc_tb = error_detail.exc_info()
    
    # Get the filename where the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Format the error message with the filename, line number, and error description
    error_message = f"Error occurred in script: [{file_name}] at line number: [{exc_tb.tb_lineno}] error message: [{str(error)}]"
        
    return error_message


class CustomException(Exception):
    """
    A custom exception class that extends the base Exception class.
    Provides a detailed error message including the filename and line number.
    """
    def __init__(self, error_message, error_detail: sys):
        """
        Initialize CustomException with an error message and error details.
        """
        super().__init__(error_message)
        # Generate a detailed error message using the helper function
        self.error_message = error_message_detail(error_message, error_detail)
        
    def __str__(self):
        """
        Returns the detailed error message when the exception is converted to a string.
        """
        return self.error_message
