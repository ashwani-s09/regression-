# IT WILL SAME FOR EVERY PROJECTS JUST WHAT WE CAN CHANGE , WE CAN ONLY CHANGE THE FORMAT OF PRINTING IT 

import sys
from logger import logging # from sys.logger import logging

def error_message_detail(error,error_details:sys):
    _,_,exc_tb = error_detail,exc_info()
    file_name= exc_tb.tb_frame.f_code.co_filename

    error_message="Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):

    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message= error_messsage_details(error_message,error_details=error_detais)

    def __str__(self):
        return self.error_message

'''
if __name__=="__main__":
    logging.info("Logging has started")

    try:
        a=1/0
    except Exception as e:
        logging.info('Dicision by zero') 
        raise CustomException(e,sys)
'''