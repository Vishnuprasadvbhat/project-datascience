import sys
import logging
from src.logger import logging


def error_message_detail(error, error_detail: sys):
  _,_,exec_tb =error_detail.exc_info() # we need only the info, that is 3rd parameter. 1st two are blank
  file_name = exec_tb.tb_frame.f_code.co_filename
  error_message= "error occured in python script name [{0}] line number [{1}] error message  [{2}] ".format(
    file_name,exec_tb.tb_lineno, str(error)
  )
  return error_message


class CustomException(Exception):
  def __init__(self,error_message,error_details: sys):

    super().__init__(error_message)
    self.error_message= error_message_detail(error_message, error_detail=error_details)


  def __str__(self):
    return self.error_message
  


if __name__ == '__main__':
  try:
    a= 1/0
  except Exception as e:
    logging.info('Zero Division Error')
    raise CustomException(e,sys)

