import sys

def error_msg_detail(error, detail:sys):
    try:
        _,_,exc_tb = detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_no = exc_tb.tb_lineno
    except:
        file_name = 'NA';line_no = 'NA'

    return f'Error occured in python script name [{file_name}] ' \
          f'line number [{line_no}] with error message [{error}]'

class CustomException(Exception):
    def __init__(self, error_msg, error_detail:sys):
        super().__init__(error_msg)
        self.error_msg = error_msg_detail(error_msg, detail=error_detail)

    def __str__(self):
        return self.error_msg