import pandas as pd 
import time
import logging

class Logger():
    
    def __init__(self,
                logger_type='csv',
                log_path = './log/csv_logs/'):


        self.logger_type = logger_type
        self.log_path = log_path
        self.file_path = self.log_path + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '.csv'

        self.base_log = logging.basicConfig(level=logging.DEBUG)

    def log_record_csv(self, info):
        # file_path = 
        # print(info)
        columns_name = [f'layer_{x}' for x in range(len(info))]
        info = [info]
        info = pd.DataFrame(info, columns=columns_name)
        info.to_csv(self.file_path, mode='a', header=False)

    def log_record(self, info, level):

        if level == 'DEBUG':
            logging.debug(info)
        elif level == 'INFO':
            logging.info(info)
        elif level == 'ERROR':
            logging.info(info)

