import logging
import os
from datetime import datetime
from src.utils import getPath

project_path = getPath()
log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path = os.path.join(project_path,'logs')
os.makedirs(log_path, exist_ok=True)
log_file_path = os.path.join(log_path,log_file)

logging.basicConfig(
    filename= log_file_path,
    format='[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def log(msg):
    print(msg)
    logging.info(msg)
