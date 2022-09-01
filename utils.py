# some utility functions
import os
import glob
import pandas as pd
import logging
from datetime import datetime


def combine_csv(path):
    # os.chdir(path)
    print(os.getcwd())
    all_file = [i for i in glob.glob(f'{path}*.csv')]
    combined = pd.concat([pd.read_csv(f) for f in all_file])
    return combined


# load training and test set for each crime type
def load_dataset(c_type:str):
    data_path = f'./data/dataset/{c_type}/train_test.csv'
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
    else:
        print('generating datasets...')
        data = combine_csv(f'./data/dataset/{c_type}/')
        data.to_csv(data_path, index=False)
    data['Month'] = pd.to_datetime(data['Month'], yearfirst=True)
    return data


def get_logger(logdir):
    logger = logging.getLogger("ptsemseg")
    ts = str(datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger
