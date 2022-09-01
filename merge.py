import pandas as pd
import os
import glob

path = './data/London Crime/'
# os.chdir(path)
print(os.getcwd())
all_file = [i for i in glob.glob(f'{path}*/*.csv')]
print(all_file)
combined = pd.concat([pd.read_csv(f) for f in all_file])
combined.to_csv('./data/London Crime/all_crime.csv', index=False, encoding='utf-8-sig')