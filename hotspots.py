import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import datetime
from dateutil import relativedelta
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_dataset

if __name__ == '__main__':
    crime_type = ['Drugs', 'Possession of weapons', 'Violence and sexual offences']
    borough = gpd.read_file('data/statistical-gis-boundaries-london/ESRI/London_Borough_Excluding_MHW.shp')
    borough = borough.to_crs(epsg=27700)
    
    fig, ax = plt.subplots(2,3, figsize=(24,12))
    sns.set(style='whitegrid')
    sns.despine(left=True, bottom=True)
    sns.set_context('paper')
    for i in range(0, 3):
        print(crime_type[i])
        data = load_dataset(crime_type[i])
        
        base_month = datetime.datetime(2021, 4, 1)
        split_month = datetime.datetime(2022, 3, 1)
        test_month = split_month + relativedelta.relativedelta(months=1)

        train = data.loc[(data['Month'] >= base_month) & (data['Month'] <= split_month)]
        test = data.loc[data['Month'] == test_month]
        X_train = train.drop(columns=['LSOACD', 'LSOANM', 'LADCD', 'LADNM', 'Month', 'label', 'loc_x', 'loc_y']).values
        X_test = test.drop(columns=['LSOACD', 'LSOANM', 'LADCD', 'LADNM', 'Month', 'label', 'loc_x', 'loc_y']).values
        y_train = train['label'].values
        y_test = test['label'].values

        gt = test.loc[test['label']==1]
        borough.boundary.plot(ax=ax[0, i], zorder=2)
        sns.kdeplot(x=gt['loc_x'], y=gt['loc_y'], shade=True, bw_method='scott', bw_adjust=0.3, cmap='Reds', ax=ax[0, i], zorder=1)
        ax[0,i].set(xlabel='', ylabel='')
        ax[0,i].set_title(f'{crime_type[i]} ground truth', fontsize=16)

        clf = RandomForestClassifier(criterion='entropy', max_samples=0.8, n_estimators=125, n_jobs=-1)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        print(classification_report(y_test, pred))
        pred = pd.DataFrame({'loc_x': test['loc_x'].values, 'loc_y': test['loc_y'].values, 'label': pred})
        pred = pred.loc[pred['label']==1]
        borough.boundary.plot(ax=ax[1, i], zorder=2)
        sns.kdeplot(x=pred['loc_x'], y=pred['loc_y'], shade=True, bw_method='scott', bw_adjust=0.3, cmap='Blues', ax=ax[1, i], zorder=1)
        ax[1,i].set(xlabel='', ylabel='')
        ax[1,i].set_title(f'{crime_type[i]} predicted', fontsize=16)

    plt.tight_layout()
    plt.savefig(f'fig/RF_gt_pred.png', bbox_inches='tight', dpi=200)
    plt.show()
