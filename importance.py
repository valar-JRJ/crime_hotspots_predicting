import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import datetime
from dateutil import relativedelta
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    crime_type = ['Drugs', 'Possession of weapons', 'Violence and sexual offences']
    fig, ax = plt.subplots(1,3, figsize=(24,6))
    sns.set_theme(style='whitegrid')
    for i in range(0, 3):
        data_path = f'./data/dataset/{crime_type[i]}/train_test.csv'
        print(crime_type[i])
        data = pd.read_csv(data_path)
        data['Month'] = pd.to_datetime(data['Month'], yearfirst=True)
        data = data.rename(columns={'Living Environment': 'Living Env.'})

        base_month = datetime.datetime(2021, 1, 1)
        split_month = datetime.datetime(2021, 12, 1)
        test_month = split_month + relativedelta.relativedelta(months=1)

        mean = np.zeros(14)
        std = np.zeros(14)
        while test_month < datetime.datetime(2022, 5, 1):
            train = data.loc[(data['Month'] >= base_month) & (data['Month'] <= split_month)]
            test = data.loc[data['Month'] == test_month]
            train = train.drop(columns=['LSOACD', 'LSOANM', 'LADCD', 'LADNM', 'Month'])
            test = test.drop(columns=['LSOACD', 'LSOANM', 'LADCD', 'LADNM', 'Month'])
            X_train = train.drop(columns=['label', 'loc_x', 'loc_y'])
            col = X_train.columns
            X_train = X_train.values
            X_test = test.drop(columns=['label', 'loc_x', 'loc_y']).values

            y_train = train['label'].values
            y_test = test['label'].values

            clf = RandomForestClassifier(criterion='entropy', max_samples=0.8, n_estimators=125, n_jobs=-1)
            clf.fit(X_train, y_train)
            result = permutation_importance(clf, X_test, y_test, scoring='f1_weighted', n_jobs=-1)
            mean += result.importances_mean
            std += result.importances_std

            base_month = base_month + relativedelta.relativedelta(months=1)
            split_month = split_month + relativedelta.relativedelta(months=1)
            test_month = test_month + relativedelta.relativedelta(months=1)

        mean = mean/4
        std = std/4
        importance = {'Importance': mean, 'std': std}
        col = [attr.capitalize() for attr in col]
        importance = pd.DataFrame(importance, index=col).sort_values(by='Importance', ascending=False)[:5]
        sns.barplot(x=importance['Importance'], y=importance.index, ax=ax[i])
        ax[i].set_title(f'{crime_type[i]}', fontsize=16)
    plt.savefig('fig/importance.png', bbox_inches='tight', dpi=200)
    plt.show()
