# plotting roc curve for random forest classifier
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_roc_curve
import datetime
from dateutil import relativedelta
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    fig, ax = plt.subplots(1,1, figsize=(8,6))
    sns.set_theme(style='whitegrid')
    data_path = f'./data/dataset/Drugs/train_test.csv'
    data = pd.read_csv(data_path)
    data['Month'] = pd.to_datetime(data['Month'], yearfirst=True)
    data = data.rename(columns={'Living Environment': 'Living Env.'})

    base_month = datetime.datetime(2021, 1, 1)
    split_month = datetime.datetime(2021, 12, 1)
    test_month = split_month + relativedelta.relativedelta(months=1)

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

    plot_roc_curve(clf, X_test, y_test, ax=ax, color='darkorange', lw=2)
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.set_title('ROC curve')
    plt.savefig('./fig/roc.png', bbox_inches='tight', dpi=200)
    plt.show()
