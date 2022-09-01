import os
import glob
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import datetime
from dateutil import relativedelta
from utils import get_logger, load_dataset
import argparse


def append_score(res, items):
    for i in ['acc', 'f1', 'auc']:
        info = items[:4]
        if i == 'acc':
            info.extend(['acc', items[4]])
            res.append(info)
        elif i == 'f1':
            info.extend(['f1', items[5]])
            res.append(info)
        elif i == 'auc':
            info.extend(['auc', items[6]])
            res.append(info)
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stand', dest='stand', action='store_true')
    parser.add_argument('--no-stand', dest='stand', action='store_false')
    parser.add_argument('--baseline', type=str, default='kde')
    parser.add_argument('--type', type=str, default='violence')
    parser.add_argument('--logistic', dest='logistic', action='store_true')
    parser.add_argument('--no-logistic', dest='logistic', action='store_false')
    parser.add_argument('--bayes', dest='bayes', action='store_true')
    parser.add_argument('--no-bayes', dest='bayes', action='store_false')
    parser.add_argument('--forest', dest='forest', action='store_true')
    parser.add_argument('--no-forest', dest='forest', action='store_false')
    parser.add_argument('--knn', dest='knn', action='store_true')
    parser.add_argument('--no-knn', dest='knn', action='store_false')
    parser.add_argument('--svm', dest='svm', action='store_true')
    parser.add_argument('--no-svm', dest='svm', action='store_false')
    parser.add_argument('--record', dest='record', action='store_true')
    parser.set_defaults(stand=True, logistic=True, bayes=True, forest=True, knn=True, svm=True)
    args = parser.parse_args()
    logger = get_logger('log/')

    crime_type = {'drugs': 'Drugs', 'weapons': 'Possession of weapons', 'violence': 'Violence and sexual offences'}
    data = load_dataset(crime_type[args.type])
    logger.info(crime_type[args.type])
    print(data.isnull().sum())

    # training & validation
    base_month = datetime.datetime(2021, 1, 1)
    split_month = datetime.datetime(2021, 12, 1)
    test_month = split_month + relativedelta.relativedelta(months=1)

    results = []
    while test_month < datetime.datetime(2022, 5, 1):
        train = data.loc[(data['Month'] >= base_month) & (data['Month'] <= split_month)]
        test = data.loc[data['Month'] == test_month]
        # train['Month_int'] = train['Month'].apply(lambda x: int(x.month))
        # test['Month_int'] = test['Month'].apply(lambda x: int(x.month))

        # dataframe to numpy
        train = train.drop(columns=['LSOACD', 'LSOANM', 'LADCD', 'LADNM', 'Month'])
        test = test.drop(columns=['LSOACD', 'LSOANM', 'LADCD', 'LADNM', 'Month'])
        
        # baseline
        if args.baseline == 'imd':
            col = ['Income', 'Employment', 'Education', 'Health', 'Crime', 'Barriers', 'Living Environment']
            logger.info(f'columns {col}')
            X_train = train[col].values
            X_test = test[col].values
        elif args.baseline == 'mob':
            col = ['retail', 'grocery', 'parks', 'stations', 'workplaces', 'residential']
            logger.info(f'columns {col}')
            X_train = train[col].values
            X_test = test[col].values
        elif args.baseline == 'kde':
            col = ['density']
            logger.info(f'columns {col}')
            X_train = train[col].values
            X_test = test[col].values
        elif args.baseline == 'no-mob':
            logger.info('no-mob')
            col = ['density', 'Income', 'Employment', 'Education', 'Health', 'Crime', 'Barriers', 'Living Environment']
            logger.info(f'columns {col}')
            X_train = train[col].values
            X_test = test[col].values
        elif args.baseline == 'no-imd':
            logger.info('no-imd')
            col = ['density', 'retail', 'grocery', 'parks', 'stations', 'workplaces', 'residential']
            logger.info(f'columns {col}')
            X_train = train[col].values
            X_test = test[col].values
        elif args.baseline == 'no-kde':
            logger.info('no-kde')
            X_train = train.drop(columns=['label', 'loc_x', 'loc_y', 'density']).values
            X_test = test.drop(columns=['label', 'loc_x', 'loc_y', 'density']).values
        else:
            logger.info('full attributes')
            X_train = train.drop(columns=['label', 'loc_x', 'loc_y']).values
            X_test = test.drop(columns=['label', 'loc_x', 'loc_y']).values
        
        y_train = train['label'].values
        y_test = test['label'].values

        # standardization
        if args.stand:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)

        # logistic regression
        if args.logistic:
            clf = LogisticRegressionCV(cv=5, random_state=0, max_iter=150, scoring='f1_weighted', n_jobs=-1)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            
            report = classification_report(y_test, pred, output_dict=True)
            auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
            results = append_score(results, ['LR', crime_type[args.type], test_month.month, args.baseline, report['accuracy'], report['weighted avg']['f1-score'], auc])
            report = classification_report(y_test, pred)
            print(f'model: Logistic Regression\ntest month: {test_month} \n{report}\nauc: {auc}')
            logger.info(f'model: Logistic Regression\ntest month: {test_month} \n{report}\nauc: {auc}')

        # naive bayes
        if args.bayes:
            clf = GaussianNB()
            param_grid = {'var_smoothing': np.logspace(0, -9, num=100)}
            grid_clf = GridSearchCV(clf, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
            grid_clf.fit(X_train, y_train)
            print(grid_clf.best_params_)
            pred = grid_clf.predict(X_test)

            report = classification_report(y_test, pred, output_dict=True)
            auc = roc_auc_score(y_test, grid_clf.predict_proba(X_test)[:, 1])
            append_score(results, ['NB', crime_type[args.type], test_month.month, args.baseline, report['accuracy'], report['weighted avg']['f1-score'], auc])
            report = classification_report(y_test, pred)
            print(f'model: Naive Bayes\ntest month: {test_month} \n{report}\nauc: {auc}')
            logger.info(f'model: Naive Bayes\ntest month: {test_month} \n{report}\nauc: {auc}')

        # random forest
        if args.forest:
            # clf = RandomForestClassifier(min_samples_leaf=5, max_samples=0.8, n_jobs=-1) # drugs
            clf = RandomForestClassifier(criterion='entropy', max_samples=0.8, n_jobs=-1)
            param_grid = {
                # 'n_estimators': [100]
                'n_estimators': [100, 125, 150],
                # 'max_samples': [0.6, 0.8, None]
            }
            grid_clf = GridSearchCV(clf, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
            grid_clf.fit(X_train, y_train)
            print(grid_clf.best_params_)
            logger.info(grid_clf.best_params_)
            pred = grid_clf.predict(X_test)

            report = classification_report(y_test, pred, output_dict=True)
            auc = roc_auc_score(y_test, grid_clf.predict_proba(X_test)[:, 1])
            append_score(results, ['RF', crime_type[args.type], test_month.month, args.baseline, report['accuracy'], report['weighted avg']['f1-score'], auc])
            report = classification_report(y_test, pred)
            print(f'model: Random Forest\ntest month: {test_month} \n{report}\nauc: {auc}')
            logger.info(f'model: Random Forest\ntest month: {test_month} \n{report}\nauc: {auc}')

        # knn
        if args.knn:
            clf = KNeighborsClassifier(weights='distance', leaf_size=300, n_jobs=5)
            param_grid = {
                # 'n_neighbors' : [5, 10, 20]   # drugs
                'n_neighbors' : [3, 5, 7]
            }
            grid_clf = GridSearchCV(clf, param_grid, cv=5, scoring='f1_weighted', n_jobs=5)
            grid_clf.fit(X_train, y_train)
            print(grid_clf.best_params_)
            pred = grid_clf.predict(X_test)

            report = classification_report(y_test, pred, output_dict=True)
            auc = roc_auc_score(y_test, grid_clf.predict_proba(X_test)[:, 1])
            append_score(results, ['KNN', crime_type[args.type], test_month.month, args.baseline, report['accuracy'], report['weighted avg']['f1-score'], auc])
            report = classification_report(y_test, pred)
            print(f'model: KNN\ntest month: {test_month} \n{report}\nauc: {auc}')
            logger.info(f'model: KNN\ntest month: {test_month} \n{report}\nauc: {auc}')

        if args.svm:
            clf = SVC(kernel='rbf', gamma='scale', C=3, cache_size=1000)
            bagging_clf = BaggingClassifier(clf, n_estimators=5, max_samples=0.5, n_jobs=-1).fit(X_train, y_train)
            # bagging_clf = BaggingClassifier(clf, n_estimators=5, max_samples=0.7, n_jobs=-1).fit(X_train, y_train)
            pred = bagging_clf.predict(X_test)

            report = classification_report(y_test, pred, output_dict=True)
            auc = roc_auc_score(y_test, bagging_clf.predict_proba(X_test)[:, 1])
            append_score(results, ['SVM', crime_type[args.type], test_month.month, args.baseline, report['accuracy'], report['weighted avg']['f1-score'], auc])
            report = classification_report(y_test, pred)
            print(f'model: SVM\ntest month: {test_month} \n{report}\nauc: {auc}')
            logger.info(f'model: SVM\ntest month: {test_month} \n{report}\nauc: {auc}')

        base_month = base_month + relativedelta.relativedelta(months=1)
        split_month = split_month + relativedelta.relativedelta(months=1)
        test_month = test_month + relativedelta.relativedelta(months=1)

    results = pd.DataFrame(results, columns=['Model', 'type', 'month', 'baseline', 'measure', 'score'])
    if args.record:
        results.to_csv(f'./results/{args.type}/{args.type}_{args.baseline}_stand{args.stand}.csv', index=False)
        # results.to_csv(f'./results/{args.type}/{args.type}_{args.baseline}_stand{args.stand}_svm.csv', index=False)

