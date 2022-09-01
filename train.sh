# python train.py --type='weapons' --baseline='kde' --record
# python train.py --type='weapons' --baseline='kde' --no-stand --record
# python train.py --type='weapons' --baseline='imd' --record
# python train.py --type='weapons' --baseline='imd' --no-stand --record
# python train.py --type='weapons' --baseline='mob' --record
# python train.py --type='weapons' --baseline='mob' --no-stand --record
# python train.py --type='weapons' --baseline='no-kde' --record
# python train.py --type='weapons' --baseline='no-kde' --no-stand --record
# python train.py --type='weapons' --baseline='no-mob' --record
# python train.py --type='weapons' --baseline='no-mob' --no-stand --record
# python train.py --type='weapons' --baseline='no-imd' --record
# python train.py --type='weapons' --baseline='no-imd' --no-stand --record
# python train.py --type='weapons' --baseline='full' --record
# python train.py --type='weapons' --baseline='full' --no-stand --record
# python train.py --type='drugs' --baseline='kde' --record
# python train.py --type='drugs' --baseline='kde' --no-stand --record
# python train.py --type='drugs' --baseline='imd' --record
# python train.py --type='drugs' --baseline='imd' --no-stand --record
# python train.py --type='drugs' --baseline='mob' --record
# python train.py --type='drugs' --baseline='mob' --no-stand --record
# python train.py --type='drugs' --baseline='no-kde' --record
# python train.py --type='drugs' --baseline='no-kde' --no-stand --record
# python train.py --type='drugs' --baseline='no-mob' --record
# python train.py --type='drugs' --baseline='no-mob' --no-stand --record
# python train.py --type='drugs' --baseline='no-imd' --record
# python train.py --type='drugs' --baseline='no-imd' --no-stand --record
# python train.py --type='drugs' --baseline='full' --record
# python train.py --type='drugs' --baseline='full' --no-stand --record
# python train.py --type='violence' --baseline='kde' --record
# python train.py --type='violence' --baseline='kde' --no-stand --record
# python train.py --type='violence' --baseline='imd' --record
# python train.py --type='violence' --baseline='imd' --no-stand --record
# python train.py --type='violence' --baseline='mob' --record
# python train.py --type='violence' --baseline='mob' --no-stand --record --no-svm
# python train.py --type='violence' --baseline='no-kde' --record --no-svm
# python train.py --type='violence' --baseline='no-kde' --no-stand --record --no-svm
# python train.py --type='violence' --baseline='no-mob' --record --no-svm
# python train.py --type='violence' --baseline='no-mob' --no-stand --record --no-svm
# python train.py --type='violence' --baseline='no-imd' --record --no-svm
# python train.py --type='violence' --baseline='no-imd' --no-stand --record --no-svm
# python train.py --type='violence' --baseline='full' --record --no-svm
# python train.py --type='violence' --baseline='full' --no-stand --record --no-svm

python train.py --type='violence' --baseline='no-kde' --no-logistic --no-bayes --no-forest --no-knn --record
python train.py --type='violence' --baseline='no-mob' --no-logistic --no-bayes --no-forest --no-knn --record
python train.py --type='violence' --baseline='no-imd' --no-logistic --no-bayes --no-forest --no-knn --record
python train.py --type='violence' --baseline='full' --no-logistic --no-bayes --no-forest --no-knn --record