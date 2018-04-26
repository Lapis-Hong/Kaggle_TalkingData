"""This version result in auc: 0.9778"""
import time
import gc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import lightgbm as lgb

# train_path = '/Volumes/Transcend/Safari/mnt-2/ssd/kaggle-talkingdata2/competition_files/train.csv'
train_path = 'train_sample.csv'
test_path = '/Volumes/Transcend/Safari/test.csv'

predictors = []


def do_next_prev_click(df, agg_suffix, agg_type='float32'):
    """ Extracting next click feature"""
    print('Extracting new features...')
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('int8')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('int8')
    df['minute'] = pd.to_datetime(df.click_time).dt.minute.astype('int8')
    predictors.append('minute')
    df['second'] = pd.to_datetime(df.click_time).dt.second.astype('int8')
    predictors.append('second')
    print("\nExtracting {} time calculation features...\n".format(agg_suffix))
    
    GROUP_BY_NEXT_CLICKS = [
    # V1
    # {'groupby': ['ip']},
    # {'groupby': ['ip', 'app']},
    # {'groupby': ['ip', 'channel']},
    # {'groupby': ['ip', 'os']},
    # V3
    {'groupby': ['ip', 'app', 'device', 'os', 'channel']},
    {'groupby': ['ip', 'os', 'device']},
    {'groupby': ['ip', 'os', 'device', 'app']}
    ]
    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:
        # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']), agg_suffix)
    
        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']
        # Run calculation
        print("Grouping by {}, and saving time to {} in: {}".format(spec['groupby'], agg_suffix, new_feature))
        if agg_suffix == "nextClick":
            df[new_feature] = (df[all_features].groupby(spec[
                'groupby']).click_time.shift(-1) - df.click_time).dt.seconds.astype(agg_type)
        elif agg_suffix == "prevClick":
            df[new_feature] = (df.click_time - df[all_features].groupby(spec[
                'groupby']).click_time.shift(+1)).dt.seconds.astype(agg_type)
        predictors.append(new_feature)
        gc.collect()
    return df


def do_count(df, group_cols, agg_type='uint32', show_max=False, show_agg=True):
    """ Extract count feature by aggregating different cols"""
    agg_name ='{}count'.format('_'.join(group_cols))
    if show_agg:
        print("\nAggregating by {}... and saved in {}".format(group_cols, agg_name))
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
    gc.collect()
    return df


def do_countuniq(df, group_cols, counted, agg_type='uint32', show_max=False, show_agg=True):
    """ Extract unique count feature from different cols"""
    agg_name = '{}_by_{}_countuniq'.format(('_'.join(group_cols)), (counted))
    if show_agg:
        print("\nCounting unqiue {} by {}... and saved in {}".format(counted, group_cols, agg_name))
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
    gc.collect()
    return df


def do_cumcount(df, group_cols, counted, agg_type='uint32', show_max=False, show_agg=True):
    """ Extract cumulative count feature from different cols"""
    agg_name = '{}_by_{}_cumcount'.format('_'.join(group_cols), counted)
    if show_agg:
        print("\nCumulative count by {} and saved in {}".format(group_cols, agg_name))
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name] = gp.values
    del gp
    if show_max:
        print(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
    gc.collect()
    return df


def do_mean(df, group_cols, counted, agg_type='float32', show_max=False, show_agg=True):
    """Extract mean feature from different cols"""
    agg_name = '{}_by_{}_mean'.format('_'.join(group_cols), counted)
    if show_agg:
        print("\nCalculating mean of {} by {}... and saved in {}".format(counted, group_cols, agg_name))
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].mean().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
    gc.collect()
    return df


def do_var(df, group_cols, counted, agg_type='float32', show_max=False, show_agg=True):
    """Extract variance feature from different cols"""
    agg_name = '{}_by_{}_var'.format(('_'.join(group_cols)), (counted))
    if show_agg:
        print("\nCalculating variance of {} by {}... and saved in {}".format(counted, group_cols, agg_name))
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
    gc.collect()
    return df


def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                 feval=None, early_stopping_rounds=50, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric': metrics,
        'learning_rate': 0.05,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0.99,  # L1 regularization term on weights
        'reg_lambda': 0.9,  # L2 regularization term on weights
        'nthread': 8,
        'verbose': 0,
    }

    lgb_params.update(params)

    print("preparing validation datasets")
    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    evals_results = {}

    bst1 = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgtrain, xgvalid], 
                     valid_names=['train', 'valid'],
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=feval)

    print("\nModel Report")
    print("bst1.best_iteration: ", bst1.best_iteration)
    print(metrics+":", evals_results['valid'][metrics][bst1.best_iteration-1])

    return bst1, bst1.best_iteration


def DO(fileno=0, frm=None, to=None, val_size=2500000, load_origin=True, save=True):
    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint8',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32',
            }

    if load_origin:
        print('loading origin train data from {} to {} ...'.format(frm, to))
        if not frm or not to:
            train_df = pd.read_csv(train_path, parse_dates=['click_time'], dtype=dtypes,
                                   usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'])
        else:
            train_df = pd.read_csv(train_path, parse_dates=['click_time'], skiprows=range(1, frm), nrows=to - frm,
                                   dtype=dtypes,
                                   usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'])
            # train_df = train_df.sample(frac=1, replace=False)  # need or not ?

        print('loading origin test data...')
        test_df = pd.read_csv(test_path, parse_dates=['click_time'], dtype=dtypes,
                                  usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'])

        len_train = len(train_df)
        train_df = train_df.append(test_df)
        del test_df

        gc.collect()
        train_df = do_next_prev_click(train_df, agg_suffix='nextClick', agg_type='float32')
        gc.collect()
        # train_df = do_next_prev_Click( train_df,agg_suffix='prevClick', agg_type='float32'  ); gc.collect()  ## Removed temporarily due RAM sortage.
        train_df = do_countuniq(train_df, ['ip'], 'channel')
        gc.collect()
        train_df = do_countuniq(train_df, ['ip', 'device', 'os'], 'app')
        gc.collect()
        train_df = do_countuniq(train_df, ['ip', 'day'], 'hour')
        gc.collect()
        train_df = do_countuniq(train_df, ['ip'], 'app')
        gc.collect()
        train_df = do_countuniq(train_df, ['ip', 'app'], 'os')
        gc.collect()
        train_df = do_countuniq(train_df, ['ip'], 'device')
        gc.collect()
        train_df = do_countuniq(train_df, ['app'], 'channel')
        gc.collect()
        train_df = do_cumcount(train_df, ['ip'], 'os')
        gc.collect()
        train_df = do_cumcount(train_df, ['ip', 'device', 'os'], 'app')
        gc.collect()
        train_df = do_count(train_df, ['ip', 'day', 'hour'])
        gc.collect()
        train_df = do_count(train_df, ['ip', 'app'])
        gc.collect()
        train_df = do_count(train_df, ['ip', 'app', 'os'])
        gc.collect()
        train_df = do_var(train_df, ['ip', 'day', 'channel'], 'hour')
        gc.collect()
        train_df = do_var(train_df, ['ip', 'app', 'os'], 'hour')
        gc.collect()
        train_df = do_var(train_df, ['ip', 'app', 'channel'], 'day')
        gc.collect()
        train_df = do_mean(train_df, ['ip', 'app', 'channel'], 'hour')
        gc.collect()
        if save:
            print(train_df.head(5))
            train_df.to_csv('train_with_feature.csv', index=False)

    else:
        print('loading preprocess train data...')
        train_df = pd.read_csv('train_with_feature.csv', dtype=dtypes)
        # train_df = train_df.sample(frac=1, replace=False)  # need or not ?

    target = 'is_attributed'
    features = ['app', 'device', 'os', 'channel', 'hour', 'day', 'minute', 'second']
    for feature in features:
        if feature not in predictors:
            predictors.append(feature)
    categorical = ['app', 'device', 'os', 'channel', 'hour', 'day', 'minute', 'second']
    print('\n\nPredictors...\n\n', sorted(predictors))

    test_df = train_df[len_train:]
    val_df = train_df[(len_train-val_size):len_train]
    train_df = train_df[:(len_train-val_size)]

    print("\ntrain size: {}".format(len(train_df)))
    print("\nvalid size: {}".format(len(val_df)))
    print("\ntest size : {}".format(len(test_df)))

    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')

    print("Training...")
    start_time = time.time()

    params = {
        'learning_rate': 0.20,
        #'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 7,  # 2^max_depth - 1  31
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 200,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 200,  # Number of bucketed bin for feature values, 100
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight': 100  # because training data is extremely unbalanced(400:1), 300~400
    }
    (bst, best_iteration) = lgb_modelfit_nocv(params,
                            train_df, 
                            val_df, 
                            predictors, 
                            target, 
                            objective='binary', 
                            metrics='auc',
                            early_stopping_rounds=30, 
                            verbose_eval=True, 
                            num_boost_round=1000, 
                            categorical_features=categorical)

    print('[{}]: model training time'.format(time.time() - start_time))
    del train_df
    del val_df
    gc.collect()

    # ax = lgb.plot_importance(bst, max_num_features=100)
    # plt.show()
    # plt.savefig('foo.png')

    print("Predicting...")
    sub['is_attributed'] = bst.predict(test_df[predictors], num_iteration=best_iteration)

    sub.to_csv('gbm{}.csv'.format(fileno), index=False, float_format='%.9f')
    print("done...")
    return sub

####### Chunk size defining and final run  ############

nrows = 184903891-1
frm = nrows-40000000
nchunk = 40000000
to = frm+nchunk

sub = DO(fileno=1, frm=frm, to=to, val_size=2500000, load_origin=True, save=True)
