#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/4/26
import random
import pandas as pd


def down_sampling(infile, outfile, keep_ratio=0.15):
    with open(outfile, 'w') as fo:
        for line in open(infile):
            if line[-2] == '0':
                if random.random() > keep_ratio:  # remove
                    continue
            fo.write(line)


def weighted_score(w1=0.5, w2=0.5):
    """ensemble different model results"""
    df1 = pd.read_csv('submit/gbm_0.9750.csv')
    df2 = pd.read_csv('submit/sub_it5.csv')
    df = pd.DataFrame()
    df['click_id'] = df1['click_id'].astype('int')

    mean1 = df1.mean()[1]
    mean2 = df2.mean()[1]
    if mean1 > mean2:
        ratio = mean1 / float(mean2)
        df1['is_attributed'] = w1 * df1['is_attributed'] / float(ratio) + w2 * df2['is_attributed']
    else:
        ratio = mean2 / float(mean1)
        df1['is_attributed'] = w1 * df1['is_attributed'] + w2 * df2['is_attributed'] / float(ratio)

    df1.to_csv('submit/merge.csv', float_format='%.8f', index=False)


if __name__ == '__main__':
    weighted_score(0.4, 0.6)
    #down_sampling('/Volumes/Transcend/Safari/mnt-2/ssd/kaggle-talkingdata2/competition_files/train.csv', 'train_sample.csv')
