import os
import gc
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'

# path = '../input/'
train_path = '/Volumes/Transcend/Safari/mnt-2/ssd/kaggle-talkingdata2/competition_files/train.csv'
eval_path = '/Volumes/Transcend/Safari/test_supplement.csv'
test_path = '/Volumes/Transcend/Safari/test.csv'

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
print('load train data....')
train_df = pd.read_csv(train_path, dtype=dtypes, skiprows=range(1, 131886954),
                       usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'])
print('load test data....')
test_df = pd.read_csv(test_path, dtype=dtypes,
                      usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'])
len_train = len(train_df)
train_df = train_df.append(test_df)
del test_df
gc.collect()

print('add feature hour, day, wday....')
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
train_df['wday'] = pd.to_datetime(train_df.click_time).dt.dayofweek.astype('uint8')

print('grouping by ip-day-hour combination....')
gp = train_df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'qty'})
train_df = train_df.merge(gp, on=['ip', 'day', 'hour'], how='left')
del gp
gc.collect()

print('group by ip-app combination....')
gp = train_df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
train_df = train_df.merge(gp, on=['ip', 'app'], how='left')
del gp
gc.collect()

print('group by ip-app-os combination....')
gp = train_df[['ip', 'app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
train_df = train_df.merge(gp, on=['ip', 'app', 'os'], how='left')
del gp
gc.collect()

print("vars and data type....")
train_df['qty'] = train_df['qty'].astype('uint16')
train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')

print("label encoding....")
from sklearn.preprocessing import LabelEncoder
train_df[['app', 'device', 'os', 'channel', 'hour', 'day', 'wday']].apply(LabelEncoder().fit_transform)

print ('split train test, drop features...')
max_app = train_df['app'].max() + 1
max_ch = train_df['channel'].max() + 1
max_dev = train_df['device'].max() + 1
max_os = train_df['os'].max() + 1
max_h = train_df['hour'].max() + 1
max_d = train_df['day'].max() + 1
max_wd = train_df['wday'].max() + 1
max_qty = train_df['qty'].max() + 1
max_c1 = train_df['ip_app_count'].max() + 1
max_c2 = train_df['ip_app_os_count'].max() + 1

train_df = train_df[:len_train]
test_df = train_df[len_train:]
y_train = train_df['is_attributed'].values
train_df.drop(['click_id', 'click_time', 'ip', 'is_attributed'], 1, inplace=True)

print ('neural network....')
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, SpatialDropout1D, Conv1D
from keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import Adam


def get_keras_data(dataset):
    X = {
        'app': np.array(dataset.app),
        'ch': np.array(dataset.channel),
        'dev': np.array(dataset.device),
        'os': np.array(dataset.os),
        'h': np.array(dataset.hour),
        'd': np.array(dataset.day),
        'wd': np.array(dataset.wday),
        'qty': np.array(dataset.qty),
        'c1': np.array(dataset.ip_app_count),
        'c2': np.array(dataset.ip_app_os_count)
    }
    return X
train_df = get_keras_data(train_df)

emb_n = 50
dense_n = 1000
in_app = Input(shape=[1], name='app')
emb_app = Embedding(max_app, emb_n)(in_app)
in_ch = Input(shape=[1], name='ch')
emb_ch = Embedding(max_ch, emb_n)(in_ch)
in_dev = Input(shape=[1], name='dev')
emb_dev = Embedding(max_dev, emb_n)(in_dev)
in_os = Input(shape=[1], name='os')
emb_os = Embedding(max_os, emb_n)(in_os)
in_h = Input(shape=[1], name='h')
emb_h = Embedding(max_h, emb_n)(in_h) 
in_d = Input(shape=[1], name='d')
emb_d = Embedding(max_d, emb_n)(in_d) 
in_wd = Input(shape=[1], name='wd')
emb_wd = Embedding(max_wd, emb_n)(in_wd) 
in_qty = Input(shape=[1], name='qty')
emb_qty = Embedding(max_qty, emb_n)(in_qty) 
in_c1 = Input(shape=[1], name='c1')
emb_c1 = Embedding(max_c1, emb_n)(in_c1) 
in_c2 = Input(shape=[1], name='c2')
emb_c2 = Embedding(max_c2, emb_n)(in_c2) 
fe = concatenate([emb_app, emb_ch, emb_dev, emb_os, emb_h,
                 emb_d, emb_wd, emb_qty, emb_c1, emb_c2])
s_dout = SpatialDropout1D(0.2)(fe)
fl1 = Flatten()(s_dout)
conv = Conv1D(100, kernel_size=4, strides=1, padding='same')(s_dout)
fl2 = Flatten()(conv)
concat = concatenate([fl1, fl2])
x = Dropout(0.2)(Dense(dense_n, activation='relu')(concat))
x = Dropout(0.2)(Dense(dense_n, activation='relu')(x))
outp = Dense(1, activation='sigmoid')(x)
model = Model(inputs=[in_app, in_ch, in_dev, in_os, in_h, in_d, in_wd, in_qty, in_c1, in_c2], outputs=outp)

batch_size = 20000
epochs = 2
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(train_df) / batch_size) * epochs
lr_init, lr_fin = 0.001, 0.0001
lr_decay = exp_decay(lr_init, lr_fin, steps)
optimizer_adam = Adam(lr=0.001, decay=lr_decay)
model.compile(loss='binary_crossentropy', optimizer=optimizer_adam, metrics=['accuracy'])

model.summary()

class_weight = {0: .01, 1: .99}  # magic
model.fit(train_df, y_train, batch_size=batch_size, epochs=2, class_weight=class_weight, shuffle=True, verbose=2)
del train_df, y_train; gc.collect()
model.save_weights('imbalanced_data.h5')

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')
test_df.drop(['click_id', 'click_time', 'ip', 'is_attributed'], 1, inplace=True)
test_df = get_keras_data(test_df)

print("predicting....")
sub['is_attributed'] = model.predict(test_df, batch_size=batch_size, verbose=2)
del test_df
gc.collect()
print("writing....")
sub.to_csv('nn.csv', index=False)

