
# coding: utf-8

# In[16]:


# Import Library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().magic('matplotlib inline')
from keras.preprocessing.text import Tokenizer # for Dictionary set up
from keras.preprocessing import sequence # for data preprocess: work transfer to number list

# Import Data
## input data as CSV
#df = pd.read_csv("input\Multi Same Hold.csv", encoding='ISO-8859-15')

## Import Data as SQL
import pyodbc
import pymssql
import cx_Oracle

conn=pyodbc.connect(r'DRIVER={SQL Server Native Client 10.0};SERVER="";DATABASE="";UID="";PWD=""') 
#for security issue, no show sever/databass/UID/PWD at here

cursor = conn.cursor()

df = pd.read_sql("select t7.* from (select t6.lot_id,t6.repeat,t6.master_cate,t6.cate,t6.max_time,t6.min_time,wm_concat(t6.note) over(partition by t6.lot_id,t6.repeat,t6.master_cate,t6.cate,t6.max_time,t6.min_time order by t6.hold_time) as note,row_number() over (partition by t6.lot_id,t6.repeat,t6.master_cate,t6.cate,t6.max_time,t6.min_time order by t6.hold_time desc ) rank,LENGTH(t6.hold_note) length_holdnote from (select t5.lot_id,t5.repeat,t5.master_cate,t5.cate,t5.max_time,t5.min_time,t.stage,t.ope_no,t.hold_code,t.hold_user,t.release_user,t.hold_time,t.hold_note,t.release_memo,t.hold_user||'--'||t.hold_note||'--'||t.release_user||'--'||t.release_memo||'----' as notefrom (select t4.lot_id,t4.ope_no,case when t4.check2 like '%Data XferAMHS%' then 'Invalid transfer'else t4.master_cate end as master_cate,t4.hold_note ,max(t4.release_memo) release_memo,t4.cate,t4.rank repeat,t4.check2,max(t4.hold_time) max_time,min(t4.hold_time) min_timefrom (select t.*,t3.rankfrom (select t2.lot_id,t2.ope_no,t2.check2,max(rank) rankfrom (select t1.lot_id,t1.ope_no,t1.check2,row_number() over (partition by lot_id,ope_no,check2 order by hold_time ) rankfrom (select lot_id,ope_no,hold_time,lot_id||ope_no||hold_code||hold_user||release_user||hold_type||master_cate||cate||hold_note as check2 from MFGDEV.MFG_HOLD_ANA_EVER_HOLD_BT t where 1=1 and t.hold_time >=TO_DATE('2018/11/11 07:20:00','YYYY/MM/DD HH24:MI:SS') and t.hold_time <TO_DATE('2018/11/05 07:20:00','YYYY/MM/DD HH24:MI:SS') and t.hold_time <TO_DATE('2018/11/14 07:20:00','YYYY/MM/DD HH24:MI:SS') and t.tech like 'N07%' and t.part not like '%+RD%' order by 1,2,4,3) t1) t2 where t2.rank>1 group by t2.lot_id,t2.ope_no,t2.check2) t3,(select lot_id,stage,ope_no,hold_time,master_cate,cate,hold_note,release_memo,lot_id||ope_no||hold_code||hold_user||release_user||hold_type||master_cate||cate||hold_note as check2 from MFGDEV.MFG_HOLD_ANA_EVER_HOLD_BT) t where t3.lot_id=t.lot_id and t3.ope_no=t.ope_no and t3.check2=t.check2) t4 where 1=1 --t4.check2 not like '%Data XferAMHS%' and t4.Hold_note not like 'Transfer hold status from%' and t4.Hold_note not like 'Add HoldCode for Data Xfer%' and t4.ope_no not like '001.%' and t4.stage not like '%RWK%' group by t4.lot_id,t4.ope_no,case when t4.check2 like '%Data XferAMHS%' then 'Invalid transfer' else t4.master_cate end,t4.hold_note,t4.cate,t4.rank,t4.check2) t5,MFGDEV.MFG_HOLD_ANA_EVER_HOLD_BT t where t.lot_id=t5.lot_id --and t.ope_no=t5.ope_no and t.hold_time >=t5.min_time and t.hold_time <=t5.max_time and (select min(a.qty) qty from MFGDEV.MFG_HOLD_ANA_EVER_HOLD_BT a where a.lot_id=t5.lot_id and a.hold_time=t5.min_time) = (select min(a.qty) qty from MFGDEV.MFG_HOLD_ANA_EVER_HOLD_BT a where a.lot_id=t5.lot_id and a.hold_time=t5.max_time) and (t5.max_time-t5.min_time)*24/t5.repeat < 2 and t5.repeat <=10 and LENGTH(t5.hold_note) <=50 order by t5.lot_id,t5.max_time,t5.min_time,t.hold_time) t6) t7 where t7.rank=1", cursor)

all_texts = []
for i in range(922):
    all_texts.append(df.iloc[i, 6])
    
all_labels = []
for i in range(922):
    all_labels.append(df.iloc[i, 9])
    
# Data Normalization
import re
def rm_tags(text):
    re_tag = re.compile('<[^>]+>')     ### Remove html tag
    return re_tag.sub('', text)         

train_text = []
for i in range(df.shape[0]):
    train_text.append(rm_tags("".join(all_texts[i])))

# Tokenizer
token = Tokenizer(num_words=2000) #建立一個2000個字的字典
token.fit_on_texts(train_text)

# Data Transfer
x_train_seq = token.texts_to_sequences(train_text)

# Pad Sequences
x_train = sequence.pad_sequences(x_train_seq, maxlen=500)

# Data conversion of Label
y_train = np.array(df['Label'])

# Model training
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN

model_RNN = Sequential()
model_RNN.add(Embedding(input_dim=2000, input_length=500, output_dim=32))
model_RNN.add(Dropout(0.35))
model_RNN.add(SimpleRNN(units=16)) # 16層RNN
model_RNN.add(Dense(units=256,activation='relu')) #影藏層256層
model_RNN.add(Dropout(0.35))
model_RNN.add(Dense(units=1, activation='sigmoid'))
model_RNN.summary()

model_RNN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

train_history = model_RNN.fit(x_train, y_train, batch_size=100, epochs=10, verbose=2, validation_split=0.2) 

fontsize=20
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History', fontsize=fontsize)
    plt.ylabel(train, fontsize=fontsize)
    plt.xlabel('Epoch', fontsize=fontsize)
    plt.tick_params(axis='y',labelsize=20)
    plt.tick_params(axis='x',labelsize=20)
    plt.legend(['train','validation'], loc='upper left', fontsize=fontsize) #設定顯示標題於左上角
    plt.show()
    
plt.figure(figsize=(12,6))
show_train_history(train_history,'acc','val_acc')

