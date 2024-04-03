from river import metrics
import pandas as pd
from datetime import datetime
import math
from orb_online_perf import ORB
from statistics import mean
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

sec_predict = 0
sec_train = 0


df_tr = pd.read_csv("neutron.csv",sep=";")
df_tr.sort_values(by=['author_date_unix_timestamp'], inplace=True)
df_tr = df_tr.drop(df_tr[df_tr.days_to_first_fix < 0].index)
df_tr.reset_index(drop=True, inplace=True)

y_pred = []
y_true = []

correct_def, n_defect = 0,0
correct_clean, n_clean = 0,0

previous_date = None
predicted_instances = 0
trained_instances = 0

initial_time = datetime.now()

orb = ORB(wt=90,l0=10,l1=12,threshold=0.3,n=100,m=3)

for i, inst in tqdm(df_tr.iterrows(), total=df_tr.shape[0]):

    predicted_instances += 1
    d = inst.to_dict()

    predict_date = datetime.utcfromtimestamp(int(d["author_date_unix_timestamp"]))

    d["target"] = d["contains_bug"]
    d["commit_date"] = predict_date

    #instnce to be predicted
    com = d.copy()
    del com["target"]
    del com["author_date_unix_timestamp"]
    del com["days_to_first_fix"]
    del com["contains_bug"]
    del com["dataset"]
    del com["commit_date"]

    # Test the current model on the new "unobserved" sample
    # first_time = datetime.now()
    c = orb.model.predict_one(com)
    # later_time = datetime.now()
    # difference = later_time - first_time
    # sec_predict += difference.seconds + (difference.microseconds / 1000000)

    if(c is None):
        y_pred.append(0)
    else:
        y_pred.append(c)

    #update the stored orb performance
    orb.update_online_perf(index=i, y_hat=y_pred[-1], y=d["target"], pred_date=predict_date)

    if(y_pred[-1] == 0 and not d["target"]):
        correct_clean += 1

    if (y_pred[-1] == 1 and d["target"]):
        correct_def += 1

    if(d["target"]):
        n_defect += 1
    else:
        n_clean +=1

    #important in order to compute the ORB average predictions rate
    if(len(y_pred) <100):
        orb.last_n_pred = y_pred.copy()
    else:
        orb.last_n_pred = y_pred[-100:]

    if(d["target"]):
        y_true.append(1)
    else:
        y_true.append(0)

    orb.store_training_inst(d, i)

    #given a date, check the instances whose the labels are available and then train the model on them
    # first_time = datetime.now()
    orb.train_on_available_inst(orb.get_available_tr_inst(predict_date))
    # later_time = datetime.now()
    # difference = later_time - first_time
    # sec_train += difference.seconds + (difference.microseconds/1000000)

    if (previous_date != predict_date):
        previous_date = predict_date
#
# metric = metrics.ROCAUC()
#
# for yt, yp in zip(y_true, y_pred):
#     metric = metric.update(yt, yp)
#

print("media rec 0: ",mean(orb.df_preq_perf["rec_0"]))
print("media rec 1: ",mean(orb.df_preq_perf["rec_1"]))
print("media gmean: ",mean(orb.df_preq_perf["gmean"]))

# print("test time: ",sec_predict)
# print("training time: ",sec_train)

final_time = datetime.now()
difference = final_time - initial_time
tot_time = difference.seconds + (difference.microseconds / 1000000)
print("total time: ",tot_time)

# print(metric)
#
# print(orb.get_perceived_performance())
#
# print("recall 0:", (correct_clean / n_clean))
# print("recall 1:", (correct_def / n_defect))