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

#load the training projects
projects = ["fabric8","tomcat","JGroups","spring-integration","camel","brackets","nova","neutron","BroadleafCommerce","npm",]
target = 3
cp = [2,4,1,0]

df_all = pd.read_csv("datasets/all_projects_with_dates.csv",sep=";")

df_cp = pd.DataFrame(columns=["author_date_unix_timestamp","fix","contains_bug","ns","nd","nf","entrophy","la","ld","lt","ndev","age","nuc","exp","rexp","sexp","dataset","days_to_first_fix","proj_target"])
for n in range(10):
    if(n in cp):
        d = df_all.loc[df_all['dataset'] == projects[n]]
        d["proj_target"] = False
        df_cp = pd.concat([df_cp,d])

    if(n == target):
        d = df_all.loc[df_all['dataset'] == projects[n]]
        d["proj_target"] = True
        df_cp = pd.concat([df_cp, d])



df_cp.sort_values(by=['author_date_unix_timestamp'], inplace=True)
df_cp = df_cp.drop(df_cp[df_cp.days_to_first_fix < 0].index)
df_cp.reset_index(drop=True, inplace=True)



y_pred = []
y_true = []

correct_def, n_defect = 0,0
correct_clean, n_clean = 0,0

previous_date = None
predicted_instances = 0
trained_instances = 0

initial_time = datetime.now()

orb = ORB(wt=45,l0=6,l1=8,threshold=0.5,n=30,m=3)

ct_predictions = 0

for i, inst in tqdm(df_cp.iterrows(), total=df_cp.shape[0]):

    predicted_instances += 1
    d = inst.to_dict()

    predict_date = datetime.utcfromtimestamp(int(d["author_date_unix_timestamp"]))

    d["target"] = d["contains_bug"]
    d["commit_date"] = predict_date

    #create a flag to determine tf the instance must be computed for the performance
    is_proj_target = d["proj_target"]
    del d["proj_target"]

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
    if(is_proj_target):
        orb.update_online_perf(index=i, y_hat=y_pred[-1], y=d["target"], pred_date=predict_date)
        ct_predictions+=1
        # print(ct_predictions)

    # if(y_pred[-1] == 0 and not d["target"]):
    #     correct_clean += 1
    #
    # if (y_pred[-1] == 1 and d["target"]):
    #     correct_def += 1

    # if(d["target"]):
    #     n_defect += 1
    # else:
    #     n_clean +=1

    #important in order to compute the ORB average predictions rate
    if(len(y_pred) <n):
        orb.last_n_pred = y_pred.copy()
    else:
        orb.last_n_pred = y_pred[-n:]

    # if(d["target"]):
    #     y_true.append(1)
    # else:
    #     y_true.append(0)

    orb.store_training_inst(d, i)

    #given a date, check the instances whose the labels are available and then train the model on them
    # first_time = datetime.now()
    orb.train_on_available_inst(orb.get_available_tr_inst(predict_date))
    # later_time = datetime.now()
    # difference = later_time - first_time
    # sec_train += difference.seconds + (difference.microseconds/1000000)

    if (previous_date != predict_date):
        previous_date = predict_date

print("media rec 0: ",mean(orb.df_preq_perf["rec_0"]))
print("media rec 1: ",mean(orb.df_preq_perf["rec_1"]))
print("media gmean: ",mean(orb.df_preq_perf["gmean"]))

orb.df_preq_perf.to_csv("results/"+projects[target]+"_"+f'{datetime.now():%Y-%m-%d_%H-%M-%S%z}'+".csv",sep=";")


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