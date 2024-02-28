from river import ensemble
from river import evaluate
from river.tree import HoeffdingTreeClassifier
from river import metrics
import pandas as pd
from datetime import datetime, timedelta
import math
from statistics import mean
import numpy as np

class_size = [0.5,0.5]
theta = 0.999
threshold = 0.4
l1 = 12
l0 = 10
m = 1.5
wt = 60
n = 3

df_tr = pd.read_csv("neutron.csv",sep=";")
df_tr.sort_values(by=['author_date_unix_timestamp'], inplace=True)
df_tr = df_tr.drop(df_tr[df_tr.days_to_first_fix < 0].index)
predicted_instances = 0
trained_instances = 0
dic_ocurr_clean_inst = {}

last_n_pred = []

#model = HoeffdingTreeClassifier(grace_period=200)
model = ensemble.BaggingClassifier(model=(HoeffdingTreeClassifier(grace_period=200)),n_models=20,seed=12)

current_date = None
list_predictions = []
size_list_pred = 100

df_pool_tr = pd.DataFrame(columns=['fix', 'ns', 'nd', 'nf', 'entrophy', 'la', 'ld', 'lt', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp', 'target', 'train_date'])

def get_lambda(target):
    maj_class = 0

    if(class_size[1] > class_size[0]):
        maj_class = 1

    lamb = class_size[maj_class]/class_size[target]

    return lamb

def get_obf_pred_avg():

    avg = mean(last_n_pred)

    obf = [1,1]

    #boost class1
    if(avg < threshold):

        avg = abs(avg-threshold)
        obf[1] = (((m ** (avg*10))-1)/ ((m**(threshold*10))-1))*l1 + 1

    else:
        # boost class 0
        obf[0] = (((m ** (avg * 10)) - (m**(threshold * 10)))
                        / ((m ** 10) - (m**(threshold * 10)))) * l0 + 1
        #obf[0] = obf[0] * -1

    return obf


def update_class_size(target):
    global class_size

    for idx in [0,1]:

        class_size[idx] = theta * class_size[idx]

        if (target == idx):
            class_size[idx] += (1 - theta)

def store_training_inst(orig):
    global df_pool_tr

    inst = orig.copy()

    if(inst["target"] == False):

        inst["train_date"] = inst["commit_date"].date() + timedelta(days=wt)
        del inst["author_date_unix_timestamp"]
        del inst["contains_bug"]
        del inst["days_to_first_fix"]
        del inst["dataset"]
        del inst["commit_date"]

        for k in inst.keys():
            inst[k] = [inst[k]]

        df_aux = pd.DataFrame(inst)
        df_pool_tr = pd.concat([df_aux,df_pool_tr])

    else:

        commit_date = inst["commit_date"].date()
        days_to_fix = inst["days_to_first_fix"]
        inst["train_date"] = inst["commit_date"].date() + timedelta(days=math.ceil(inst["days_to_first_fix"]))
        del inst["author_date_unix_timestamp"]
        del inst["contains_bug"]
        del inst["days_to_first_fix"]
        del inst["dataset"]
        del inst["commit_date"]

        for k in inst.keys():
            inst[k] = [inst[k]]

        df_aux = pd.DataFrame(inst)
        df_pool_tr = pd.concat([df_aux, df_pool_tr],ignore_index=True)

        if(days_to_fix > wt):
            inst_noise = inst.copy()
            inst_noise["train_date"] = commit_date + timedelta(days=wt)
            inst_noise["target"] = False

            df_aux = pd.DataFrame(inst_noise)
            df_pool_tr = pd.concat([df_aux, df_pool_tr])

def get_available_tr_inst(cur_date):
    global df_pool_tr
    global trained_instances

    df_pool_tr.sort_values(by=['train_date'],inplace=True)
    df_pool_tr.sort_values(by=['train_date', "target"], inplace=True, ascending=[True, True])
    df_available = df_pool_tr.loc[df_pool_tr["train_date"] <= cur_date.date()]


    #add for training a defect inducing instance only if a clean instance is also trained to avoid
    #overemphazise the defect-inducing class
    l_training_idx = []
    while True:
        #search for a clean instance
        added0 = False
        added1 = False
        for idx,r in df_available.iterrows():
            if (r["target"] == False and idx not in l_training_idx):
                l_training_idx.append(idx)
                added0 = True
                break

        for idx, r in df_available.iterrows():
            if(r["target"] == True and added0 and idx not in l_training_idx):
                l_training_idx.append(idx)
                added0 = False
                added1 = True
                break

        if(not added0 and not added1):
            break

    df_return = df_available.loc[l_training_idx]
    df_pool_tr.drop(l_training_idx,inplace=True)

    return df_return

def train_on_available_inst(df_avail):

    for idx, inst in df_avail.iterrows():

        target = 0
        if (inst["target"]):
            target = 1

        update_class_size(target)

        x = inst.copy()
        del x["target"]
        del x["train_date"]

        #update the dictionary that is going to be used to check potential noisy defect-inducing instances
        if(target == 0):
            s = ""
            for k in x.keys():
                s += str(x[k]) + "-"

            if(dic_ocurr_clean_inst.__contains__(s)):
                dic_ocurr_clean_inst[s] = dic_ocurr_clean_inst[s] + 1
            else:
                dic_ocurr_clean_inst[s] = 1
        else:
            #skip the trainig instance if it is defect inducing and has been also very used to train as clean
            #the noisy mechanism of orb
            s = ""
            for k in x.keys():
                s += str(x[k]) + "-"
            if (dic_ocurr_clean_inst.__contains__(s)):
                if(dic_ocurr_clean_inst[s] > n):
                    return





        for classifier in model:
            weight_k = np.random.poisson(lam=(get_lambda(target)), size=(1))
            if (weight_k[0] < 1):
                weight_k = 1
            else:
                weight_k = weight_k[0]

            obf = get_obf_pred_avg()[target]
            weight_obf = weight_k * obf

            classifier.learn_one(x, target, sample_weight=weight_obf)

y_pred = []
y_true = []

acerto_def, n_defect = 0,0
acerto_clean, n_clean = 0,0

previous_date = None
for i, inst in df_tr.iterrows():

    predicted_instances += 1
    d = inst.to_dict()

    predict_date = datetime.utcfromtimestamp(int(d["author_date_unix_timestamp"]))
    d["target"] = d["contains_bug"]
    d["commit_date"] = predict_date

    #predict instance
    com = d.copy()
    del com["target"]
    del com["author_date_unix_timestamp"]
    del com["days_to_first_fix"]
    del com["contains_bug"]
    del com["dataset"]
    del com["commit_date"]

    # Test the current model on the new "unobserved" sample
    c = model.predict_one(com)

    #c = model.predict_one(com)
    if(c is None):
        y_pred.append(0)
    else:
        y_pred.append(c)

    if(y_pred[-1] == 0 and not d["target"]):
        acerto_clean += 1

    if (y_pred[-1] == 1 and d["target"]):
        acerto_def += 1

    if(d["target"]):
        n_defect += 1
    else:
        n_clean +=1

    #important to compute the average predictions rate
    if(len(y_pred) <100):
        last_n_pred = y_pred.copy()
    else:
        last_n_pred = y_pred[-100:]

    if(d["target"]):
        y_true.append(1)
    else:
        y_true.append(0)

    store_training_inst(d)

    #train on available instances
    if(previous_date == None or predict_date.date() != previous_date.date):
        train_on_available_inst(get_available_tr_inst(predict_date))

    if(predicted_instances%1000 == 0):
        print(predicted_instances)

    if (previous_date != predict_date):
        previous_date = predict_date

metric = metrics.ROCAUC()

for yt, yp in zip(y_true, y_pred):
    metric = metric.update(yt, yp)

print(metric)

print("recall 0:",(acerto_clean/n_clean))
print("recall 1:",(acerto_def/n_defect))