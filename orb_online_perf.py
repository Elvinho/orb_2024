from river import ensemble
from river.tree import HoeffdingTreeClassifier
from river import metrics
import pandas as pd
from datetime import datetime, timedelta
import math
from statistics import mean
import numpy as np

class ORB():

    #parameters
    threshold = 0.4
    l1 = 12
    l0 = 10
    m = 1.5
    wt = 60
    n = 3

    class_size = [0.5,0.5]

    theta = 0.999
    s_alpha_0 = 0
    n_alpha_0 = 0
    s_alpha_1 = 0
    n_alpha_1 = 0

    dic_ocurr_clean_inst = {}

    last_n_pred = []

    df_preq_perf = pd.DataFrame(columns=["date","y","rec_0","rec_1","gmean","perf_avail"])

    #model = HoeffdingTreeClassifier(grace_period=200)
    model = None

    current_date = None
    list_predictions = []
    df_online_perf = pd.DataFrame(columns=['index', 'y_hat', "y","perf_avail"])
    size_list_pred = 100

    df_pool_tr = pd.DataFrame(columns=['author_date_unix_timestamp','fix', 'ns', 'nd', 'nf', 'entrophy', 'la', 'ld', 'lt', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp', 'target', 'train_date'])

    def update_online_perf(self,index,y_hat,y, pred_date):
        # global df_online_perf
        #
        # if index not in list(self.df_online_perf["index"]):
        #     dic_aux = {"index":[int(index)],"y_hat":[y_hat],"y":[y]}
        #     df_aux = pd.DataFrame(dic_aux)
        #     self.df_online_perf = pd.concat([df_aux,self.df_online_perf])
        # else:
        #     self.df_online_perf.loc[self.df_online_perf['index'] == int(index), 'y'] = y

        #update the prequential errors
        last_rec_0 = 0
        last_rec_1 = 0

        if len(self.df_preq_perf) != 0:
            last_rec_0 = self.df_preq_perf["rec_0"].iloc[-1]
            last_rec_1 = self.df_preq_perf["rec_1"].iloc[-1]

        acc = 0

        preq_rec_0, preq_rec_1 = 0,0

        if(y==0 or y == False):
            if(y_hat == 0):
                acc = 1
            self.s_alpha_0 = acc + self.theta * self.s_alpha_0
            self.n_alpha_0 = 1 + self.theta * self.n_alpha_0
            preq_rec_0 = self.s_alpha_0/self.n_alpha_0
            preq_rec_1 = last_rec_1
        elif(y==1 or y == True):
            if (y_hat == 1):
                acc = 1
            self.s_alpha_1 = acc + self.theta * self.s_alpha_1
            self.n_alpha_1 = 1 + self.theta * self.n_alpha_1
            preq_rec_1 = self.s_alpha_1 / self.n_alpha_1
            preq_rec_0 = last_rec_0

        dic_aux = {"date":[pred_date],"y":y,"rec_0":[preq_rec_0],"rec_1":[preq_rec_1],"gmean":[math.sqrt(preq_rec_1*preq_rec_0)],"perf_avail":[False]}
        df_aux = pd.DataFrame(dic_aux)
        self.df_preq_perf = pd.concat([self.df_preq_perf, df_aux])
        self.df_preq_perf.reset_index(inplace=True, drop=True)




    def get_perceived_performance(self):

        df_perf = self.df_online_perf.loc[(self.df_online_perf['y'].isin([0, 1])) & (self.df_online_perf['y_hat'].isin([0, 1]))]

        rec_0, rec_1 = 0, 0

        df_rec_0 = df_perf.loc[(df_perf['y']==0)]
        if(len(df_rec_0) > 0):
            rec_0 = len(df_rec_0.loc[(df_rec_0['y_hat'] == 0)])/len(df_rec_0)

        df_rec_1 = df_perf.loc[(df_perf['y'] == 1)]
        if(len(df_rec_1) > 0):
            rec_1 = len(df_rec_1.loc[(df_rec_1['y_hat'] == 1)])/len(df_rec_1)

        return rec_0, rec_1

    def get_lambda(self, target):
        maj_class = 0

        if(self.class_size[1] > self.class_size[0]):
            maj_class = 1

        lamb = self.class_size[maj_class]/self.class_size[target]

        return lamb

    def get_obf_pred_avg(self):

        avg = mean(self.last_n_pred)

        obf = [1,1]

        #boost class1
        if(avg < self.threshold):

            avg = abs(avg-self.threshold)
            obf[1] = (((self.m ** (avg*10))-1)/ ((self.m**(self.threshold*10))-1))*self.l1 + 1

        else:
            # boost class 0
            obf[0] = (((self.m ** (avg * 10)) - (self.m**(self.threshold * 10)))
                            / ((self.m ** 10) - (self.m**(self.threshold * 10)))) * self.l0 + 1

        return obf


    def update_class_size(self, target):
        #global class_size

        for idx in [0,1]:

            self.class_size[idx] = self.theta * self.class_size[idx]

            if (target == idx):
                self.class_size[idx] += (1 - self.theta)

    def store_training_inst(self, orig, idx):
        global df_pool_tr

        inst = orig.copy()
        inst["index_predict"] = idx

        if(inst["target"] == False):

            inst["train_date"] = inst["commit_date"].date() + timedelta(days=self.wt)
            del inst["contains_bug"]
            del inst["days_to_first_fix"]
            del inst["dataset"]
            del inst["commit_date"]

            for k in inst.keys():
                inst[k] = [inst[k]]

            df_aux = pd.DataFrame(inst)
            self.df_pool_tr = pd.concat([df_aux,self.df_pool_tr])

        else:

            commit_date = inst["commit_date"].date()
            days_to_fix = inst["days_to_first_fix"]
            inst["train_date"] = inst["commit_date"].date() + timedelta(days=math.ceil(inst["days_to_first_fix"]))
            del inst["contains_bug"]
            del inst["days_to_first_fix"]
            del inst["dataset"]
            del inst["commit_date"]

            for k in inst.keys():
                inst[k] = [inst[k]]

            df_aux = pd.DataFrame(inst)
            self.df_pool_tr = pd.concat([df_aux, self.df_pool_tr],ignore_index=True)

            if(days_to_fix > self.wt):
                inst_noise = inst.copy()
                inst_noise["train_date"] = commit_date + timedelta(days=self.wt)
                inst_noise["target"] = False

                df_aux = pd.DataFrame(inst_noise)
                self.df_pool_tr = pd.concat([df_aux, self.df_pool_tr])

    def get_available_tr_inst(self, cur_date):
        # global df_pool_tr
        # global trained_instances

        self.df_pool_tr.sort_values(by=['train_date'],inplace=True)
        self.df_pool_tr.sort_values(by=['train_date', "target"], inplace=True, ascending=[True, True])
        df_available = self.df_pool_tr.loc[self.df_pool_tr["train_date"] <= cur_date.date()]


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
        self.df_pool_tr.drop(l_training_idx,inplace=True)

        return df_return

    def train_on_available_inst(self, df_avail):

        for idx, inst in df_avail.iterrows():

            target = 0
            if (inst["target"]):
                target = 1

            self.update_class_size(target)

            x = inst.copy()
            del x["target"]
            del x["train_date"]

            #update the dictionary that is going to be used to check potential noisy defect-inducing instances
            if(target == 0):
                s = ""
                for k in x.keys():
                    if(k != "author_date_unix_timestamp"):
                        s += str(x[k]) + "-"

                if(self.dic_ocurr_clean_inst.__contains__(s)):
                    self.dic_ocurr_clean_inst[s] = self.dic_ocurr_clean_inst[s] + 1
                else:
                    self.dic_ocurr_clean_inst[s] = 1
            else:
                #skip the trainig instance if it is defect inducing and has been also very used to train as clean
                #the noisy mechanism of orb
                s = ""
                for k in x.keys():
                    if (k != "author_date_unix_timestamp"):
                        s += str(x[k]) + "-"

                if (self.dic_ocurr_clean_inst.__contains__(s)):
                    if(self.dic_ocurr_clean_inst[s] > self.n):
                        return

            avail_date  = datetime.utcfromtimestamp(int(x["author_date_unix_timestamp"]))
            if(target == 0):
                self.df_preq_perf.loc[self.df_preq_perf["date"] < avail_date,["perf_avail"]] = True
            # df1.loc[df1['stream'] == 2, ['feat', 'another_feat']] = 'aaaa'
            # self.update_online_perf(index=x["index_predict"],y_hat=math.nan, y=target)
            del x["author_date_unix_timestamp"]
            del x["index_predict"]

            for classifier in self.model:
                weight_k = np.random.poisson(lam=(self.get_lambda(target)), size=(1))
                if (weight_k[0] < 1):
                    weight_k = 1
                else:
                    weight_k = weight_k[0]

                obf = self.get_obf_pred_avg()[target]
                weight_obf = weight_k * obf

                classifier.learn_one(x, target, sample_weight=weight_obf)

    def __init__(
            self,
            m: int = 3,
            l0: float = 10,
            l1: float = 10,
            n: int = 100,
            threshold: float = 0.4,
            wt: int = 90,):

        self.m = m
        self.l0 = l0
        self.l1 = l1
        self.n = n
        self.threshold = threshold
        self.wt = wt

        self.model = ensemble.BaggingClassifier(model=(HoeffdingTreeClassifier(grace_period=100)),n_models=20,seed=12)
