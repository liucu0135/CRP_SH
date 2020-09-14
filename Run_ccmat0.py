from VVR_Bank import VVR_Bank as Bank
from VVR_Simulator import VVR_Simulator as VVR_Sim
from Simulator import Simulator as Sim
import copy
import pickle
import pandas as pd
import time





paras = []
noc = [14,14,14,14,14,14]
nom = [7, 10, 7, 10, 7, 10]
ll = [6, 6, 8, 8, 10, 10]
nol = [5, 5, 7, 7, 10, 10]
nof=[7,7,6,6,6,6]
ccs=[]
cc=[]
data=[noc,nom,nol,ll,nof]
sets=[]
# data=[]
ts=[]
times=[]
columns=['colors','models','lanes','lane length', 'filling','color change','time']


for set in range(6):
    preFill = nol[set]*ll[set]*nof[set] // 10
    st = VVR_Sim(num_color=noc[set], num_model=nom[set], num_lanes=nol[set], lane_length=ll[set], capacity=0,cc_file='./csv_files/cost.csv')
    s = Sim(num_color=noc[set], num_model=nom[set], num_lanes=nol[set], lane_length=ll[set], capacity=0, VVR_temp=st,cc_file='./csv_files/cost.csv', color_dist_file='./csv_files/total_orders.csv',repeat=0)

    repeat_epoches=10
    # print(s.mc_tab)
    # print(s.bank.get_view_state())
    # print(s.bank.front_hist())
    # print(s.bank.front_view())

    cc_sum = 0
    t_sum = 0
    for epoch in range(repeat_epoches):
        s.reset()
        start_time = time.time()
        for i in range(1000):
            # sp.bank_c.state=copy.deepcopy(s.bank_c.state)
            if i%200==0:
                print(i, ' out of 1000 finihsed!     ', epoch, ' out of 10 epcoh')
            if i < 2000:
                s.BBA_rule_step_in()
            if i > preFill:
                if not s.VVR_rule_out():
                    print("release failed")
        times.append(time.time() - start_time)
        cc.append(s.cc)
        sets.append(set)
        cc_sum += s.cc
        t_sum += time.time() - start_time
    ccs.append(cc_sum/repeat_epoches)
    ts.append(t_sum/repeat_epoches)

    print("color-modle:{}-{}, bank: {}X{}, cc:{}".format(noc[set],nom[set],nol[set],ll[set],cc_sum/repeat_epoches))
data.append(ccs)
data.append(ts)
data.append(sets)
data.append(cc)
data.append(times)
result=pd.DataFrame.from_records(data)
result.to_csv('result_ccmat0.csv')