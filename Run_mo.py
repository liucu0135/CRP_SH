from VVR_Bank import VVR_Bank as Bank
from VVR_Simulator import VVR_Simulator as VVR_Sim
from Simulator import Simulator as Sim
import copy
import pickle
import pandas as pd
import time


# color-modle:10-10, bank: 7X8, cc:96.0, td:197.5999999999999
# color-modle:10-10, bank: 7X8, cc:141.0, td:138.59999999999994
# color-modle:10-10, bank: 7X8, cc:139.0, td:138.2000000000001
# color-modle:10-10, bank: 7X8, cc:153.0, td:103.70000000000003
# color-modle:10-10, bank: 7X8, cc:174.0, td:95.69999999999999


paras = []
noc = [10, 20, 10, 10, 10, 20]
nom = [5,  10, 5,  10,  5, 10]
# nom = [10, 20, 10, 20, 10, 20]
# noc = [10, 5, 10, 5, 10, 5]
ll = [6, 6, 8, 8, 10, 10]
nol = [5, 5, 7, 7, 10, 10]
nof=[7,7,6,6,6,6]
ccs=[]
cc=[]
data=[noc,nom,nol,ll,nof]
sets=[]
# data=[]
ts=[]
tds=[]
times=[]
columns=['colors','models','lanes','lane length', 'filling','color change','time']


# for set in range(6):
set=3
pres=[0.2, 0.4, 0.5, 0.6, 0.8]
for pref in pres:
    preFill = nol[set]*ll[set]*nof[set] // 10
    st = VVR_Sim(num_color=noc[set], num_model=nom[set], num_lanes=nol[set], lane_length=ll[set], capacity=preFill, preference=pref)
    s = Sim(num_color=noc[set], num_model=nom[set], num_lanes=nol[set], lane_length=ll[set], capacity=preFill, VVR_temp=st,repeat=100, preference=pref)

    repeat_epoches=20
    # print(s.mc_tab)
    # print(s.bank.get_view_state())
    # print(s.bank.front_hist())
    # print(s.bank.front_view())

    cc_sum = 0
    td_sum = 0
    t_sum = 0
    for epoch in range(repeat_epoches):
        s.reset()
        start_time = time.time()
        for i in range(300-preFill-1):
            # if i%10==0:
            #     print(i, ' out of 1000 finihsed!     ', epoch, ' out of 10 epcoh', td_sum)
            s.BBA_rule_step_in()
            if s.VVR_rule_out(i%10==0):
                td=s.get_distortion(absolute=True, tollerance=0)/10
                td_sum+=td
            else:
                print("release failed")
        times.append(time.time() - start_time)
        cc.append(s.cc)
        tds.append(td)
        sets.append(set)
        cc_sum += s.cc
        t_sum += time.time() - start_time
    ccs.append(cc_sum/repeat_epoches)
    ts.append(t_sum/repeat_epoches)
    tds.append(t_sum/repeat_epoches)

    print("color-modle:{}-{}, bank: {}X{}, cc:{}, td:{}".format(noc[set],nom[set],nol[set],ll[set],cc_sum/repeat_epoches,td_sum/repeat_epoches))
    data.append(ccs)
    data.append(ts)
    data.append(pres)
    data.append(cc)
    data.append(times)
result=pd.DataFrame.from_records(data)
result.to_csv('result_mo_100.csv')