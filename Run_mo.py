from VVR_Bank import VVR_Bank as Bank
from VVR_Simulator import VVR_Simulator as VVR_Sim
from Simulator import Simulator as Sim
import copy
import pickle
import pandas as pd
import time

# vvr_rep10, vvr%10:
# color-modle:10-10, bank: 7X8, cc:96.15, td:177.44999999999916
# color-modle:10-10, bank: 7X8, cc:129.55, td:124.5549999999995
# color-modle:10-10, bank: 7X8, cc:151.55, td:94.62999999999963
# color-modle:10-10, bank: 7X8, cc:168.25, td:83.93499999999969
# color-modle:10-10, bank: 7X8, cc:194.9, td:54.75500000000062


# vvr_rep100, vvr%10: ccm
# color-modle:10-10, bank: 7X8, cc:152.73299999999995, td:182.69499999999948
# color-modle:10-10, bank: 7X8, cc:202.20150000000004, td:138.4749999999995
# color-modle:10-10, bank: 7X8, cc:229.28100000000003, td:115.04000000000003
# color-modle:10-10, bank: 7X8, cc:254.74900000000002, td:96.66499999999982
# color-modle:10-10, bank: 7X8, cc:288.29550000000006, td:75.40999999999976

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
    st = VVR_Sim(num_color=noc[set], num_model=nom[set], num_lanes=nol[set], lane_length=ll[set], capacity=preFill, preference=pref,cc_file='./csv_files/cost1.csv')
    s = Sim(num_color=noc[set], num_model=nom[set], num_lanes=nol[set], lane_length=ll[set], capacity=preFill, VVR_temp=st,repeat=100, preference=pref,cc_file='./csv_files/cost1.csv', color_dist_file='./csv_files/total_orders.csv')

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
            if s.VVR_rule_out(i%1==0):
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
result.to_csv('result_mo_ccm_mod1.csv')