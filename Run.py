from VVR_Bank import VVR_Bank as Bank
from VVR_Simulator import VVR_Simulator as VVR_Sim
from Simulator import Simulator as Sim
import copy
import pickle
import pandas as pd
import time





paras = []
noc = [10, 20, 10, 20, 10, 20]
nom = [5, 10, 5, 10, 5, 10]
ll = [6, 6, 8, 8, 10, 10]
nol = [5, 5, 7, 7, 10, 10]
nof=[7,7,6,6,6,6]
ccs=[]
data=[noc,nom,nol,ll,nof]
times=[]
columns=['colors','models','lanes','lane length', 'filling','color change']


for set in range(6):
    preFill = nol[set]*ll[set]*nof[set] // 10
    st = VVR_Sim(num_color=noc[set], num_model=nom[set], num_lanes=nol[set], lane_length=ll[set], capacity=0)
    s = Sim(num_color=noc[set], num_model=nom[set], num_lanes=nol[set], lane_length=ll[set], capacity=0, VVR_temp=st)

    repeat_epoches=10
    # print(s.mc_tab)
    # print(s.bank.get_view_state())
    # print(s.bank.front_hist())
    # print(s.bank.front_view())

    cc_sum = 0
    for epoch in range(repeat_epoches):
        start_time = time.time()
        s.reset()
        for i in range(1000):
            # sp.bank_c.state=copy.deepcopy(s.bank_c.state)
            if i%200==0:
                print(i, ' out of 1000 finihsed!     ', epoch, ' out of 10 epcoh')
            if i < 2000:
                s.BBA_rule_step_in()
            if i > preFill:
                if not s.VVR_rule_out():
                    print("release failed")
        cc_sum += s.cc
        times.append(time.time() - start_time)
    ccs.append(cc_sum/repeat_epoches)

    print("color-modle:{}-{}, bank: {}X{}, cc:{}".format(noc[set],nom[set],nol[set],ll[set],cc_sum/repeat_epoches))
data.append(times)
result=pd.DataFrame.from_items(zip(columns,data))
result.to_csv('result_time.csv')