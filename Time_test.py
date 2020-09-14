from VVR_Bank import VVR_Bank as Bank
import numpy as np
from Simulator import Simulator as Sim
from VVR_Simulator import VVR_Simulator as vSim
import time
st = vSim(num_color=14, num_model=7, num_lanes=7, lane_length=8,)
s = Sim(num_color=14, num_model=7, num_lanes=7, lane_length=8, capacity=0,VVR_temp=st, cc_file='./csv_files/cost.csv')
preFill=30
ccsum=0
start_time=time.time()
for rep in range(1):
    s.reset()
    for i in range(1000):
        # sp.bank_c.state=copy.deepcopy(s.bank_c.state)
        if i % 10 == 0:
            print(i, ' out of 1000 finihsed')
        if i < 2000:
            s.BBA_rule_step_in()
        if i > preFill:
            if not s.VVR_rule_out():
                print("release failed")
    print(rep,':',s.cc)
    ccsum+=s.cc
print('average:{}'.format(ccsum))
print('time:{}'.format(time.time()-start_time))

