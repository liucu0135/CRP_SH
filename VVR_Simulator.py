from VVR_Bank import VVR_Bank as Bank
import numpy as np
from copy import deepcopy
import numpy.random as random
import pandas as pd

class VVR_Simulator():
    def __init__(self, num_color=8, num_model=3, capacity=35, num_lanes=5, lane_length=8, VVR_temp=None, cc_file=None, repeat=100, stoch=False):
        self.num_color = num_color
        self.num_model = num_model
        self.num_lanes = num_lanes
        self.lane_length = lane_length
        self.rewards = [1, 0, -1]  # [0]for unchange, [1]for change, [2]for error
        self.capacity = capacity
        self.reset()
        self.bug_count = 0
        self.VVR_sim = VVR_temp
        self.repeat=repeat
        self.ccm=None
        self.stoch=stoch
        if cc_file is not None:
            self.ccm=self.read_cc_matrix(cc_file)


    def read_cc_matrix(self,cc_file):
        ccm=pd.read_csv(cc_file)
        ccm=ccm.to_numpy(dtype=np.float,copy=True)[:,1:]
        return ccm

    def reset(self):
        self.start_sequencec = np.random.randint(0, self.num_color, 1500).tolist()
        self.start_sequencem = np.random.randint(0, self.num_model, 1500).tolist()
        self.bank = Bank(fix_init=True, num_of_colors=self.num_model, sequence=self.start_sequencem,
                         num_of_lanes=self.num_lanes,
                         lane_length=self.lane_length)
        self.bank_c = Bank(fix_init=True, num_of_colors=self.num_color, sequence=self.start_sequencec,
                           num_of_lanes=self.num_lanes,
                           lane_length=self.lane_length)
        self.mc_tab = np.zeros((self.num_model, self.num_color), dtype=np.int)
        self.last_color = -1
        self.job_list = np.zeros(self.num_model)
        self.cc = -1
        self.bug_count = 0
        # for i in range(self.capacity):
        #     self.BBA_rule_step_in()

    def save_temp(self):
        self.VVR_sim.bank_c.cursors = deepcopy(self.bank_c.cursors)
        self.VVR_sim.bank_c.state = self.get_swaped_state()
        self.VVR_sim.last_color = deepcopy(self.last_color)
        self.VVR_sim.cc = 0






    def get_swaped_state(self):
        models = np.sum(self.mc_tab, 1)
        models = models > 0;
        m=random.choice(range(self.num_model), 1, p=models / sum(models))[0]
        mm=np.reshape(self.bank.state[m,:,:-1],(1,-1)).squeeze()
        index=random.choice(range(len(mm)), 2, p=mm / sum(mm))
        index_depth=index%self.lane_length
        index_lane=(index-index_depth)//self.lane_length

        if not self.bank.state[m,index_lane[0],index_depth[0]]:
            if max(self.bank.state[m,index_lane[0],index_depth[0]]+self.bank.state[m,index_lane[1],index_depth[1]])<2:
                print('taking wrong position')

        ts=deepcopy(self.bank_c.state)
        temp=deepcopy(ts[:,index_lane[0],index_depth[0]])
        ts[:, index_lane[0], index_depth[0]]=deepcopy(ts[:, index_lane[1], index_depth[1]])
        ts[:, index_lane[1], index_depth[1]]=deepcopy(temp)
        return ts

    def sim_flush(self):
        for i in range(sum(self.bank_c.cursors - 1)):
            if not self.VVR_rule_out():
                print('vvr error')
        return self.cc

    def VVR_rule_out(self):
        last_color = self.select_color(self.last_color)
        for l in range(self.num_lanes):
            if last_color == self.bank_c.front_view()[l]:
                return self.release(l)
        print(self.bank_c.front_view())
        print(self.bank_c.get_view_state())
        print("empty")

    def select_color(self, last_color):
        c_hist = self.bank_c.front_together()
        c_disc = self.bank_c.front_discrete()
        if (last_color > -1) and (c_hist[last_color] > 0):  # continue if last color is available
            return last_color
        if max(c_hist * c_disc) == 0:  # if all colors are discrete then follow the together number
            # print(self.bank_c.get_view_state())
            jl = c_hist
            self.bug_count += 1
        else:
            jl = c_disc * c_hist  # if any color is not discrete, go for those with max together number
        color = np.argmax(jl)
        if color == None:
            print("not posiible")
            print(self.bank.get_view_state())
            print(self.mc_tab)
        return color


    def release(self, lane):
        # model = self.bank.front_view()
        color = self.bank_c.front_view()
        # model = model[lane]
        color = color[lane]
        if color < 0:
            return False  # return false if the lane is empty
        if not self.bank_c.release(lane):
            return False
        if not (color == self.last_color):
            if (self.ccm is None ) or (self.last_color<0):
                self.cc += 1
            elif self.stoch:
                self.cc += int(np.random.uniform() < (self.ccm[color, self.last_color] / 10))
            else:
                self.cc+=self.ccm[color, self.last_color]
            self.last_color = color
        return True  # return True if the release success
