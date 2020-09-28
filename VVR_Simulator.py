from VVR_Bank import VVR_Bank as Bank
import numpy as np
from copy import deepcopy
import numpy.random as random
import pandas as pd

class VVR_Simulator():
    def __init__(self, num_color=8, num_model=3, capacity=35, num_lanes=5, lane_length=8, VVR_temp=None, cc_file=None, repeat=10, stoch=False, preference=0):
        self.preference= preference
        self.num_color = num_color
        self.num_model = num_model
        self.num_lanes = num_lanes
        self.lane_length = lane_length
        self.orders_num=300
        self.rewards = [1, 0, -1]  # [0]for unchange, [1]for change, [2]for error
        self.capacity = capacity
        self.reset()
        self.bug_count = 0
        self.tard=0
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
        self.start_sequencec = np.random.randint(0, self.num_color, self.orders_num).tolist()
        self.start_sequencem = np.random.randint(0, self.num_model, self.orders_num).tolist()
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
        self.plan_list = {k: [c, m] for k, c, m in
                          zip(range(len(self.start_sequencec)), self.start_sequencec, self.start_sequencem)}
        # for i in range(self.capacity):
        #     self.BBA_rule_step_in()

    def save_temp(self):
        self.VVR_sim.bank_c.cursors = deepcopy(self.bank_c.cursors)
        self.VVR_sim.bank_c.state = self.get_swaped_state()
        self.VVR_sim.last_color = deepcopy(self.last_color)
        self.VVR_sim.cc = 0






    def get_swaped_state(self):
        models = np.sum(self.mc_tab, 1)
        models = models > 0
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
        return self.cc, self.tard

    # def sim_flush(self):
    #     for i in range(sum(self.bank_c.cursors - 1)):
    #         if not self.VVR_rule_out():
    #             print('vvr error')
    #     return self.cc


    def find_nearest_order(self, m,c):
        for k in range(self.orders_num+1):
            if self.orders_num-k in self.plan_list:
                if self.plan_list[self.orders_num-k][0]==c and self.plan_list[self.orders_num-k][1]==m:
                    return self.orders_num-k
        print('nearest order not found')

    def search_model(self, color):
        job_list, color = self.find_job_list(color)
        for l in range(self.num_lanes):
            model = self.bank.front_view()[l]
            if (job_list[model]) and (model > -1):
                return model
        return 0
    def get_distortion(self, absolute=False, tollerance=0, color=0, model=0, sequence=None):
        if sequence is None:
            # absolute gives the actual KPI of tardness:  the deviation of each order in the sequence with a tolerance
            if absolute:
                n=np.maximum((self.released_order_index-len(self.start_sequencec)-self.capacity)-tollerance, 0)
                return n
            # self.plan_list is the orders that are delayed
            dist_count=0
            # $len(self.start_sequencec) + self.capacity$ is the number of order that should be painted now
            # i counts to $self.orders_num+1$ which is the total number of all generated orders in the simulator
            # this for loop iterates all orders which should be painted the
            for i in range(len(self.start_sequencec)+self.capacity, self.orders_num+1):
                if i in self.plan_list:  # if an order should be painted is not painted
                    dist_count+=i-len(self.start_sequencec)-self.capacity  # count the delay
            return dist_count
        else:
            n = np.maximum((self.released_order_index - len(self.start_sequencec) - self.capacity) - tollerance, 0)
            return n


    # def VVR_rule_out(self):
    #     last_color = self.select_color(self.last_color)
    #     for l in range(self.num_lanes):
    #         if last_color == self.bank_c.front_view()[l]:
    #             return self.release(l)
    #     print(self.bank_c.front_view())
    #     print(self.bank_c.get_view_state())
    #     print("empty")
    def VVR_rule_out(self, VVR=False):
        r = np.random.uniform(0, 1)
        if r < self.preference:
            models = self.bank.front_view()
            colors = self.bank_c.front_view()
            ks = []
            for l in range(len(models)):
                m = models[l]
                c = colors[l]
                if m < 0:
                    ks.append(0)
                else:
                    ks.append(self.find_nearest_order(m, c))
            ks = np.array(ks).astype(np.float)
            return self.release(np.argmax(ks))
            # return self.release(np.random.choice(range(len(ks)),p=ks/sum(ks)))
        else:
            last_color = self.select_color(self.last_color)
            for l in range(self.num_lanes):
                if last_color == self.bank_c.front_view()[l]:
                    # f = self.bank_c.front_view()[l]
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
        models = self.bank.front_view()
        colors = self.bank_c.front_view()
        model = models[lane]
        color = colors[lane]
        if color < 0:
            return False  # return false if the lane is empty
        if self.bank_c.release(lane):
            self.bank.release(lane)
            k=self.find_nearest_order(model,color)
            self.plan_list.pop(k)
            self.released_order_index=k
            self.tard+=self.get_distortion(absolute=True, tollerance=15)/5
        else:
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
