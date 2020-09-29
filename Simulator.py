from VVR_Bank import VVR_Bank as Bank
import numpy as np
from copy import deepcopy
import numpy.random as random
import pandas as pd

class Simulator():
    def __init__(self, num_color=8, num_model=3, capacity=30, num_lanes=5, lane_length=8, VVR_temp=None, cc_file=None, color_dist_file=None, repeat=100, stoch=False, preference=1):
        self.num_color = num_color
        self.num_model = num_model
        self.num_lanes = num_lanes
        self.preference=preference
        self.repeat=repeat
        self.cc_file=cc_file
        self.lane_length = lane_length
        self.rewards = [1, 0, -1]  # [0]for unchange, [1]for change, [2]for error
        self.capacity = capacity
        self.bug_count = 0
        self.VVR_sim = VVR_temp
        self.ccm=None
        self.orders_num = 300
        self.stoch=stoch
        self.color_dist_file=None
        if cc_file is not None:
            self.ccm=self.read_cc_matrix(cc_file)
        if color_dist_file is not None:
            self.color_dist=self.read_color_dist(color_dist_file)
            self.model_dist = self.color_dist[:self.num_model] / sum(self.color_dist[:self.num_model])
            self.color_dist_file=color_dist_file
        self.reset()

    def read_cc_matrix(self,cc_file):
        ccm=pd.read_csv(cc_file)
        ccm=ccm.to_numpy(dtype=np.float,copy=True)[:,1:]/7.6
        return ccm

    def read_color_dist(self, color_file):
        ccm = pd.read_csv(color_file)
        ccm = ccm.to_numpy(dtype=np.float, copy=True).squeeze()
        ccm=ccm/sum(ccm)
        return ccm


    def reset(self):
        if self.color_dist_file is not None:
            self.start_sequencec = np.random.choice(range(self.num_color), self.orders_num, p=self.color_dist).tolist()
        else:
            self.start_sequencec = np.random.choice(range(self.num_color), self.orders_num).tolist()
        if self.color_dist_file is not None:
            self.start_sequencem = np.random.choice(range(self.num_model), self.orders_num, p=self.model_dist).tolist()
        else:
            self.start_sequencem = np.random.choice(range(self.num_model), self.orders_num).tolist()
        self.bank = Bank(fix_init=True, num_of_colors=self.num_model, sequence=self.start_sequencem,
                         num_of_lanes=self.num_lanes,
                         lane_length=self.lane_length)
        self.bank_c = Bank(fix_init=True, num_of_colors=self.num_color, sequence=self.start_sequencec,
                           num_of_lanes=self.num_lanes,
                           lane_length=self.lane_length)
        self.plan_list = {k: [c, m] for k, c, m in
                          zip(range(len(self.start_sequencec)), self.start_sequencec, self.start_sequencem)}
        self.mc_tab = np.zeros((self.num_model, self.num_color), dtype=np.int)
        self.last_color = -1
        self.job_list = np.zeros(self.num_model)
        self.cc = -1
        self.tard = 0
        self.bug_count = 0
        for i in range(self.capacity):
            self.BBA_rule_step_in()

    def init_VVR_sim(self):
        self.VVR_sim.bank_c.cursors = deepcopy(self.bank_c.cursors)
        self.VVR_sim.bank_c.state = self.get_swaped_state()
        self.VVR_sim.bank.cursors = deepcopy(self.bank.cursors)
        self.VVR_sim.bank.state = deepcopy(self.bank.state)
        self.VVR_sim.last_color = deepcopy(self.last_color)
        self.VVR_sim.plan_list = deepcopy(self.plan_list)
        self.VVR_sim.start_sequencec = deepcopy(self.start_sequencec)
        self.VVR_sim.start_sequencem = deepcopy(self.start_sequencem)
        self.VVR_sim.cc = 0
        self.VVR_sim.tard = 0

    def try_once_VVR(self):
        for i in range(sum(self.bank_c.cursors - 1)):
            if not self.VVR_sim.VVR_rule_out():
                print('error in VVR_out')
        return self.VVR_sim.cc

    def try_VVR(self):
        # if self.repeat==0:
        #     return
        self.init_VVR_sim()
        min_cc, min_tard = self.VVR_sim.sim_flush()
        min_obj=min_cc*(1-self.preference)+self.preference*min_tard
        new_obj=min_obj
        # lower_bound = np.sum(self.bank_c.state, 1)
        # lower_bound = sum(np.sum(lower_bound, 1) > 0) - 1
        for i in range(self.repeat):
            self.init_VVR_sim()
            st = self.get_swaped_state()
            self.VVR_sim.bank_c.state=deepcopy(st)# swap the color in sim
            new_cc, new_tard=self.VVR_sim.sim_flush()# calculate cc
            new_obj = new_cc * (1 - self.preference) + self.preference * new_tard
            if min_obj>new_obj:
                min_obj=new_obj
                self.bank_c.state=st
            # if lower_bound==new_cc:
            #     break
        return new_obj
        # print('after',min_cc)





    def get_swaped_state(self):
        models = np.sum(self.mc_tab, 1)
        models = models > 0
        models = models.astype(np.float)
        m = random.choice(range(self.num_model), 1, p=models / sum(models))[0]
        mm = np.reshape(self.bank.state[m, :, :-1], (1, -1)).squeeze()
        index = random.choice(range(len(mm)), 2, p=mm / sum(mm))
        index_depth = index % self.lane_length
        index_lane = (index - index_depth) // self.lane_length

        if not self.bank.state[m, index_lane[0], index_depth[0]]:
            if max(self.bank.state[m, index_lane[0], index_depth[0]] + self.bank.state[
                m, index_lane[1], index_depth[1]]) < 2:
                print('taking wrong position')

        ts = deepcopy(self.bank_c.state)
        temp = deepcopy(ts[:, index_lane[0], index_depth[0]])
        ts[:, index_lane[0], index_depth[0]] = deepcopy(ts[:, index_lane[1], index_depth[1]])
        ts[:, index_lane[1], index_depth[1]] = deepcopy(temp)
        return ts



    def VVR_rule_out(self, VVR=False):
        if VVR:
            self.try_VVR()
        r=np.random.uniform(0,1)

        if r<self.preference:
            models = self.bank.front_view()
            colors = self.bank_c.front_view()
            ks=[]
            for l in range(len(models)):
                m=models[l]
                c=colors[l]
                if m<0:
                    ks.append(0)
                else:
                    ks.append(self.find_nearest_order(m, c))
            ks=np.array(ks).astype(np.float)
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

    def BBA_rule_step_in(self):
        color = self.bank_c.in_queue[-1]
        model = self.bank.in_queue[-1]
        c = self.bank_c.check_rear(color)
        # c0 = self.bank_c.check_rear(model)
        # c0 = self.bank_c.check_all(color)
        # c1 = np.minimum(c, c0)
        if max(c) > 0:
            r = self.bank_c.insert(np.argmax(c))
            r *= self.bank.insert(np.argmax(c))
            self.mc_tab[model, color] += r
        else:
            c = self.bank_c.check_all(color)
            if max(c) > 0:
                r = self.bank_c.insert(np.argmax(c))
                r *= self.bank.insert(np.argmax(c))
                self.mc_tab[model, color] += r
            else:
                r = self.bank_c.insert(np.argmax(self.lane_length - self.bank_c.cursors))
                r *= self.bank.insert(np.argmax(self.lane_length - self.bank.cursors))
                self.mc_tab[model, color] += r

        return r

    def release(self, lane):
        model = self.bank.front_view()
        color = self.bank_c.front_view()
        model = model[lane]
        color = color[lane]
        if model < 0 or color < 0:
            return False  # return false if the lane is empty
        if self.mc_tab[model, color] > 0:
            if not self.bank.release(lane):
                return False  # return false if the release fails
            if not self.bank_c.release(lane):
                return False
            k=self.find_nearest_order(model,color)
            self.plan_list.pop(k)
            self.released_order_index=k
            self.mc_tab[model, color] -= 1
            if not (color == self.last_color):
                if self.ccm is None:
                    self.cc += 1
                elif self.stoch:
                    self.cc += int(np.random.uniform()<(self.ccm[color, self.last_color]))
                else:
                    self.cc +=self.ccm[color, self.last_color]
                self.last_color = color
            return True  # return True if the release success
        else:
            return False  # return false if the corresponding model and color was not needed anymore

    def find_nearest_order(self, m,c):
        for k in range(self.orders_num):
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