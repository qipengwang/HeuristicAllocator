from bufferAllocator import BufferAllocator
import math
import os
from collections import defaultdict
from collections import deque
import json

root = '/Users/wangqipeng/Desktop/MNN/build/'
IOrate=104094472/4452 #单位 bytes/ms
Swap_time=defaultdict(int)
Swap=defaultdict(list)

class Profiler:
    def __init__(self, model, batch):
        self.model = model
        self.batch = batch
        self.tensor_size = {}
        self.io_info = []
        self.resize_info = []
        self.redundent_parent = {}  # 存在不断alloc free alloc free的情况，这种应该是同一个tensor，去重
        self.cost_info = {}
        self.tensor_from_opid = {}

        self.fp_thres = -1
        if self.model == 'Googlenet':
            self.fp_thres = 218
        elif self.model == 'MobilenetV2':
            self.fp_thres = 1438
        elif self.model == 'MobilenetV1':
            self.fp_thres = None
        elif self.model == 'Squeezenet':
            self.fp_thres = 119
        # self.profile()
        # self.resize()

    def profile(self):
        def add_info(ln, tag):
            nonlocal self
            ln = ln.strip().split(':')[-1].strip().replace('[', '').replace(']', '').strip().split(',')
            tmp = set()
            for item in ln:
                if len(item):
                    item = item.strip().replace('(', '').replace(')', '').split()
                    tid, tsize = int(item[0]), int(item[1])
                    self.tensor_size[tid] = tsize
                    tmp.add(tid)
            self.io_info[-1][tag] = list(tmp)

        profile_flag = False
        with open(os.path.join(root, 'profile', f'{self.model}.{self.batch}.profile.out')) as f:
            for line in f:
                if line.strip().endswith('start read-map'):
                    profile_flag = True
                elif line.strip().endswith('finish read-map & start replace'):
                    profile_flag = False
                if not profile_flag:
                    continue

                if line.strip().startswith('current Op'):
                    op = line.strip().split()[-1]
                    opid = len(self.io_info)
                    self.io_info.append({'op': op, 'id': opid})
                elif line.startswith('\t') and line.strip().startswith('outputs'):
                    add_info(line, 'outputs')
                    for t in self.io_info[-1]['outputs']:
                        self.tensor_from_opid[t] = len(self.io_info) - 1
                elif line.startswith('\t') and line.strip().startswith('release'):
                    add_info(line, 'release')
                elif line.startswith('\t') and line.strip().startswith('inputs'):
                    add_info(line, 'inputs')
                elif line.startswith('\t') and line.strip().startswith('temporary'):
                    add_info(line, 'temporary')
                    for t in self.io_info[-1]['temporary']:
                        assert t in self.io_info[-1]['outputs']

    def resize(self):
        resize_flag = False
        opid = None
        resize_tid = 0
        compute_flag = False
        freed = None
        with open(os.path.join(root, 'resize', f'{self.model}.{self.batch}.resize.out')) as f:
            for line in f:
                if 'start read-map' in line:
                    compute_flag = True
                if 'finish read-map' in line:
                    compute_flag = False
                if not compute_flag:
                    continue

                if line.strip().startswith('start compute cmd'):
                    opid = int(line.strip().split('[')[1].split(']')[0])
                    # print(opid)
                if line.strip().startswith('finish allocate memory for cmd'):
                    resize_flag = True
                    # print(self.resize_info)
                    assert opid == len(self.resize_info)
                    self.resize_info.append([])
                    freed = set()
                if line.strip().startswith('try get') and resize_flag:
                    size = BufferAllocator.aligned_size(int(line.strip().split()[2]))
                    rtid = f'{opid}:{resize_tid}'
                    self.tensor_size[rtid] = size
                    redundent = None
                    for t in freed:
                        if self.tensor_size[t] == size:
                            freed.remove(t)
                            redundent = t
                            break
                    if redundent:
                        for i in range(len(self.resize_info[-1]) - 1, -1, -1):
                            if self.resize_info[-1][i][1] == redundent:
                                self.resize_info[-1].pop(i)
                        self.redundent_parent[redundent] = rtid
                    self.resize_info[-1].append(('alloc', rtid))
                    resize_tid += 1
                if line.strip().startswith('try return') and resize_flag:
                    size = BufferAllocator.aligned_size(int(line.strip().split()[2]))
                    talloc, tfree = [], []
                    for a, tid in self.resize_info[-1]:
                        if a == 'alloc' and self.tensor_size[tid] == size:
                            talloc.append(tid)
                        if a == 'free' and self.tensor_size[tid] == size:
                            tfree.append(tid)
                    # print(opid, self.resize_info[opid], [self.tensor_size[t] for a, t in self.resize_info[opid] if a == 'alloc'], size)
                    tid = [t for t in talloc if t not in tfree][-1]
                    self.resize_info[-1].append(('free', tid))
                    freed.add(tid)
                if line.strip().startswith('finish resize cmd'):
                    resize_flag = False
                    resize_tid = 0

    def cost(self):
        cost_flag = False
        with open(os.path.join(root, 'cost', f'{self.model}.{self.batch}.cost.out')) as f:
            for line in f:
                if line.strip().endswith('start read-map'):
                    cost_flag = True
                elif line.strip().endswith('finish read-map & start replace'):
                    cost_flag = False
                if not cost_flag:
                    continue

                if line.strip().startswith('current Op'):
                    op = line.strip().split()[-1]
                    opid = int(op.split('th')[0])
                elif 'cost time' in line:
                    self.cost_info[opid] = float(line.strip().split()[-2])

    def load_infos(self):
        with open(os.path.join('data/profiler', self.model, f'{self.model}.io_info.json'), 'r') as f:
            self.io_info = json.load(f)
        for i in range(len(self.io_info)):
            for t in self.io_info[i]['outputs']:
                self.tensor_from_opid[t] = i
        with open(os.path.join('data/profiler', self.model, f'{self.model}.{self.batch}.resize_info.json'), 'r') as f:
            self.resize_info = json.load(f)
        with open(os.path.join('data/profiler', self.model, f'{self.model}.{self.batch}.tensor_size.json'), 'r') as f:
            self.tensor_size = json.load(f)
        for t in list(self.tensor_size.keys()):
            if ':' not in t:
                self.tensor_size[int(t)] = self.tensor_size.pop(t)
        with open(os.path.join('data/profiler', self.model, f'{self.model}.{self.batch}.redundent_parent.json'), 'r') as f:
            self.redundent_parent = json.load(f)
        with open(f'data/profiler/{self.model}/{self.model}.{self.batch}.cost_info.json') as f:
            self.cost_info=json.load(f)
        for t in list(self.cost_info.keys()):
            self.cost_info[int(t)] = self.cost_info[t]


class Allocator:
    def __init__(self):
        self.memory_peak=0      
        self.time=0
    
    def Semi_Synchronous(self,profiler: Profiler):#半同步
        buffer_allocator=BufferAllocator()

        for i in range(profiler.fp_thres):
            # 对于每次计算，一定是先分配io_info的output，resize的alloc；
            # 然后释放resize的free，io_info的release
            # 最后swap out tensor
            for t in profiler.io_info[i]['outputs']:
                buffer_allocator.alloc(t, profiler.tensor_size[t])
 
            for a, t in profiler.resize_info[i]:  # tmp
                if a == 'alloc':
                    buffer_allocator.alloc(t, profiler.tensor_size[t])
            # compute
            for a, t in profiler.resize_info[i]:
                if a == 'free':
                    buffer_allocator.free(t)
            for t in profiler.io_info[i]['release']:  # rel input
                buffer_allocator.free(t)
            # if i==0:
            #     self.time+=profiler.cost_info[i]
            # else:
            self.time+=max(profiler.cost_info[i],Swap_time[i-1])#一次op的耗时 
            for t in Swap[i-1]:#swapout之后将内存释放
                buffer_allocator.free(t)
                
          #################################################################### 
                 
        self.time+=Swap_time[profiler.fp_thres-1] #FP中最后一个tensor的swapout
        for t in Swap[profiler.fp_thres-1]:#swapout之后将内存释放
            buffer_allocator.free(t)
              
        self.time+=Swap_time[profiler.fp_thres] #BP中第一个tensor的swapin
        for t in Swap[profiler.fp_thres]:#swapin载入内存
            buffer_allocator.alloc(t, profiler.tensor_size[t])       
                
          #################################################################### 
                  
        for i in range(profiler.fp_thres,len(profiler.io_info)):
            # 对于每次计算，先分配swapin tensor, 分配io_info的output，resize的alloc；
            # 然后释放resize的free，io_info的release
            self.time+=max(profiler.cost_info[i],Swap_time[i+1]) #半同步
            for t in Swap[i+1]:
                buffer_allocator.alloc(t, profiler.tensor_size[t])    #将下一步需要的tensor换入内存
           
            for t in profiler.io_info[i]['outputs']:
                buffer_allocator.alloc(t, profiler.tensor_size[t])
            
            
            for a, t in profiler.resize_info[i]:  # tmp
                if a == 'alloc':
                    buffer_allocator.alloc(t, profiler.tensor_size[t])
            # compute
            for a, t in profiler.resize_info[i]:
                if a == 'free':
                    buffer_allocator.free(t)
            for t in profiler.io_info[i]['release']:  # rel input
                buffer_allocator.free(t)

                
        self.memory_peak =buffer_allocator.tot_size
        return buffer_allocator.tot_size
    
    
    ####################################################################    
    def Synchronous(self,profiler: Profiler):#全同步
        buffer_allocator=BufferAllocator()

        for i in range(profiler.fp_thres):
            # 对于每次计算，一定是先分配io_info的output，resize的alloc；
            # 然后释放resize的free，io_info的release
            # 最后swap out tensor
            for t in profiler.io_info[i]['outputs']:
                buffer_allocator.alloc(t, profiler.tensor_size[t])
 
            for a, t in profiler.resize_info[i]:  # tmp
                if a == 'alloc':
                    buffer_allocator.alloc(t, profiler.tensor_size[t])
            # compute
            for a, t in profiler.resize_info[i]:
                if a == 'free':
                    buffer_allocator.free(t)
            for t in profiler.io_info[i]['release']:  # rel input
                buffer_allocator.free(t)
                
            self.time+=profiler.cost_info[i]+Swap_time[i]
            for t in Swap[i]:#swapout之后将内存释放
                buffer_allocator.free(t)

                
          #################################################################### 
                  
        for i in range(profiler.fp_thres,len(profiler.io_info)):
            # 对于每次计算，一定是先swapin tensor, 分配io_info的output，resize的alloc；
            # 然后释放resize的free，io_info的release
            self.time+=Swap_time[i]+profiler.cost_info[i]
            for t in Swap[i]:
                buffer_allocator.alloc(t, profiler.tensor_size[t])    #将下一步需要的tensor换入内存
           
            for t in profiler.io_info[i]['outputs']:
                buffer_allocator.alloc(t, profiler.tensor_size[t])
            
            
            for a, t in profiler.resize_info[i]:  # tmp
                if a == 'alloc':
                    buffer_allocator.alloc(t, profiler.tensor_size[t])
            # compute
            for a, t in profiler.resize_info[i]:
                if a == 'free':
                    buffer_allocator.free(t)
            for t in profiler.io_info[i]['release']:  # rel input
                buffer_allocator.free(t)

                
        self.memory_peak =buffer_allocator.tot_size
        return buffer_allocator.tot_size
    
    ####################################################################
    def Asynchronous(self,profiler: Profiler):#异步
        buffer_allocator=BufferAllocator()
        time_swap=0
        FIFO=deque()
        for i in range(profiler.fp_thres):#FP阶段，计算第i个op
           
            for t in profiler.io_info[i]['outputs']:
                buffer_allocator.alloc(t, profiler.tensor_size[t])
 
            for a, t in profiler.resize_info[i]:  # tmp
                if a == 'alloc':
                    buffer_allocator.alloc(t, profiler.tensor_size[t])
            # compute
            for a, t in profiler.resize_info[i]:
                if a == 'free':
                    buffer_allocator.free(t)
            for t in profiler.io_info[i]['release']:  # rel input
                buffer_allocator.free(t)
                
       
            self.time+=profiler.cost_info[i]    #self.time：第i个op后时间花费
            while len(FIFO) and FIFO[0][0]<=self.time:
                buffer_allocator.free(FIFO[0][1])#swapout之后将内存释放       
                FIFO.popleft()
                
            time_swap=max(time_swap,self.time)#time_swap：第i个op后swapout总时间花费
            
            for t in Swap[i]:         
                time_swap+=profiler.tensor_size[t]/IOrate
                FIFO.append((time_swap,t))#在time_swap时刻完成tensor t 的换出
                
       ####################################################################           
        while len(FIFO):
            buffer_allocator.free(FIFO[0][1])#swapout之后将内存释放     
            FIFO.popleft()        
       
        j=  profiler.fp_thres#接下来换入第j个op的tensor
        time_swap+=Swap_time[j]
        for t in Swap[j]:
            buffer_allocator.alloc(t, profiler.tensor_size[t])   
        j+=1
                
        #################################################################### 
        
        for i in range(profiler.fp_thres,len(profiler.io_info)):#BP阶段，计算第i个op
            # 对于每次op计算，先分配swapin的tensor, io_info的output，resize的alloc；
            # 然后释放resize的free，io_info的release
            
            if i==j-1:                      #前j-1个op的tensor已换入
                self.time=max(self.time,time_swap)#第i个op需等待swapin[i]完成后在进行

            time_swap=max(self.time,time_swap)   #前j-1个op的换入总用时是time_swap
            while time_swap<=self.time and j<len(profiler.io_info):
                #若time_swap<=self.time，则第j个换入的开始时刻是self.time
                time_swap+=Swap_time[j]
                for t in Swap[j]:
                    buffer_allocator.alloc(t, profiler.tensor_size[t])   
                j+=1
          
            self.time+=profiler.cost_info[i] #self.time：前i个op总用时

            while time_swap+Swap_time[j]<=self.time and j<len(profiler.io_info):
                #在第i个op申请多个swapin
                time_swap+=Swap_time[j]
                for t in Swap[j]:
                    buffer_allocator.alloc(t, profiler.tensor_size[t])   
                j+=1
            ####################################################################
            for t in profiler.io_info[i]['outputs']:
                buffer_allocator.alloc(t, profiler.tensor_size[t])
           
            for a, t in profiler.resize_info[i]:  # tmp
                if a == 'alloc':
                    buffer_allocator.alloc(t, profiler.tensor_size[t])
            # compute
            for a, t in profiler.resize_info[i]:
                if a == 'free':
                    buffer_allocator.free(t)
            for t in profiler.io_info[i]['release']:  # rel input
                buffer_allocator.free(t)

                
        self.memory_peak =buffer_allocator.tot_size
        return buffer_allocator.tot_size
    
########################################################################################################################################    
def oracle(profiler):
    cur, peak = 0, 0
    for i in range(len(profiler.io_info)):
        for t in profiler.io_info[i]['outputs']:
            cur += profiler.tensor_size[t]
            peak = max(cur, peak)
        for a, t in profiler.resize_info[i]:
            if a == 'alloc':
                cur += profiler.tensor_size[t]
                peak = max(cur, peak)
        # compute
        for a, t in profiler.resize_info[i]:
            if a == 'free':
                cur -= profiler.tensor_size[t]
        for t in profiler.io_info[i]['release']:
            cur -= profiler.tensor_size[t]
    print(cur, peak)
    return peak


def get_featuremap(profiler: Profiler):
    feature_map = set()
    bo = set()
    for i in range(profiler.fp_thres):
        for t in profiler.io_info[i]['outputs']:
            feature_map.add(t)
        for t in profiler.io_info[i]['release']:
            feature_map.remove(t)
    # print(sorted(feature_map))
    for i in range(profiler.fp_thres-1,-1,-1):#第i次op
        for c in ("inputs","outputs"):
            for t in profiler.io_info[i][c]:
                if t in feature_map and t not in bo:#将tensor t换出
                    bo.add(t)
                    Swap_time[i]+=profiler.tensor_size[t]/IOrate
                    Swap[i].append(t)
                    
    for i in range(profiler.fp_thres,len(profiler.io_info)):   #第i次op
        for t in profiler.io_info[i]['inputs']:
            if t in feature_map and t in bo:#将tensor t换入
                bo.remove(t)
                Swap_time[i]+=profiler.tensor_size[t]/IOrate
                Swap[i].append(t)      
    # for i in range(len(profiler.io_info)):
    #     if i == profiler.fp_thres:
    #         print('-'*50)
    #     if Swap[i]:
    #         print(i,end=":")
    #         for t in Swap[i]:
    #             print(t,end=",")
    #         print()
    return feature_map


if __name__ == '__main__':
    # for model in ['Squeezenet', 'Googlenet', 'MobilenetV1', 'MobilenetV2']:
    #     for batch in range(2, 17):
    #         print(model, batch)
    #         profiler = Profiler(model, batch)
    #         b = baseline(profiler)
    #         o = oracle(profiler)
    #         print(b, o)
    #         # heuristic_allocator = HeuristicAllocator(profiler)
    #         # heuristic_allocator.dump_heuristic_info()
    #         # heuristic_allocator.heuristic_alloc()
    #         input()
    # assert 0
    model = 'Googlenet'
    profiler = Profiler(model, 4)
    profiler.load_infos()  # 我把数据都dump下来了，这句话直接读进来即可
    feature_map = get_featuremap(profiler)
    
    
    b=Allocator()
    b.Synchronous(profiler)
    print("     Synchronous memory:", b.memory_peak,"time:",b.time)
    
    a=Allocator()
    a.Semi_Synchronous(profiler)
    print("Semi_Synchronous memory:", a.memory_peak,"time:",a.time)
    
    c=Allocator()
    c.Asynchronous(profiler)
    print("    Asynchronous memory:", c.memory_peak,"time:",c.time)
    
    # print(profiler.io_info[1]['release'][0])
    # print(profiler.resize_info[7])
    # print(profiler.tensor_size[36],profiler.tensor_size["36:0"])
    # print(feature_map)
    
    # todo: simulate swapping
    #       via OS  --> oracle
    #       via buffer_allocator
