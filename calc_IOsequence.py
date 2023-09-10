from bufferAllocator import BufferAllocator
import math
import os
from collections import defaultdict
from collections import deque
import json
import heapq

root = '/Users/wangqipeng/Desktop/MNN/build/'


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
            self.cost_info = json.load(f)
        for t in list(self.cost_info.keys()):
            self.cost_info[int(t)] = self.cost_info[t]


########################################################################################################################################

class DXR:
    bo = defaultdict(int)
    # bo[t] 0:not in buffer / 1:useless_tensor / 2:buffer_tensor / 3:not allow swapout
    Swap = defaultdict(list)
    SwapOut = defaultdict(list)
    SwapIn = defaultdict(list)
    buffer_size = 4 * 1024**3  # 1.92e8
    useless_tensor = []
    buffer_tensor = []
    IOrate = 107374182400 / 270759.343750  # 单位 bytes/ms


class Fix_Buffer:
    @classmethod
    def Replace(cls, cur, tensor_size, i):
        while (cur + tensor_size > DXR.buffer_size):

            # swap out from ueseless_tensor
            while len(DXR.useless_tensor) and DXR.bo[DXR.useless_tensor[0][1]] != 1:
                heapq.heappop(DXR.useless_tensor)
            if len(DXR.useless_tensor):
                t = DXR.useless_tensor[0][1]
                # print(f"Swapout {t} size:{profiler.tensor_size[t]}  ")
                cur += DXR.useless_tensor[0][0]  # swao out
                DXR.bo[DXR.useless_tensor[0][1]] = 0
                DXR.SwapOut[i].append(DXR.useless_tensor[0][1])

                heapq.heappop(DXR.useless_tensor)

            else:  # swap out from FP_tensor
                while len(DXR.buffer_tensor) and DXR.bo[DXR.buffer_tensor[0][1]] != 2:
                    # print(i, tensor_size, len(DXR.buffer_tensor)," tensor: ",DXR.buffer_tensor[0][1],DXR.bo[DXR.buffer_tensor[0][1]])
                    heapq.heappop(DXR.buffer_tensor)
                if len(DXR.buffer_tensor):

                    cur += DXR.buffer_tensor[0][0]  # swao out
                    DXR.bo[DXR.buffer_tensor[0][1]] = 0
                    DXR.SwapOut[i].append(DXR.buffer_tensor[0][1])
                    heapq.heappop(DXR.buffer_tensor)
                else:
                    print("推荐将buffer_size 设置为", cur + tensor_size)
                    assert 0, "buffer_size is too small"
                    # cur是局部变量，要修改oracle的cur
        return cur

    @classmethod
    def oracle(cls, profiler):
        cur = 0
        Swap_Mark = 0  # 是否存在IO
        t1 = 0
        t2 = 0
        t3 = 0
        for i in range(len(profiler.io_info)):
            # cancel useless_tensor, merge useless_tensor and buffer_tensor
            if i == profiler.fp_thres:
                while len(DXR.useless_tensor):
                    t = DXR.useless_tensor[0][1]
                    if DXR.bo[t] == 1:
                        heapq.heappush(DXR.buffer_tensor, (-profiler.tensor_size[t], t))
                        DXR.bo[t] = 2
                    heapq.heappop(DXR.useless_tensor)

            # alloc
            Alloc = []
            for c in ("inputs", "outputs"):
                for t in profiler.io_info[i][c]:
                    if DXR.bo[t] == 0:
                        Alloc.append(t)
                        if c == "inputs":
                            DXR.SwapIn[i].append(t)
                    DXR.bo[t] = 3

            for t in Alloc:
                cur = Fix_Buffer.Replace(cur, profiler.tensor_size[t], i)
                cur += profiler.tensor_size[t]

            temporary = 0
            for a, t in profiler.resize_info[i]:
                if a == 'alloc':
                    temporary += profiler.tensor_size[t]
            cur = Fix_Buffer.Replace(cur, temporary, i)

            # compute

            for c in ("inputs", "outputs"):
                for t in profiler.io_info[i][c]:
                    if t in profiler.io_info[i]['release']:
                        DXR.bo[t] = 0
                        cur -= profiler.tensor_size[t]
                    else:
                        if t in DXR.Swap[i]:  # t is useless in FP
                            heapq.heappush(DXR.useless_tensor, (-profiler.tensor_size[t], t))
                            DXR.bo[t] = 1
                        else:
                            heapq.heappush(DXR.buffer_tensor, (-profiler.tensor_size[t], t))
                            DXR.bo[t] = 2
            ########################################################################################################################################
            # 输出swapIO tensor,计算operation和swap并行时的用时

            if len(DXR.SwapIn[i]) + len(DXR.SwapOut[i]):  # 存在IO
                Swap_Mark = 1

                print("\n", '-' * 20, "\noperation ", i)
            if len(DXR.SwapIn[i]):
                print("SwapIn")
                for t in DXR.SwapIn[i]:
                    print(f"{t} size:{profiler.tensor_size[t]}  ", end="")
                    t3 += profiler.tensor_size[t]
            if len(DXR.SwapOut[i]):
                print("SwapOut")
                for t in DXR.SwapOut[i]:
                    print(f"{t} size:{profiler.tensor_size[t]}  ", end="")
                    t3 += profiler.tensor_size[t]
            if Swap_Mark == 0:
                t1 += profiler.cost_info[i]
            else:
                t2 += profiler.cost_info[i]
        print("\nTotal Time :", t1 + max(t2, 1.0 * t3 / DXR.IOrate))
        print(f"t1:{t1}  t2:{t2}  t3:{1.0 * t3 / DXR.IOrate}")

    ########################################################################################################################################

    @classmethod
    def Featuremap(cls, profiler: Profiler):
        feature_map = set()
        for i in range(profiler.fp_thres):
            for t in profiler.io_info[i]['outputs']:
                feature_map.add(t)
            for t in profiler.io_info[i]['release']:
                feature_map.remove(t)
        # print(sorted(feature_map))
        for i in range(profiler.fp_thres - 1, -1, -1):  # 第i次op
            for c in ("inputs", "outputs"):
                for t in profiler.io_info[i][c]:
                    if t in feature_map and t not in DXR.bo:  # 将tensor t换出
                        DXR.bo[t] = 0
                        DXR.Swap[i].append(t)

        return feature_map

    # def Print(profiler:Profiler):
    #     for i in range(len(profiler.io_info)):
    #         print("operation ",i,"-"*20)
    #         if len(DXR.SwapIn[i]):
    #             print("SwapIn")
    #             for t in SwapIn[i]:
    #                 print(f"{t} size:{profiler.tensor_size[t]}  ",end="")
    #         if len(SwapOut[i]):
    #             print("SwapOut")
    #             for t in SwapOut[i]:
    #                 print(f"{t} size:{profiler.tensor_size[t]}  ",end="")


if __name__ == '__main__':
    model = 'MobilenetV1'
    profiler = Profiler(model, 64)
    profiler.load_infos()  # 我把数据都dump下来了，这句话直接读进来即可
    feature_map = Fix_Buffer.Featuremap(profiler)

    Fix_Buffer.oracle(profiler)
    # Print(profiler)
