from bufferAllocator import BufferAllocator
from tarjan import Tarjan
import os
from collections import defaultdict
import bisect
import json

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
            self.cost_info=json.load(f)
        for t in list(self.cost_info.keys()):
            self.cost_info[int(t)] = self.cost_info[t]


class HeuristicAllocator:
    def __init__(self, profiler: Profiler):
        self.heuristic_info = defaultdict(dict)
        self.heuristic_address = dict()
        self.profiler = profiler
        self.inf = 1e10
        self.get_heuristic_info()

    def get_heuristic_info(self):
        for i, info in enumerate(profiler.resize_info):
            for a, t in info:
                if a == 'alloc':
                    self.heuristic_info[t]['alloc'] = i
                else:
                    self.heuristic_info[t]['free'] = i + 1
        for i, info in enumerate(profiler.io_info):
            for t in info['outputs']:
                self.heuristic_info[t]['alloc'] = i
            for t in info['release']:
                self.heuristic_info[t]['free'] = i + 1
        for t in self.heuristic_info:
            self.heuristic_info[t]['size'] = BufferAllocator.aligned_size(self.profiler.tensor_size[t])
            if 'free' not in self.heuristic_info[t]:
                self.heuristic_info[t]['free'] = len(profiler.io_info)
            assert all([i in self.heuristic_info[t] for i in ('alloc', 'free', 'size')])
        # print(*self.heuristic_info.items(), sep='\n')

    def dump_heuristic_info(self):
        with open(f'heuristic/{self.profiler.model}_{self.profiler.batch}.heuristic.json', 'w') as f:
            json.dump(self.heuristic_info, f, indent=4)
        with open(f'redundent/{self.profiler.model}_{self.profiler.batch}.redundent.json', 'w') as f:
            json.dump(self.profiler.redundent_parent, f, indent=4)

    def heuristic_alloc(self):
        heuristic = defaultdict(list)  # 用来贪心决策的临时变量，heuristic[op]表示在第op个生命周期，即[op-1, op)区间，当中分配的地址段包含哪些，升序排列 + 方便二分查找

        # list中的元素是若干个[a, b)的区间

        def overlap(s1, t1, s2, t2):  # [s1, t1) & [s2, t2) 是否重叠
            return not (t1 <= s2 or t2 <= s1)

        infos = list(sorted(self.heuristic_info.items(), key=lambda x: [-x[1]['size'], x[1]['alloc'] - x[1]['free'], x[1]['alloc'], x[1]['free']]))
        # 按照size降序，生命周期降序排列
        print(*infos, sep='\n')

        for idx, (t, info) in enumerate(infos):
            print(f'try \t{idx}/{len(infos)}\t{t}\t{info}')
            possible = []
            for op in range(info['alloc'], info['free']):
                if not heuristic[op]:
                    possible.append((0, self.inf))
                    continue
                for i in range(len(heuristic[op]) - 1):
                    if heuristic[op][i + 1][0] - heuristic[op][i][1] >= info['size']:
                        possible.append((heuristic[op][i][1], heuristic[op][i + 1][0] - heuristic[op][i][1]))  # 在第op下可能的位置
                possible.append((heuristic[op][-1][1], self.inf))  # 直接放在最后面
            possible = sorted(possible, key=lambda x: [x[1], x[0]])  # size更小更靠近0的好，最佳分配策略
            # print(f'finish get {len(possible)} possible pos')
            for b, _ in possible:
                e = b + info['size']
                ok = True
                for op in range(info['alloc'], info['free']):
                    pos = bisect.bisect_left(heuristic[op], (b, e))  # 二分lower_bound
                    # if idx == 308:
                    #     print('debug idx==308: ', op, pos, heuristic[op])
                    if (0 <= pos - 1 < len(heuristic[op]) and overlap(b, e, heuristic[op][pos - 1][0], heuristic[op][pos - 1][1])) or \
                            (0 <= pos < len(heuristic[op]) and overlap(b, e, heuristic[op][pos][0], heuristic[op][pos][1])):
                        ok = False
                        break

                if ok:
                    print('find', b, e)
                    for op in range(info['alloc'], info['free']):
                        bisect.insort_left(heuristic[op], (b, e))
                        # if op == 1:
                        #     print('debug op == 1', op, heuristic[op])
                    self.heuristic_address[t] = b
                    break
            assert t in self.heuristic_address

        for t in self.profiler.redundent_parent:
            par = self.profiler.redundent_parent[t]
            while par in self.profiler.redundent_parent:
                par = self.profiler.redundent_parent[par]
            assert par in self.heuristic_address and t not in self.heuristic_address
            self.heuristic_address[t] = self.heuristic_address[par]

        # verify correctness
        for op in heuristic:
            for i in range(len(heuristic[op]) - 1):
                assert heuristic[op][i][1] <= heuristic[op][i + 1][0]

        max_address = max([self.heuristic_address[t] + BufferAllocator.aligned_size(self.profiler.tensor_size[t]) for t in self.heuristic_address])
        print(max_address)
        with open(f'address/{self.profiler.model}_{self.profiler.batch}.address.json', 'w') as f:
            json.dump(self.heuristic_address, f, indent=4)


def selectCheckpoint(profiler: Profiler):
    table_in = defaultdict(set)
    table_out = defaultdict(set)
    indegree = defaultdict(int)
    for idx, info in enumerate(profiler.io_info):
        for t in info['inputs']:
            opid = profiler.tensor_from_opid[t]
            indegree[idx] += 1
            table_out[opid].add(idx)
            table_in[idx].add(opid)
    # handle = [i for i in range(len(d)) if not table_out[i]]
    # print(handle)
    if profiler.model == 'Googlenet':
        fp_thres = 217
    elif profiler.model == 'MobilenetV2':
        fp_thres = 1436
    elif profiler.model == 'MobilenetV1':
        fp_thres = None
    elif profiler.model == 'Squeezenet':
        fp_thres = 119
    else:
        assert 0

    cut_point = sorted(Tarjan([(op, i) for i in range(fp_thres + 1) for op in table_in[i] if table_in[op]]))

    if profiler.model == 'Googlenet':
        while cut_point[0] <= 15:
            cut_point.pop(0)
    elif profiler.model == 'MobilenetV2':
        while cut_point[0] <= 13:
            cut_point.pop(0)
    elif profiler.model == 'Squeezenet':
        while cut_point[-1] >= 101:
            cut_point.pop(-1)
    elif profiler.model == 'MobilenetV1':
        pass
    else:
        assert 0
    print(cut_point)

    for bgt_mb in range(10, 100 + 1, 2):
        end_index = len(cut_point) - 1
        checkpoints = []
        index = len(cut_point) - 2
        mem_budget = bgt_mb * 1024 * 1024
        success = True
        while index >= 0:
            dynamic_allocator = BufferAllocator()
            cur, peak = 0, 0

            for t in checkpoints:
                dynamic_allocator.alloc(t, profiler.tensor_size[t])
                cur += profiler.tensor_size[t]
            peak = max(cur, peak)

            for i in range(cut_point[index], cut_point[end_index]):
                for t in profiler.io_info[i]['outputs']:
                    dynamic_allocator.alloc(t, profiler.tensor_size[t])
                    cur += profiler.tensor_size[t]
                peak = max(cur, peak)

                for a, t in profiler.resize_info[i]:
                    if a == 'alloc':
                        dynamic_allocator.alloc(t, profiler.tensor_size[t])
                        cur += profiler.tensor_size[t]
                    else:
                        dynamic_allocator.free(t)
                        cur -= profiler.tensor_size[t]
                peak = max(cur, peak)

                for t in profiler.io_info[i]['release']:
                    if cut_point[index] <= t < cut_point[end_index]:
                        dynamic_allocator.free(t)
                        cur -= profiler.tensor_size[t]
                peak = max(cur, peak)

            if peak >= mem_budget:
                if end_index == index + 1:
                    success = False
                    break
                # print(f'add {cut_point[index + 1]} as ckpt')
                checkpoints.append(cut_point[index + 1])
                end_index = index + 1
            else:
                index -= 1

        if success:
            print(bgt_mb, '\t', sorted(checkpoints))
        else:
            print(bgt_mb, '\tfail')


def comp_ith(ith, allocator: BufferAllocator, skip=None):
    if skip:
        print(skip)
    for t in profiler.io_info[ith]['outputs']:
        allocator.alloc(t, profiler.tensor_size[t])
        allocated_tensors.add(t)
    for a, t in profiler.resize_info[ith]:
        if a == 'alloc':
            allocator.alloc(t, profiler.tensor_size[t])
        else:
            allocator.free(t)
    for t in profiler.io_info[ith]['release']:
        if t != skip:
            allocator.free(t)
            allocated_tensors.remove(t)


def get_compute_seq():
    pass


def simulate_recompute():
    profiler = Profiler('Googlenet', 4)
    # selectCheckpoint(profiler)
    checkpoints = [0, 18, 25]
    allocator = BufferAllocator()
    currentCheckpointIdx = 0
    allocated_tensors = set()

    table_in = defaultdict(set)
    for idx, info in enumerate(profiler.io_info):
        for t in info['inputs']:
            table_in[idx].add(profiler.tensor_from_opid[t])
    cut_point = sorted(Tarjan([(op, i) for i in range(217 + 1) for op in table_in[i] if table_in[op]]))

    for op in range(len(profiler.io_info)):
        # if op > 217:
        #     break
        comp_ith(op, allocator)

    print(allocator.tot_size)
    print(sorted(list(allocated_tensors & set(cut_point))))

    allocator = BufferAllocator()
    for op in range(len(profiler.io_info)):
        need_recomp = -1
        for t in profiler.io_info[op]['inputs']:
            if t not in allocated_tensors:
                need_recomp = t
        if need_recomp != -1:
            print(op, need_recomp)
            startRecomputeCheckpointIndex = bisect.bisect_left(checkpoints, need_recomp) - 1
            for i in range(checkpoints[startRecomputeCheckpointIndex] + 1, checkpoints[startRecomputeCheckpointIndex + 1]):
                if i > 16:
                    comp_ith(i, allocator)

        comp_ith(op, allocator, checkpoints[currentCheckpointIdx])

        if currentCheckpointIdx + 1 < len(checkpoints) and op == checkpoints[currentCheckpointIdx + 1]:
            for i in range(checkpoints[currentCheckpointIdx] + 1, checkpoints[currentCheckpointIdx + 1]):
                if i <= 16:
                    continue
                for t in profiler.io_info[i]['outputs']:
                    # print(f'release {i} outputs')
                    if t in allocated_tensors:
                        allocator.free(t)
                        allocated_tensors.remove(t)
            currentCheckpointIdx += 1

    print(allocator.tot_size)


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


def baseline(profiler):
    buffer_allocator = BufferAllocator()
    # 下面这个for是执行的流程
    for i in range(len(profiler.io_info)):
        # 对于每次计算，一定是先分配output，然后resize里面的alloc，然后resize里面的free，最后release
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
        # 触发swap操作
    print(buffer_allocator.tot_size)
    return buffer_allocator.tot_size


def get_featuremap(profiler: Profiler):
    feature_map = set()
    for i in range(profiler.fp_thres):
        for t in profiler.io_info[i]['outputs']:
            feature_map.add(t)
        for t in profiler.io_info[i]['release']:
            feature_map.remove(t)
    # print(sorted(feature_map))
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
    model = 'MobilenetV2'
    profiler = Profiler(model, 4)
    profiler.load_infos()  # 我把数据都dump下来了，这句话直接读进来即可
    feature_map = get_featuremap(profiler)
    # todo: simulate swapping
    #       via OS  --> oracle
    #       via buffer_allocator
