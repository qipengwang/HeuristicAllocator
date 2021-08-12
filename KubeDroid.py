from bufferAllocator import BufferAllocator, OSAllocator, Allocator
from tarjan import Tarjan
import os
from collections import defaultdict
import bisect
import json
from queue import Queue
from typing import List, Tuple
import numpy as np
from graphviz import Digraph
import multiprocessing as mp
import math
from ordered_set import OrderedSet

root = '/Users/wangqipeng/Desktop/MNN/build_64/'
models = ['Squeezenet', 'Googlenet', 'MobilenetV1', 'MobilenetV2']


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
        self.num_layers = 0
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

    def profile(self, fpath: str = ''):
        if not fpath:
            fpath = os.path.join(root, 'profile', self.model, f'{self.model}.{self.batch}.profile.out')

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
        with open(fpath) as f:
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

    def resize(self, fpath: str = ''):
        if not fpath:
            fpath = os.path.join(root, 'resize', self.model, f'{self.model}.{self.batch}.resize.out')

        resize_flag = False
        opid = None
        resize_tid = 0
        compute_flag = False
        freed = None
        with open(fpath) as f:
            for line in f:
                if 'start read-map' in line:
                    compute_flag = True
                if 'finish read-map' in line:
                    compute_flag = False
                if not compute_flag:
                    continue

                if line.strip().startswith('current Op is'):
                    opid = int(line.strip().split()[-1].split('th')[0])
                    # print(opid)
                if line.strip().startswith('finish allocate memory for cmd'):
                    resize_flag = True
                    # print(opid, len(self.resize_info))
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

    def cost(self, fpath: str = ''):
        if not fpath:
            fpath = os.path.join(root, 'cost', self.model, f'{self.model}.{self.batch}.cost.out')
        # print(fpath)
        cost_flag = False
        with open(fpath) as f:
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
        # print(os.path.join('data/profiler', self.model, f'{self.model}.io_info.json'), len(self.io_info))
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

    def init_from_scratch(self):
        self.profile()
        self.resize()
        self.cost()
        with open(os.path.join('data/profiler', self.model, f'{self.model}.{self.batch}.tensor_size.txt'), 'w') as f:
            for t, tsz in self.tensor_size.items():
                # print(t, tsz)
                f.write(f'{t}\t{tsz}\n')

    def dump_infos(self):
        with open(os.path.join('data/profiler', self.model, f'{self.model}.io_info.json'), 'w') as f:
            json.dump(self.io_info, f, indent=2)
        with open(os.path.join('data/profiler', self.model, f'{self.model}.{self.batch}.resize_info.json'), 'w') as f:
            json.dump(self.resize_info, f, indent=2)
        with open(os.path.join('data/profiler', self.model, f'{self.model}.{self.batch}.tensor_size.json'), 'w') as f:
            json.dump(self.tensor_size, f, indent=2)
        with open(os.path.join('data/profiler', self.model, f'{self.model}.{self.batch}.redundent_parent.json'), 'w') as f:
            json.dump(self.redundent_parent, f, indent=2)
        with open(f'data/profiler/{self.model}/{self.model}.{self.batch}.cost_info.json', 'w') as f:
            json.dump(self.cost_info, f, indent=2)


class HeuristicAllocator:
    def __init__(self, profiler: Profiler = None):
        self.heuristic_info = defaultdict(dict)
        self.heuristic_address = dict()
        self.profiler = profiler
        self.inf = 1e10
        self.max_address = 0

    def init_heuristic_info(self, heu_info=None):
        if heu_info:
            self.heuristic_info = heu_info
            return

        for i, info in enumerate(self.profiler.resize_info):
            for a, t in info:
                if a == 'alloc':
                    self.heuristic_info[t]['alloc'] = i
                else:
                    self.heuristic_info[t]['free'] = i + 1
        for i, info in enumerate(self.profiler.io_info):
            for t in info['outputs']:
                self.heuristic_info[t]['alloc'] = i
            for t in info['release']:
                self.heuristic_info[t]['free'] = i + 1
        for t in self.heuristic_info:
            self.heuristic_info[t]['size'] = BufferAllocator.aligned_size(self.profiler.tensor_size[t])
            if 'free' not in self.heuristic_info[t]:
                self.heuristic_info[t]['free'] = len(self.profiler.io_info)
            assert all([i in self.heuristic_info[t] for i in ('alloc', 'free', 'size')])
        # print(*self.heuristic_info.items(), sep='\n')

    def dump_heuristic_info(self):
        with open(f'heuristic/{self.profiler.model}_{self.profiler.batch}.heuristic.json', 'w') as f:
            json.dump(self.heuristic_info, f, indent=4)
        with open(f'redundent/{self.profiler.model}_{self.profiler.batch}.redundent.json', 'w') as f:
            json.dump(self.profiler.redundent_parent, f, indent=4)

    def heuristic_alloc(self, pre_alloc: list = None, fn_opt: str = '', dump_result: bool = True):
        # print(f'heuristic_result/allocation/{self.profiler.model}.{self.profiler.batch}{"." + fn_opt if fn_opt else ""}.address.json')
        # return
        fn = f'{self.profiler.model}.{self.profiler.batch}{"." + fn_opt if fn_opt else ""}'
        if not self.heuristic_info:
            self.init_heuristic_info()

        heuristic = defaultdict(list)

        # 用来贪心决策的临时变量，heuristic[op]表示在第op个生命周期，即[op-1, op)区间，当中分配的地址段包含哪些，升序排列 + 方便二分查找
        # list中的元素是若干个[a, b)的区间

        def overlap(s1, t1, s2, t2):  # [s1, t1) & [s2, t2) 是否重叠
            return not (t1 <= s2 or t2 <= s1)

        # infos = list(sorted(self.heuristic_info.items(), key=lambda x: [-x[1]['size'], x[1]['alloc'] - x[1]['free'], x[1]['alloc'], x[1]['free']]))
        # 按照size降序，生命周期降序排列
        infos = list(sorted(self.heuristic_info.items(),
                            key=lambda x: [x[1]['alloc'] - x[1]['free'], -x[1]['size'], x[1]['alloc'], x[1]['free']]))
        print(f'[{fn}]\ttry alloc {len(infos)} info')
        if pre_alloc:
            infos = [(t, self.heuristic_info[t]) for t in pre_alloc] + infos

        # print(*infos, sep='\n')

        for idx, (t, info) in enumerate(infos):
            # if '25:' in str(t):
            #     print(t)
            if idx % 100 == 0:
                print(f'[{fn}]\ttry \t{idx}/{len(infos)}\t{t}\t{info}')
            # if ':' in str(t):
            #     continue
            if t in self.heuristic_address:
                continue

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
                    # print('find', b, e)
                    for op in range(info['alloc'], info['free']):
                        bisect.insort_left(heuristic[op], (b, e))
                        # if op == 1:
                        #     print('debug op == 1', op, heuristic[op])
                    self.heuristic_address[t] = b
                    break
            assert t in self.heuristic_address

        # for t in self.profiler.redundent_parent:
        #     par = self.profiler.redundent_parent[t]
        #     while par in self.profiler.redundent_parent:
        #         par = self.profiler.redundent_parent[par]
        #     print(t, par)
        #     assert par in self.heuristic_address
        #     assert t not in self.heuristic_address
        #     self.heuristic_address[t] = self.heuristic_address[par]
        try:
            self.max_address = max([self.heuristic_address[t] + BufferAllocator.aligned_size(self.heuristic_info[t]['size']) for t in self.heuristic_address])
        except KeyError as e:
            print(e)
        if dump_result:
            with open(f'heuristic/allocation/{self.profiler.model}/{fn}.address.json', 'w') as f:
                json.dump(self.heuristic_address, f, indent=4)
            with open(f'heuristic/allocation/{self.profiler.model}/{fn}.address.txt', 'w') as f:
                f.write(f'maxsize\t{self.max_address}\n')
                for t, addr in self.heuristic_address.items():
                    f.write(f'{t}\t{addr}\n')


def get_heuristic_info_via_exe_seq(exec_seq: list, profiler: Profiler):
    # get heuristic allocate info via execute sequence, return a dict: tid --> info
    # where info is a dict contain size/alloc_t/free_t of the tensor
    # note that the tid is determined based on exec_seq rather than profiler
    # but the info in profiler is still useful to get redundant resize tensor relations
    print('get_heuristic_info_via_exe_seq')
    recomp_seq_info = []
    idx = -1
    op_comp_index = dict()
    for a, op in exec_seq:
        if 'compute' in a:
            idx += 1
            op_comp_index[op] = idx
            recomp_seq_info.append({
                'op': op,
                'alloc': idx,
            })
            if a == 'recompute':
                # print('fuck recompute: ', op, sorted(list(profiler.redundent_parent.keys())))
                for t in list(profiler.redundent_parent.keys()):
                    if t.split(':')[0] == str(op):
                        profiler.redundent_parent[f'{idx}:{t.split(":")[1]}'] = profiler.redundent_parent[t]
        else:
            ith = op_comp_index.pop(op)
            recomp_seq_info[ith]['free'] = len(recomp_seq_info)
    for op, ith in op_comp_index.items():
        recomp_seq_info[ith]['free'] = len(recomp_seq_info)
    # print('==========recomp_seq_info===========')
    # print(*recomp_seq_info, sep='\n')
    heuristic_info = defaultdict(dict)  # tid -> {}
    for idx, info in enumerate(recomp_seq_info):
        # output info
        heuristic_info[idx]['alloc'] = info['alloc']
        heuristic_info[idx]['free'] = info['free']
        heuristic_info[idx]['size'] = profiler.tensor_size[info['op']]
        # resize info
        # continue
        for a, t in profiler.resize_info[info['op']]:
            if a == 'alloc':
                rszid = str(idx) + ':' + t.split(':')[1]
                heuristic_info[rszid]['alloc'] = idx
                heuristic_info[rszid]['free'] = idx + 1
                heuristic_info[rszid]['size'] = profiler.tensor_size[t]
    # print('=========heuristic_info==========')
    # print(*heuristic_info.items(), sep='\n')
    return heuristic_info


def simulate_compute_direct_via_malloc(profiler):
    # simulate buffer allocator
    cur, peak = 0, 0
    for i in range(len(profiler.io_info)):
        # if i > 217:
        #     break
        for t in profiler.io_info[i]['outputs']:
            cur += profiler.tensor_size[t]
            peak = max(cur, peak)
        for a, t in profiler.resize_info[i]:
            if a == 'alloc':
                cur += profiler.tensor_size[t]
                peak = max(cur, peak)
            else:
                cur -= profiler.tensor_size[t]
        for t in profiler.io_info[i]['release']:
            cur -= profiler.tensor_size[t]
    # print(cur, peak)
    return peak


def simulate_via_buffer_allocator(profiler: Profiler):
    buffer_allocator = BufferAllocator()
    for i in range(len(profiler.io_info)):
        for t in profiler.io_info[i]['outputs']:
            buffer_allocator.alloc(t, profiler.tensor_size[t])
        for a, t in profiler.resize_info[i]:
            if a == 'alloc':
                buffer_allocator.alloc(t, profiler.tensor_size[t])
            else:
                buffer_allocator.free(t)
        for t in profiler.io_info[i]['release']:
            buffer_allocator.free(t)
    # print(buffer_allocator.total_size())
    return buffer_allocator.total_size()


def simulate_via_osallocator(profiler: Profiler):
    allocator = OSAllocator()
    for i in range(len(profiler.io_info)):
        for t in profiler.io_info[i]['outputs']:
            allocator.alloc(t, profiler.tensor_size[t])
        for a, t in profiler.resize_info[i]:
            if a == 'alloc':
                allocator.alloc(t, profiler.tensor_size[t])
            else:
                allocator.free(t)
        for t in profiler.io_info[i]['release']:
            allocator.free(t)
    # print(allocator.total_size())
    return allocator.total_size()


def get_featuremap(profiler: Profiler):
    feature_map = set()

    for i in range(profiler.fp_thres):
        for t in profiler.io_info[i]['outputs']:
            feature_map.add(t)
        for t in profiler.io_info[i]['release']:
            feature_map.remove(t)
    # print(sorted(feature_map))
    return feature_map


def get_cutpoint(profiler: Profiler):
    table_in = defaultdict(set)
    for idx, info in enumerate(profiler.io_info):
        for t in info['inputs']:
            table_in[idx].add(profiler.tensor_from_opid[t])
    cut_point = sorted(Tarjan([(op, i) for i in range(profiler.fp_thres + 1) for op in table_in[i] if table_in[op]]))
    return cut_point


def simulate_exec_plan_via_buffer_allocator(exec_plan, profiler: Profiler):
    allocator = BufferAllocator()
    for a, t in exec_plan:
        if t >= len(profiler.io_info):
            continue
        if 'compute' in a:
            allocator.alloc(t, profiler.tensor_size[t])
            for x, y in profiler.resize_info[t]:
                if x == 'alloc':
                    allocator.alloc(y, profiler.tensor_size[y])
                else:
                    allocator.free(y)
        else:
            allocator.free(t)
    # print(allocator.total_size())
    return allocator.total_size()


def simulate_exec_plan_via_osallocator(exec_plan, profiler: Profiler):
    allocator = OSAllocator()
    for a, t in exec_plan:
        if t >= len(profiler.io_info):
            continue
        if 'compute' in a:
            allocator.alloc(t, profiler.tensor_size[t])
            for x, y in profiler.resize_info[t]:
                if x == 'alloc':
                    allocator.alloc(y, profiler.tensor_size[y])
                else:
                    allocator.free(y)
        else:
            allocator.free(t)
    # print(allocator.total_size())
    return allocator.total_size()


def ondemand_recompute_via_msps(profiler: Profiler, mem_budget: int = 512 * 1024 ** 2, threshold: float = 0.5, skip_pre_rel_bp: set = None, skip_rel_fp: set = None):
    """
    :param skip_pre_rel_bp: 在bp阶段不能release的feature-map，比如当前要计算的BP阶段的某个tensor太大了，需要先release一些featuremap，这些不能被release
    :param skip_rel_fp: 在fp阶段非feature-map的不能release的tensor
    :return: None
    """
    if skip_pre_rel_bp is None:
        skip_pre_rel_bp = set()
    if skip_rel_fp is None:
        skip_rel_fp = set()
    feature_map = get_featuremap(profiler)
    msps = {}
    allocated_tensor = set()
    table_in = defaultdict(set)
    table_out = defaultdict(set)
    allocator = OSAllocator()
    exe_seq = []

    for idx, info in enumerate(profiler.io_info):
        for t in info['inputs']:
            table_out[t].add(idx)
            table_in[idx].add(t)
    compute_target = [op for op in range(len(profiler.io_info)) if not table_out[op]]

    def update_msps(ith: int):
        comp_src = set()
        comp_src.add(ith)
        que = Queue()
        for t in table_in[ith]:
            if t not in allocated_tensor:
                que.put(t)
        while not que.empty():
            cur = que.get()
            if cur not in comp_src and cur not in allocated_tensor:
                comp_src.add(cur)
                for t in table_in[cur]:
                    que.put(t)
        comp_t = sum([profiler.cost_info[src] for src in comp_src])
        msps[ith] = profiler.tensor_size[ith] / comp_t
        return

    def compute(ith: int, recompute: bool = False, skip_pre_rel: list = None, bp: bool = False):
        # skip_rel：在当前op计算完成之后如果内存太多了需要release的话，确保skip_rel里面的不会被pre-release
        # skip_rel包含了下一个op计算的input
        # if recompute:
        # print(f'call compute {ith} {recompute}')
        for t in profiler.io_info[ith]['outputs']:
            allocator.alloc(t, profiler.tensor_size[t])
            allocated_tensor.add(t)
        for a, t in profiler.resize_info[ith]:
            if a == 'alloc':
                allocator.alloc(t, profiler.tensor_size[t])
        # compute
        if recompute:
            exe_seq.append(['recompute', ith])
        else:
            exe_seq.append(['compute', ith])
        # print(f'finish {"recompute" if recompute else "compute"} {ith} with allocated={sorted(allocated_tensor)}')

        for a, t in profiler.resize_info[ith]:
            if a == 'free':
                allocator.free(t)
        for t in profiler.io_info[ith]['release']:
            allocator.free(t)
            if t not in allocated_tensor:
                print('fuck', t, sorted(allocated_tensor))
                print(*exe_seq, sep='\n')
            elif bp or t not in skip_rel_fp:
                allocated_tensor.remove(t)
                if recompute:
                    exe_seq.append(['rerelease', t])
                else:
                    exe_seq.append(['release', t])

        while allocator.cur > mem_budget * threshold:
            # print(op, allocated_tensor, sum([profiler.tensor_size[t] for t in allocated_tensor]))
            comped_fms = feature_map & allocated_tensor - set(skip_pre_rel if skip_pre_rel else [])
            if skip_pre_rel_bp and bp:
                comped_fms -= set(skip_pre_rel_bp)
                print(sorted(comped_fms), set(skip_pre_rel_bp))
            if not comped_fms:
                break
            evict_t = None
            for t in comped_fms:
                update_msps(t)
                if not evict_t or msps[evict_t] < msps[t]:
                    evict_t = t
            if not evict_t:
                break
            if evict_t in skip_pre_rel_bp:
                print("pre-release-mdzz")
            exe_seq.append(['pre-release', evict_t])
            # print('add action pre-release', evict_t)
            allocated_tensor.remove(evict_t)
            allocator.free(evict_t)

    def get_recompute_source(ith):
        comp_src = set()
        que = Queue()
        for t in table_in[ith]:
            if t not in allocated_tensor:
                que.put(t)
        while not que.empty():
            cur = que.get()
            if cur not in allocated_tensor and cur not in comp_src:
                comp_src.add(cur)
                for t in table_in[cur]:
                    que.put(t)
        return comp_src

    def adjust_exe_seq():
        to_adjust = []
        for idx, (a, t) in enumerate(exe_seq):
            if 'release' in a:
                to_adjust.append(idx)
        for idx in to_adjust:
            a, t = exe_seq[idx]
            pos = -1
            for i in range(idx, -1, -1):
                if 'compute' in exe_seq[i][0] and (exe_seq[i][1] == t or t in profiler.io_info[exe_seq[i][1]]['inputs']):
                    pos = i + 1
                    break
            assert pos != -1
            exe_seq.pop(idx)
            exe_seq.insert(pos, [a, t])

    allocated_tensor = set()

    for op in range(len(profiler.io_info)):
        comp_src = get_recompute_source(op)
        bp = op >= profiler.fp_thres
        if comp_src:
            # print(f'{op} need to recompute {sorted(comp_src)} first with inputs={sorted(table_in[op])}')
            for i in sorted(list(comp_src)):
                compute(i, True, skip_pre_rel=list(table_in[op]), bp=bp)
        compute(op, skip_pre_rel=list(table_in[op + 1]) if op + 1 < len(profiler.io_info) else None, bp=bp)
        # print(f'finish {op} with allocated-tensors = () with cur={allocator.cur} peak={allocator.peak}')
    adjust_exe_seq()
    # print(exe_seq)
    fn = f'{profiler.model}.{profiler.batch}'
    with open(f'heuristic/execution/{profiler.model}/{fn}.execution.txt', 'w') as f:
        for a, t in exe_seq:
            f.write(f'{a}\t{t}\n')
    with open(f'heuristic/execution/{profiler.model}/{fn}.execution.json', 'w') as f:
        json.dump(exe_seq, f, indent=2)
    # print(allocator.total_size(), mem_budget)
    return exe_seq


def single_search_task(profiler: Profiler, mem_bgt, ba_res, os_res, heu_res):
    exe_seq = ondemand_recompute_via_msps(profiler, mem_budget=mem_bgt * 1024 ** 2, threshold=1.0)
    ba_exe_seq_res = simulate_exec_plan_via_buffer_allocator(exe_seq, profiler)
    oa_exe_seq_res = simulate_exec_plan_via_osallocator(exe_seq, profiler)

    heu_allocator = HeuristicAllocator(profiler)
    heu_info = get_heuristic_info_via_exe_seq(exe_seq, profiler)
    heu_allocator.init_heuristic_info(heu_info)
    heu_allocator.heuristic_alloc(fn_opt=f'{mem_bgt}')
    heu_exe_seq_res = heu_allocator.max_address
    # ba_exe_seq_res = oa_exe_seq_res = heu_exe_seq_res = 0
    with open(f'log/{profiler.model}/{profiler.model}.{profiler.batch}.{mem_bgt}.log', 'w') as f:
        f.write(f'ba_exe_seq_res  ={ba_exe_seq_res:,}\n'
                f'ba_res          ={ba_res:,}\n'
                f'oa_exe_seq_res  ={oa_exe_seq_res:,}\n'
                f'os_res          ={os_res:,}\n'
                f'heu_exe_seq_res ={heu_exe_seq_res:,}\n'
                f'heu_res         ={heu_res:,}\n')
    return


def search_heu_res(model='MobilenetV2', batch=16):
    profiler = Profiler(model, batch)
    profiler.load_infos()
    k = batch // 16
    step = 50 * k
    max_mem_mb = simulate_via_osallocator(profiler) / (1024 ** 2)
    # ba_res = simulate_via_buffer_allocator(profiler)
    # os_res = simulate_via_osallocator(profiler)
    # heu_allocator = HeuristicAllocator(profiler)
    # heu_allocator.heuristic_alloc()
    # heu_res = heu_allocator.max_address
    ba_res = os_res = heu_res = -1

    num_cores = int(mp.cpu_count())
    task_set = range(1000, 5601, 100)
    num_tasks = len(task_set)
    print(f"total: {num_cores} cores {num_tasks} tasks")
    with mp.Pool(min(num_cores, max(1, int(num_cores * 0.5), num_tasks))) as pool:
        results = pool.starmap(single_search_task,
                               [(profiler, mem_bgt, ba_res, os_res, heu_res) for mem_bgt in task_set])
    print(f'finish {model} {batch}')


def sublinear(profiler: Profiler):
    fp_graph = set(range(profiler.fp_thres))
    skip = []
    table_in = defaultdict(set)
    table_out = defaultdict(set)
    for idx, info in enumerate(profiler.io_info):
        for t in info['inputs']:
            opid = profiler.tensor_from_opid[t]
            table_out[opid].add(idx)
            table_in[idx].add(opid)
    for op in list(fp_graph):
        if op not in fp_graph:
            continue
        if not table_out[op]:
            skip.append(op)
            fp_graph.remove(op)
            for t in profiler.io_info[op]['inputs']:
                fp_graph.remove(t)
                skip.append(t)
    cut_point = sorted(Tarjan([(op, i) for i in sorted(fp_graph) for op in table_in[i] if table_in[op]]))  # 防止中间的单个点被当作一个图
    with open(f'sublinear/{profiler.model}/{profiler.model}.sublinear.skip.txt', 'w') as f:
        f.write('\n'.join(map(str, skip)))

    activations = get_featuremap(profiler)
    fm_size = sum([profiler.tensor_size[t] for t in activations])
    budget = int(fm_size / math.sqrt(profiler.num_layers))
    cur, peak = 0, 0
    checkpoints = set()
    for op in range(profiler.fp_thres):
        for t in profiler.io_info[op]['outputs']:
            cur += profiler.tensor_size[t]
        for a, t in profiler.resize_info[op]:
            if a == 'alloc':
                cur += profiler.tensor_size[t]

        if cur > budget and op in cut_point:
            checkpoints.add(op)
            cur = 0
            continue

        for a, t in profiler.resize_info[op]:
            if a == 'free':
                cur -= profiler.tensor_size[t]
        for t in profiler.io_info[op]['release']:
            if not checkpoints or t > max(checkpoints):
                cur -= profiler.tensor_size[t]

    print(sorted(checkpoints))
    with open(f'sublinear/{profiler.model}/{profiler.model}.sublinear.checkpoint.txt', 'w') as f:
        f.write('\n'.join(map(str, sorted(checkpoints))))
    return skip, checkpoints


def capuchin(profiler: Profiler, candidate_ids: set, memory_saving: int, memory_peak: int):
    candidates = []
    recomps = set()
    exe_seq = []
    table_out = defaultdict(set)
    for idx, info in enumerate(profiler.io_info):
        for t in info['inputs']:
            table_out[profiler.tensor_from_opid[t]].add(idx)

    class Tensor:
        def __init__(self, id):
            self.id = id
            self.size = profiler.tensor_size[id]
            self.src = set()
            self.rp_time = 0
            self.ext_time = 0
            self.msps = 0

        def init_msps(self):
            to_comp = set()
            que = Queue()
            que.put(self.id)
            while not que.empty():
                cur = que.get()
                if cur not in to_comp and cur not in candidates:
                    to_comp.add(cur)
                    for t in profiler.io_info[self.id]['inputs']:
                        que.put(t)
            self.rp_time = sum([profiler.cost_info[t] for t in to_comp])
            self.msps = self.size / self.rp_time
            for t in to_comp:
                if t in candidate_ids:
                    self.src.add(t)

        def update_msps(self):
            self.msps = self.size / (self.rp_time + self.ext_time)

    def get_exe_seq():
        allocated_tensor = set()
        recomputed_tensor = OrderedSet()

        def get_recomp_src(inputs: list):
            comp_src = set()
            que = Queue()
            for t in inputs:
                if t not in allocated_tensor:
                    que.put(t)
            while not que.empty():
                cur = que.get()
                if cur not in allocated_tensor and cur not in comp_src:
                    comp_src.add(cur)
                    for t in profiler.io_info[cur]['inputs']:
                        que.put(t)
            return sorted(comp_src)

        for op in range(profiler.fp_thres):
            exe_seq.append(['compute', op])
            allocated_tensor.add(op)
            for t in profiler.io_info[op]['release']:
                exe_seq.append(['release', t])
                allocated_tensor.remove(t)
            for t in profiler.io_info[op]['inputs']:
                if t in recomps and bisect.bisect_right(sorted(table_out[t]), op) == bisect.bisect_right(sorted(table_out[t]), profiler.fp_thres):
                    exe_seq.append(['pre-release', t])
                    # print('pre-release', t)
                    allocated_tensor.remove(t)

        computed_tensor = OrderedSet(sorted(allocated_tensor))
        for op in range(profiler.fp_thres, len(profiler.io_info)):
            to_recomp = get_recomp_src(profiler.io_info[op]['inputs'])
            # if to_recomp:
            #     print(op, 'inputs:', profiler.io_info[op]['inputs'])
            #     print('allocated:', sorted(allocated_tensor))
            #     print(op, 'need recompute', sorted(to_recomp))
            for t in to_recomp:
                exe_seq.append(['recompute', t])
                # print('recompute', t)
                allocated_tensor.add(t)
                recomputed_tensor.add(t)
                computed_tensor.add(t)
                for rel in profiler.io_info[t]['release']:
                    exe_seq.append(['rerelease', rel])
                    # print('rerelease', rel)
                    allocated_tensor.remove(rel)
                    computed_tensor.remove(rel)
                    if rel in recomputed_tensor:
                        recomputed_tensor.remove(rel)
            raw_size = sum([profiler.tensor_size[t] for t in recomputed_tensor])
            released_size = 0
            while released_size * 2 < raw_size:
                flag = False
                for t in list(recomputed_tensor):
                    if t not in profiler.io_info[op]['inputs']:
                        recomputed_tensor.remove(t)
                        allocated_tensor.remove(t)
                        computed_tensor.remove(t)
                        exe_seq.append(['re-release', t])
                        raw_size -= profiler.tensor_size[t]
                        released_size += profiler.tensor_size[t]
                        flag = True
                        break
                if not flag:
                    break
            # if to_recomp:
            #     print('after recompute')
            #     print('recomputed_tensor:', recomputed_tensor)
            #     print('allocated_tensor:', sorted(allocated_tensor))
            exe_seq.append(['compute', op])
            # print('compute', op)
            allocated_tensor.add(op)
            computed_tensor.add(op)
            for t in profiler.io_info[op]['release']:
                exe_seq.append(['release', t])
                allocated_tensor.remove(t)
                computed_tensor.remove(t)
                if t in recomputed_tensor:
                    recomputed_tensor.remove(t)
            cur_mem = sum([profiler.tensor_size[t] for t in allocated_tensor])
            lru = True
            # continue
            while cur_mem > memory_peak:
                # print('candidate_ids:', sorted(candidate_ids))
                lru = False
                # print(op, len(recomputed_tensor), recomputed_tensor)
                if recomputed_tensor:
                    tmp_t = recomputed_tensor[0]
                    exe_seq.append(['lru-release', tmp_t])
                    # print('lru-release', tmp_t)
                    allocated_tensor.remove(tmp_t)
                    computed_tensor.remove(tmp_t)
                    recomputed_tensor.remove(tmp_t)
                    lru = True
                elif computed_tensor:
                    tmp_t = computed_tensor[0]
                    exe_seq.append(['lru-release', tmp_t])
                    # print('lru-release', tmp_t)
                    allocated_tensor.remove(tmp_t)
                    computed_tensor.remove(tmp_t)
                    lru = True
                else:
                    break
                # for i in range(len(recomputed_tensor)):
                #     tmp_t = recomputed_tensor[i]
                #     if tmp_t in candidate_ids:
                #         exe_seq.append(['lru-release', tmp_t])
                #         # print('lru-release', tmp_t)
                #         allocated_tensor.remove(tmp_t)
                #         recomputed_tensor.remove(tmp_t)
                #         # print('lru = True')
                #         lru = True
                #         break
                cur_mem = sum([profiler.tensor_size[t] for t in allocated_tensor])
                # print('fuck', lru)
                # print(sorted(allocated_tensor))
                # print(recomputed_tensor)
                # if not lru:
                #     print('cao', recomputed_tensor, lru)
                #     print(sorted(allocated_tensor))
                #     print(recomputed_tensor)
                #     break
                # lru = False
            # if not lru:
            #     print('shit', lru)
            #     print('allocated_tensor', sorted(allocated_tensor))
            #     print('recomputed_tensor', sorted(recomputed_tensor))

        return

    def adjust_exe_seq():
        to_adjust = []
        for idx, (a, t) in enumerate(exe_seq):
            if 'release' in a and a != 'release':
                to_adjust.append(idx)
        for idx in to_adjust:
            a, t = exe_seq[idx]
            pos = -1
            for i in range(idx, -1, -1):
                if 'compute' in exe_seq[i][0] and (exe_seq[i][1] == t or t in profiler.io_info[exe_seq[i][1]]['inputs']):
                    pos = i + 1
                    break
            assert pos != -1
            exe_seq.pop(idx)
            exe_seq.insert(pos, [a, t])

    # initMSPS
    for cand_id in sorted(candidate_ids):
        cand_t = Tensor(cand_id)
        cand_t.init_msps()
        candidates.append(cand_t)

    while memory_saving > 0:
        # MaxMSPS(candidates)
        max_msps = 0
        t = None
        for cand in candidates:
            if not max_msps or max_msps < cand.msps:
                max_msps = cand.msps
                t = cand
        if not t:
            break
        ext_ct = 1
        for rp in recomps:
            if t in rp.src:
                rp.src.remove(t)
                rp.src = rp.src | t.src
                ext_ct += 1

        recomps.add(t)
        candidates.remove(t)
        memory_saving -= t.size
        # update candidates' MSPS
        for cand in candidates:
            if t in cand.src:
                cand.src.remove(t)
                cand.src = cand.src | rp.src
                cand.rp_time += t.rp_time
                cand.ext_time = 0
                for rp in recomps:
                    if cand in rp.src:
                        cand.ext_time += cand.rp_time
                cand.update_msps()
            if cand in t.src:
                cand.ext_time = ext_ct * cand.rp_time
                cand.update_msps()

    recomps = sorted([rp.id for rp in recomps])
    # print(recomps)
    # return
    get_exe_seq()
    # print(exe_seq)

    allocated = set()
    comp_target = set([i for i in range(len(profiler.io_info)) if not table_out[i]])
    for a, t in exe_seq:
        if 'compute' in a:
            allocated.add(t)
        else:
            allocated.remove(t)
    for t in list(allocated):
        if t not in comp_target:
            exe_seq.append(['adj-release', t])
            allocated.remove(t)
    adjust_exe_seq()

    # print('candidate_ids:', sorted(candidate_ids))
    # print(sorted(recomps), sum([profiler.tensor_size[rp] for rp in recomps]))
    # print('allocated:', sorted(allocated))
    # print('compute target:', sorted(comp_target))
    # print(set(comp_target) - set(allocated))
    # print(*exe_seq, sep='\n', file=open('data/in.txt', 'w'))
    with open(f'capuchin/{profiler.model}/{profiler.model}_{profiler.batch}_{float(memory_peak / 1024 ** 3):.2f}.json', 'w') as f:
        json.dump(exe_seq, f, indent=2)
    with open(f'capuchin/{profiler.model}/{profiler.model}_{profiler.batch}_{float(memory_peak / 1024 ** 3):.2f}.txt', 'w') as f:
        for a, t in exe_seq:
            f.write(f'{a}\t{t}\n')
    return exe_seq


def search_capuchin(model):
    print(model)
    for batch in range(48, 257, 16):
        print(batch)
        profiler = Profiler(model, batch)
        profiler.load_infos()

        table_in, table_out = defaultdict(set), defaultdict(set)
        for idx, info in enumerate(profiler.io_info):
            for t in info['inputs']:
                table_in[idx].add(profiler.tensor_from_opid[t])
                table_out[profiler.tensor_from_opid[t]].add(idx)
        fms = get_featuremap(profiler)
        fm_size = sum([profiler.tensor_size[t] for t in fms])

        for bgt_gb in np.arange(1., 5.6, 0.1):
            memory_budget = int(bgt_gb * 1024 ** 3)
            exe_seq = capuchin(profiler, fms, memory_saving=fm_size - memory_budget, memory_peak=memory_budget)
            # with open(f'capuchin/MobilenetV2/MobilenetV2_{batch}_{bgt_gb:.2f}.json') as f:
            #     exe_seq = json.load(f)
            cost = sum([profiler.cost_info[op] for a, op in exe_seq if 'compute' in a and op < len(profiler.cost_info)])
            # print(cost)
            oares = simulate_exec_plan_via_osallocator(exe_seq, profiler)
            bares = simulate_exec_plan_via_buffer_allocator(exe_seq, profiler)
            # print(bgt_gb, oares, bares)
            print(f'bgt_gb = {bgt_gb:>.2f} GB\toares = {oares / 1024 ** 2:>5.0f} MB\tbares = {bares / 1024 ** 2:>5.0f} MB\tlatency = {cost:>6.0f} ms')
            # return
        print()



if __name__ == '__main__':
    search_heu_res('MobilenetV2')
