from collections import defaultdict
from queue import PriorityQueue
import bisect
from collections import defaultdict
from more_itertools import chunked
import json

profile_info = []
tensor_from_opid = {}
tensor_size = {}
resize_info = []
heuristic_info = defaultdict(dict)
heuristic_address = {}
heuristic = defaultdict(list)
redundent_parent = {}  # redundent_resize_group中的元素是(t1, t2)，t1是冗余分配，并且t1的地址等于t2，其中t2是已经分配的
inf = 1e10


def init():
    global profile_info, tensor_from_opid, tensor_size, resize_info, heuristic_info, heuristic_address, heuristic, redundent_parent, inf
    profile_info = []
    tensor_from_opid = {}
    tensor_size = {}
    resize_info = []
    heuristic_info = defaultdict(dict)
    heuristic_address = {}
    heuristic = defaultdict(list)
    redundent_parent = {}
    inf = 1e10


class Allocator:
    def __init__(self):
        pass

    def alloc(self, tid, tsz=None):
        pass

    def free(self, tid):
        pass

    def total_size(self):
        pass


class BufferAllocator(Allocator):
    alignment = 64

    @classmethod
    def aligned_size(cls, size):
        return (size + BufferAllocator.alignment - 1) // BufferAllocator.alignment * BufferAllocator.alignment

    class Node:
        def __init__(self):
            self.parent = None
            self.use_count = 0
            self.size = 0

    def __init__(self):
        super(BufferAllocator, self).__init__()
        self.freelist = defaultdict(list)  # size: [Node]
        self.usedlist = dict()  # tid: node
        self.tot_size = 0

    def alloc(self, tid, tsz=None):
        # print(f'alloc for {tid}')
        node = self.getFromFreelist(tid, tsz)
        if not node:
            node = BufferAllocator.Node()
            if not tsz:
                tsz = tensor_size[tid]
            node.size = BufferAllocator.aligned_size(tsz)
            self.usedlist[tid] = node
            self.tot_size += node.size
            # print(f'alloc from os for {node.size} bytes for tensor[{tid}]')

    def free(self, tid):
        # print(f'free {tid}')
        if tid not in self.usedlist:
            print(tid)
            assert 0
        node = self.usedlist[tid]
        self.usedlist.pop(tid)
        self.returnMemory(node)

    def returnMemory(self, node):
        if not node.parent:
            self.freelist[node.size].append(node)
        else:
            par = node.parent
            par.use_count -= 1
            need_merge = par.use_count == 0
            while need_merge:
                for sz in list(self.freelist.keys()):
                    to_pop = []
                    for i in range(len(self.freelist[sz])):
                        if self.freelist[sz][i].parent == par:
                            to_pop.append(i)
                    for i in to_pop[::-1]:
                        self.freelist[sz].pop(i)
                    if not self.freelist[sz]:
                        self.freelist.pop(sz)
                self.freelist[par.size].append(par)
                need_merge = False
                if par.parent:
                    par = par.parent
                    par.use_count -= 1
                    need_merge = par.use_count == 0

    def getFromFreelist(self, tid, tsz=None):
        keys = sorted(self.freelist.keys())
        key = None
        if not tsz:
            tsz = tensor_size[tid]
        for k in keys:
            if k >= tsz:
                key = k
                break

        if not key:
            return None
        # print(f'match {tensor_size[tid]} with {len(self.freelist[key])} {key} byte segments')
        node = self.freelist[key].pop()
        if not self.freelist[key]:
            self.freelist.pop(key)

        if node.parent:
            node.parent.use_count += 1

        alloc_size = BufferAllocator.aligned_size(tsz)
        if alloc_size >= node.size:
            self.usedlist[tid] = node
            return node
        else:  # split
            node.use_count += 1

            first = BufferAllocator.Node()
            first.parent = node
            first.size = alloc_size

            second = BufferAllocator.Node()
            second.parent = node
            second.size = node.size - alloc_size

            self.usedlist[tid] = first
            self.freelist[second.size].append(second)
            return first

    def free_sizes(self):
        return sorted([sz for sz in self.freelist for _ in range(len(self.freelist[sz]))])

    def used_tids(self):
        return list(self.usedlist.keys())

    def total_size(self):
        return self.tot_size


class OSAllocator(Allocator):
    def __init__(self):
        super(OSAllocator, self).__init__()
        self.cur = 0
        self.peak = 0
        self.tsz = defaultdict(int)

    def alloc(self, tid, size=None):
        if not size:
            return
        self.cur += size
        self.tsz[tid] = size
        self.peak = max(self.cur, self.peak)

    def free(self, tid):
        self.cur -= self.tsz[tid]

    def total_size(self):
        return self.peak


def get_heuristic_info():
    # timeline is [alloc, end)
    for i, info in enumerate(resize_info):
        for a, t in info:
            if a == 'alloc':
                heuristic_info[t]['alloc'] = i
            else:
                heuristic_info[t]['free'] = i + 1
    for i, info in enumerate(profile_info):
        for t in info['outputs']:
            heuristic_info[t]['alloc'] = i
        for t in info['release']:
            heuristic_info[t]['free'] = i + 1
    for t in heuristic_info:
        heuristic_info[t]['size'] = BufferAllocator.aligned_size(tensor_size[t])
        if 'free' not in heuristic_info[t]:
            heuristic_info[t]['free'] = len(profile_info)
        assert all([i in heuristic_info[t] for i in ('alloc', 'free', 'size')])
    print(*heuristic_info.items(), sep='\n')


def heuristic_alloc(model=None):
    # address range is [b, e)
    def overlap(s1, t1, s2, t2):
        return not (t1 <= s2 or t2 <= s1)

    infos = list(sorted(heuristic_info.items(), key=lambda x: [-x[1]['size'], x[1]['alloc'] - x[1]['free'], x[1]['alloc'], x[1]['free']]))
    print(*infos, sep='\n')

    for idx, (t, info) in enumerate(infos):
        print(f'try \t{idx}/{len(infos)}\t{t}\t{info}')
        possible = []
        for op in range(info['alloc'], info['free']):
            if not heuristic[op]:
                possible.append((0, inf))
                continue
            for i in range(len(heuristic[op]) - 1):
                if heuristic[op][i + 1][0] - heuristic[op][i][1] >= info['size']:
                    possible.append((heuristic[op][i][1], heuristic[op][i + 1][0] - heuristic[op][i][1]))
            possible.append((heuristic[op][-1][1], inf))
        possible = sorted(possible, key=lambda x: [x[1], x[0]])  # size更小更靠近0的好
        # print(f'finish get {len(possible)} possible pos')
        for b, _ in possible:
            e = b + info['size']
            ok = True
            for op in range(info['alloc'], info['free']):
                pos = bisect.bisect_left(heuristic[op], (b, e))
                if idx == 308:
                    print('shit', op, pos, heuristic[op])
                # if pos != len(heuristic[op]):
                #     print('fuck', b, e, heuristic[op][pos], heuristic[op])
                if (0 <= pos - 1 < len(heuristic[op]) and overlap(b, e, heuristic[op][pos - 1][0], heuristic[op][pos - 1][1])) or \
                        (0 <= pos < len(heuristic[op]) and overlap(b, e, heuristic[op][pos][0], heuristic[op][pos][1])):
                    ok = False
                    break
                if not ok:
                    break
            if ok:
                print('find', b, e)
                for op in range(info['alloc'], info['free']):
                    bisect.insort_left(heuristic[op], (b, e))
                    if op == 1:
                        print('fuck', op, heuristic[op])
                heuristic_address[t] = b
                break
        assert t in heuristic_address

    for t in redundent_parent:
        par = redundent_parent[t]
        while par in redundent_parent:
            par = redundent_parent[par]
        assert par in heuristic_address and t not in heuristic_address
        heuristic_address[t] = heuristic_address[par]

    max_address = max([heuristic_address[t] + BufferAllocator.aligned_size(tensor_size[t]) for t in heuristic_address])
    with open(f'heuristic/{model}.json', 'w') as f:
        json.dump(heuristic_address, f, indent=4)
    print(max_address)


def profile(model):
    def add_info(ln, tag):
        ln = ln.strip().split(':')[-1].strip().replace('[', '').replace(']', '').strip().split(',')
        tmp = set()
        for item in ln:
            if len(item):
                item = item.strip().replace('(', '').replace(')', '').split()
                tid, tsize = int(item[0]), int(item[1])
                tensor_size[tid] = tsize
                tmp.add(tid)
        profile_info[-1][tag] = list(tmp)

    model_path = {
        'mobilenetv2': '/Users/wangqipeng/Desktop/MNN/build_mac/output_tmp3.txt',
        'alexnet': '/Users/wangqipeng/Desktop/MNN/build/output/output_direct.alexnet.profile.txt',
        'squeezenet': '/Users/wangqipeng/Desktop/MNN/build/output/output_direct.squeezenet.profile.txt',
        'gpath': '/Users/wangqipeng/Desktop/MNN/build_mac/output.direct.googlenet.profile.txt',
        'googlenet': '/Users/wangqipeng/Desktop/MNN/build/output/output_direct.googlenet.profile.txt'
    }
    with open(model_path[model]) as f:
        for idx, line in enumerate(f):
            if line.strip().startswith('current Op'):
                op = line.strip().split()[-1]
                opid = len(profile_info)
                profile_info.append({'op': op, 'id': opid})
            elif line.startswith('\t') and line.strip().startswith('outputs'):
                add_info(line, 'outputs')
                for t in profile_info[-1]['outputs']:
                    tensor_from_opid[t] = len(profile_info) - 1
            elif line.startswith('\t') and line.strip().startswith('release'):
                add_info(line, 'release')
            elif line.startswith('\t') and line.strip().startswith('inputs'):
                add_info(line, 'inputs')
            elif line.startswith('\t') and line.strip().startswith('temporary'):
                add_info(line, 'temporary')
                for t in profile_info[-1]['temporary']:
                    assert t in profile_info[-1]['outputs']
            elif line.strip().startswith('compute'):
                profile_info[-1]['cost'] = float(line.strip().split()[-2])


def resize(model):
    model_path = {
        'squeezenet': '/Users/wangqipeng/Desktop/MNN/build/output/squeezenet.resize.out',
        'googlenet': '/Users/wangqipeng/Desktop/MNN/build/output/googlenet.resize.out',
        'mobilenetv2': '/Users/wangqipeng/Desktop/MNN/build/output/mobilenetv2.resize.out',
        'alexnet': '/Users/wangqipeng/Desktop/MNN/build/output/alexnet.resize.out'
    }
    resize_flag = False
    opid = None
    resize_tid = 0
    compute_flag = False
    freed = None
    with open(model_path[model]) as f:
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
                # print(resize_info)
                assert opid == len(resize_info)
                resize_info.append([])
                freed = set()
            if line.strip().startswith('try get') and resize_flag:
                size = BufferAllocator.aligned_size(int(line.strip().split()[2]))
                rtid = f'{opid}:{resize_tid}'
                tensor_size[rtid] = size
                redundent = None
                for t in freed:
                    if tensor_size[t] == size:
                        freed.remove(t)
                        redundent = t
                        break
                if redundent:
                    for i in range(len(resize_info[-1]) - 1, -1, -1):
                        if resize_info[-1][i][1] == redundent:
                            resize_info[-1].pop(i)
                    redundent_parent[redundent] = rtid
                resize_info[-1].append(('alloc', rtid))
                resize_tid += 1
            if line.strip().startswith('try return') and resize_flag:
                size = BufferAllocator.aligned_size(int(line.strip().split()[2]))
                talloc, tfree = [], []
                for a, tid in resize_info[-1]:
                    if a == 'alloc' and tensor_size[tid] == size:
                        talloc.append(tid)
                    if a == 'free' and tensor_size[tid] == size:
                        tfree.append(tid)
                # print(opid, resize_info[opid], [tensor_size[t] for a, t in resize_info[opid] if a == 'alloc'], size)
                tid = [t for t in talloc if t not in tfree][-1]
                resize_info[-1].append(('free', tid))
                freed.add(tid)
            if line.strip().startswith('finish resize cmd'):
                resize_flag = False
                resize_tid = 0

        print(*resize_info, sep='\n')


def memory_pool():
    buffer_allocator = BufferAllocator()
    # tensor_size[-1] = BufferAllocator.aligned_size(267739136)
    # buffer_allocator.alloc(-1)
    # buffer_allocator.free(-1)
    for i in range(len(profile_info)):
        for t in profile_info[i]['outputs']:
            buffer_allocator.alloc(t)
        for a, t in resize_info[i]:
            if a == 'alloc':
                buffer_allocator.alloc(t)
            else:
                buffer_allocator.free(t)
        for t in profile_info[i]['release']:
            buffer_allocator.free(t)
    print(buffer_allocator.tot_size)


def verify():
    for op in range(len(profile_info)):
        for i in range(len(heuristic[op]) - 1):
            assert heuristic[op][i][1] <= heuristic[op][i + 1][0]


if __name__ == '__main__':
    for model in ['Squeezenet', 'Googlenet', 'MobilenetV1', 'MobilenetV2']:
        init()
        # get_heuristic_info()
        with open(f'heuristic/{model}_4.heuristic.json') as f:
            heuristic_info = json.load(f)
        print('heuristic_alloc', model)
        heuristic_alloc(model)
        print('verify', model)
        verify()
        input()
