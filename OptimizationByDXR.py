from collections import defaultdict
import bisect
import json

import numpy as np
import matplotlib.pyplot as plt
import math

model = "Squeezenet"
batch = 14
file_name = f'heuristic/{model}_{batch}.heuristic.json'
inf = 1e10  # infinate number
pre_alloc = ["18:0", "239", "4:2"]  # 预先固定的矩形id的list
pre_size = len(pre_alloc)


class OptimizationByDXR:
    class HeuristicAllocator:
        def __init__(self):
            # self.heuristic_info = defaultdict(dict)
            self.heuristic_address = dict()
            self.infos = list()
            self.max_address = inf
            # self.profiler = profiler
            # self.get_heuristic_info()

        def heuristic_alloc(self):

            heuristic = defaultdict(list)  # 用来贪心决策的临时变量，heuristic[op]表示在第op个生命周期，即[op-1, op)区间，当中分配的地址段包含哪些，升序排列 + 方便二分查找

            # list中的元素是若干个[a, b)的区间

            def overlap(s1, t1, s2, t2):  # [s1, t1) & [s2, t2) 是否重叠
                return not (t1 <= s2 or t2 <= s1)

            for idx, (t, info) in enumerate(self.infos):
                # print(f'try \t{idx}/{len(self.infos)}\t{t}\t{info}')
                possible = []
                for op in range(info['alloc'], info['free']):
                    if not heuristic[op]:
                        possible.append((0, inf))
                        continue
                    for i in range(len(heuristic[op]) - 1):
                        if heuristic[op][i + 1][0] - heuristic[op][i][1] >= info['size']:
                            possible.append((heuristic[op][i][1], heuristic[op][i + 1][0] - heuristic[op][i][1]))  # 在第op下可能的位置
                    possible.append((heuristic[op][-1][1], inf))  # 直接放在最后面
                possible = sorted(possible, key=lambda x: [x[1], x[0]])  # size更小更靠近0的好，最佳分配策略

                for b, _ in possible:
                    e = b + info['size']
                    ok = True
                    for op in range(info['alloc'], info['free']):
                        pos = bisect.bisect_left(heuristic[op], (b, e))  # 二分lower_bound

                        if (0 <= pos - 1 < len(heuristic[op]) and overlap(b, e, heuristic[op][pos - 1][0], heuristic[op][pos - 1][1])) or \
                                (0 <= pos < len(heuristic[op]) and overlap(b, e, heuristic[op][pos][0], heuristic[op][pos][1])):
                            ok = False
                            break

                    if ok:
                        # print('find', b, e)
                        for op in range(info['alloc'], info['free']):
                            bisect.insort_left(heuristic[op], (b, e))

                        self.heuristic_address[t] = e  # 改成了e
                        break
                assert t in self.heuristic_address

            max_address = max([self.heuristic_address[t] for t in self.heuristic_address])  # 删去了+ BufferAllocator.aligned_size(self.profiler.tensor_size[t]
            # print("order")
            # for i in range(50):
            #     print(self.infos[i][0],"/",self.infos[i][1]["size"], end="  ")
            print("current: ", max_address)
            self.max_address = max_address

    ############################################################################################################
    @classmethod
    def MainWork(cls):
        def Copy(b, a):  # 用a修改b
            b.__init__()
            for i in a.infos:  # 不能写作y=x,必须infos(list)中的元素将逐一添加到y，heuristic_address是dict类型，dict的修改不会影响原对象
                b.infos.append(i)
            for (i, j) in a.heuristic_address.items():
                b.heuristic_address[i] = j
            b.max_address = a.max_address

        def sample_change(x, y):  # 修改x序列中siz个元素的排列，新的排列记作y
            for i in x.infos:  # 不能写作y=x,必须infos(list)中的元素将逐一添加到y，heuristic_address是dict类型，dict的修改不会影响原对象
                y.infos.append(i)

            siz = int(number * T / 10000)
            siz = min(siz * 2, number)
            change_list = np.random.randint(low=pre_size, high=number, size=siz)  # 前pre_size的排列固定不变
            for i in range(0, siz, 2):
                y.infos[change_list[i]], y.infos[change_list[i + 1]] = y.infos[change_list[i + 1]], y.infos[change_list[i]]
            return y

        def RandomMethod():  # 随机算法
            global T, Tmin, k, t
            T = 1000  # initiate temperature
            Tmin = 10  # minimum value of terperature
            k = 20  # times of internal circulation
            t = 0  # times of external circulation

            x = OptimizationByDXR.HeuristicAllocator()
            Copy(x, optimal)
            while T >= Tmin:
                print(f"Temperature is {T} at {t} times;  optimal is {optimal.max_address}; {int(number * T / 10000)} heuristics moved")
                for i in range(k):
                    y = OptimizationByDXR.HeuristicAllocator()

                    sample_change(x, y)
                    y.heuristic_alloc()

                    if y.max_address < x.max_address:
                        Copy(x, y)
                        Copy(optimal, y)

                t += 1
                # print(t)
                T = 1000 / (1 + t)

        def SimulatedAnnealing():  # 模拟退火
            global T, Tmin, k, t
            T = 1000  # initiate temperature
            Tmin = 10  # minimum value of terperature
            k = 20  # times of internal circulation
            t = 0  # times of external circulation

            x = OptimizationByDXR.HeuristicAllocator()
            Copy(x, optimal)
            # 模拟退火
            while T >= Tmin:
                print(f"Temperature is {T} at {t} times;  optimal is {optimal.max_address}; {int(number * T / 10000)} heuristics moved")
                for i in range(k):
                    y = OptimizationByDXR.HeuristicAllocator()
                    sample_change(x, y)
                    y.heuristic_alloc()

                    if y.max_address <= optimal.max_address:
                        Copy(optimal, y)
                        Copy(x, y)
                    else:
                        # metropolis principle
                        p = math.exp(-(y.max_address - optimal.max_address) / T / 10000)
                        r = np.random.uniform(low=0, high=1)
                        if r < p:
                            Copy(x, y)

                t += 1
                # print(t)
                T = 1000 / (1 + t)

        def PrintOptimal():

            print('-' * 50, "\nmax_address:", optimal.max_address)
            print('-' * 50, "\nheuristic_address:", optimal.heuristic_address)  # 矩形放置的内存位置
            print('-' * 50, "\ninfos:", optimal.infos)  # 矩形的堆叠次序

        ############################################################################################################
        # main_work
        global optimal
        optimal = OptimizationByDXR.HeuristicAllocator()

        file = dict()
        with open(file_name) as f:  # 读取json文件file_name
            file = json.load(f)  # 解析json文件

        for i in range(pre_size):
            file[pre_alloc[i]]['priority'] = i - pre_size  # 排在最前的priority最小

        for x in file.items():
            # if x[0] not in pre_alloc:
            #     x[1]['priority']=0
            if 'priority' not in x[1]:
                file[x[0]]['priority'] = 0

        optimal.infos = list(sorted(file.items(), key=lambda x: [x[1]['priority'], -x[1]['size'], x[1]['alloc'] - x[1]['free'], x[1]['alloc'], x[1]['free']]))
        # print(optimal.infos)
        # 按照size降序，生命周期降序排列
        # print(*infos, sep='\n')
        number = len(optimal.infos)  # sample's size

        optimal.heuristic_alloc()  # 初始result
        SimulatedAnnealing()  # 模拟退火
        RandomMethod()  # 随机算法

        # PrintOptimal()
        optimal.heuristic_alloc()
        PrintOptimal()


############################################################################################################
if __name__ == '__main__':
    OptimizationByDXR.MainWork()
