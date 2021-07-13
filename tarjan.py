from collections import defaultdict


def Tarjan(edges):
    link, dfn, low = defaultdict(list), defaultdict(int), defaultdict(int)
    global_time = [0]
    for a, b in edges:
        link[a].append(b)
        link[b].append(a)
        dfn[a], dfn[b] = 0x7fffffff, 0x7fffffff
        low[a], low[b] = 0x7fffffff, 0x7fffffff

    cutting_points = []

    def dfs(cur, prev, root):
        global_time[0] += 1
        dfn[cur], low[cur] = global_time[0], global_time[0]

        children_cnt = 0
        flag = False
        for next in link[cur]:
            if next != prev:
                if dfn[next] == 0x7fffffff:
                    children_cnt += 1
                    dfs(next, cur, root)

                    if cur != root and low[next] >= dfn[cur]:
                        flag = True
                    low[cur] = min(low[cur], low[next])

                else:
                    low[cur] = min(low[cur], dfn[next])

        if flag or (cur == root and children_cnt >= 2):
            cutting_points.append(cur)

    dfs(edges[0][0], None, edges[0][0])
    return cutting_points