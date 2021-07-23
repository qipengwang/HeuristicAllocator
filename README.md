# HeuristicAllocator

## 数据

### 输入输出信息io_info

- 位置`os.path.join('data/profiler', self.model, f'{self.model}.io_info.json')`

- 说明

  ```json
  [
      {
      "op": "1th:7:BinaryOp",
      "id": 1,
      "inputs": [
        0
      ],
      "outputs": [
        1
      ],
      "temporary": [
        1
      ],
      "release": [
        0
      ]
    }
  ]
  ```

- 一个list，每个元素是一个dict
  - op表示当前计算操作是什么
  - id表示当前是第几个计算（从0开始）
  - inputs表示输入数据的id（数据称为tensor）
  - outputs表示输出的数据的id
  - release表示完成计算之后释放的数据的id
- Profiler里面有tensor_from_opid表示这个tensor是哪个op产生的（第i个op产生第i个数据，是一一对应的）

### 临时空间信息resize_info

- 位置 `os.path.join('data/profiler', self.model, f'{self.model}.{self.batch}.resize_info.json')`

- sample

  ```json
  [
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [
      [
        "alloc",
        "7:0"
      ],
      [
        "alloc",
        "7:1"
      ],
      [
        "free",
        "7:1"
      ],
      [
        "free",
        "7:0"
      ]
    ]
  ]
  ```

  

- 是一个list，每个元素也是一个list（元素可以为空集）
- resize_info[i]表示第i个op的临时空间信息，里面包含了若干个[action, tensor_id]
  - action包含alloc和free两种
  - tensor_id表示对应的tensor数据的id（冒号前面的是当前对应的第i个op，后面表示的是这是第几个临时空间，当然这个不是很重要）



### 数据大小tensor_size

- 位置 `os.path.join('data/profiler', self.model, f'{self.model}.{self.batch}.tensor_size.json')`

- sample

  ```json
  {
    "7:0": 128,
    "7:1": 64,
    "18:0": 1204224,
    "20:0": 37632,
    "20:1": 14852096,
    "20:2": 6422528,
    "20:3": 14144,
    "21:0": 6422528,
  }
  ```

- 数据是一个dict
  - key是tensor_id
  - value是tensor的大小，单位是Byte
  - 注意json.load之后所有的key是str类型，但是io_info里面产生的output的tensor_id是int类型，需要转换一下



### 每个计算的时间开销cost_info

- 位置 `f'data/profiler/{self.model}/{self.model}.{self.batch}.cost_info.json'`

- sample

  ```json
  {
    "0": 0.177,
    "1": 0.06,
    "2": 0.07,
    "3": 0.071,
    "4": 0.057,
    "5": 0.06,
    "6": 0.07,
    "7": 0.13,
    "8": 0.072,
    "9": 0.069,
    "10": 0.056,
    "11": 0.066,
  }
  ```



- 数据是一个dict
  - key是对应的op，也就是io_info的下标
  - value是计算时间，单位ms
  - 注意json.load之后所有的key是str类型，需要转换一下



## 代码

- 直接用profiler.load_infos()就可以吧需要的数据都读进来了

```python
model = 'MobilenetV2'
profiler = Profiler(model, 4)
profiler.load_infos()  # 我把数据都dump下来了，这句话直接读进来即可
```



- `get_featuremap`得到在前向传播阶段产生的、需要保留在反向阶段使用的tensor
- `oracle` 模拟从OS申请的过程，统计内存的峰值
- `baseline` 表示使用内存池技术，最终的内存峰值是 `BufferAllocator.tot_size`



- `BufferAllocator`
  - `BufferAllocator.alloc(t_id, t_size)`
  - `BufferAllocator.free(t_id)`



## 流程

- 首先为所有的输出分配空间（数据比较特殊，第i个op产生的输出就是第i个tensor）
- 然后为临时变量分配空间
- *计算*
- 释放临时变量的空间
- 释放部分输入数据

```python
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
        # 触发swap操作的位置在每个op执行完成的最后
```





## 模拟swap

### 同步方式

- 半同步
  - 在前向传播阶段，写操作和下一个op的计算并行。然后两个操作结束的时候同步，再进行下下个op的计算
  - 在反向阶段，读入操作在上一个op开始计算的时候触发，然后上一个op结束（也就是当前op开始的时候）和读操作结束的时候同步，然后执行当前的计算
- 完全异步
  - 写操作和计算完全并行，写操作用一个FIFO队列，新的写操作放在FIFO的队尾
  - 读操作也和计算并行，和写操作一个队列（磁盘的IO性能是有限的，就假设同时只能读或者写）
  - 在反向阶段需要数据的时候就计算一下输入的数据读入需要多久，然后选择一个最佳的触发读操作的时间点（某个op计算的开始，保证理论上不产生同步开销的最佳时间点）
  - 读入操作也是用FIFO队列，如果当前op的输入数据还没有读进来，就需要同步
  - 注意
    - 在反向阶段，因为读同步的开销，导致op之间并不是“连续的”，所以计算触发读操作的时间节点上要注意一下，在过程中记录一下每个op的开始结束时间就行



### 内存分配方式

- 通过OS：oracle函数
- 通过内存池：baseline函数



### 注意

- 在前向传播阶段，如果当前的tensor被换出了，那么在前向阶段不再使用的时候就应该被release！！
- 前向阶段的在profiler.fp_thres执行完成之后结束
- 也就是说如果需要用某个tensor作为输入的op都是比fp_thres大的话，那么就可以执行release/free 这个tensor了！！
  - 对于在io_info里面的release里面的tensor可以直接处理
  - 对于feature-map就需要判断一下每个op计算完成之后，如果输入包含了featuremap，那么这个feature-map在前向阶段是否还会被使用，如果不用了就release，节约内存开销
- 然后注意一下在触发swap-in的时候需要先分配空间！就是`buffer_allocator.alloc`或者直接`cur+=***`
- 磁盘IO的速度是：104094472 bytes in 4.452s
- 





