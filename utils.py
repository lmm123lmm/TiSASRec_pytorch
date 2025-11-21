import sys
import copy #用于对象的深拷贝与浅拷贝
import random
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Process, Queue #多进程

def random_neq(l, r, s):#选取一个不在s中的随机整数
    t = np.random.randint(l, r)#从l到r区间内随机生成一个整数
    while t in s:#如果这个随机生成的整数在集合s中的话
        t = np.random.randint(l, r)#再随机选取一个直到找到一个不在s中的随机整数
    return t

def computeRePos(time_seq, time_span):#构建堆成的相对时间距离矩阵
    #time_seq是输入的时间序列，time_span是最大时间跨度，超越time_span的直接去,time_span其实就是论文里面的k
    #这里的时间矩阵是一个用户的所有行为序列构成的时间矩阵，所以每个用户的时间矩阵大小不一
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i]-time_seq[j])
            if span > time_span:
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = span
    return time_matrix

def Relation(user_train, usernum, maxlen, time_span):#为每个用户都构建一个相对时间关系矩阵
    #user_train：字典，格式为 {user_id: [(item_id, timestamp), ...]}，存储每个用户的历史行为（按时间顺序排列）。
    #usernum：总用户数（假设用户 ID 从 1 到 usernum）。
    #maxlen：模型输入序列的最大长度。如果用户行为太多，只保留最近的 maxlen 个；如果太少，前面补零。
    #time_span：时间差的截断阈值（例如设为 64，表示最大只区分 0~64 的时间间隔，更大的统一视为 64）。
    data_train = dict()
    for user in tqdm(range(1, usernum+1), desc='Preparing relation matrix'):
        time_seq = np.zeros([maxlen], dtype=np.int64)
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):#从后到前填充时间戳，这里排除了最后一个交互对象
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break
        data_train[user] = computeRePos(time_seq, time_span)#把每个用户的相对时间关系矩阵存到这个里面
    return data_train

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, relation_matrix, result_queue, SEED):
    #负采样和数据批处理函数
    #user_train: 用户行为字典，{user_id: [(item, timestamp), ...]}，按时间顺序排列。
    #usernum: 总用户数（ID 从 1 到 usernum）。
    #itemnum: 总物品数（ID 从 1 到 itemnum）。
    #batch_size: 每个批次的样本数。
    #maxlen: 序列最大长度（不足补0，超出截断）。
    #relation_matrix: 预计算好的时间关系矩阵字典，{user: (maxlen, maxlen) matrix}（来自你之前的 Relation 函数）。
    #result_queue: 多进程队列，用于将生成的 batch 发送给主进程
    #SEED: 随机种子，确保该进程内随机可复现
    def sample(user):

        seq = np.zeros([maxlen], dtype=np.int32)#物品时间序列
        time_seq = np.zeros([maxlen], dtype=np.int64)#对应时间戳序列
        pos = np.zeros([maxlen], dtype=np.int32)#正样本（即下一个物品）
        neg = np.zeros([maxlen], dtype=np.int32)#负样本（随机未交互样本）
        nxt = user_train[user][-1][0]#下一个用户交互的物品的itemid
    
        idx = maxlen - 1
        ts = set(map(lambda x: x[0],user_train[user]))#就是把当前user的行为序列的物品id给取出来，放到set中，其实就是去唯一物品id
        for i in reversed(user_train[user][:-1]):#从当前用户的行为序列取出最后一个之后，从后向前开始迭代
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)#如果是有正样本的话就进行负采样，使用random_neq函数
            nxt = i[0]#对于下一个序列的正样本就是当前的样本
            idx -= 1
            if idx == -1: break
        time_matrix = relation_matrix[user]#相对时间间隔矩阵
        return (user, seq, time_seq, time_matrix, pos, neg)
    #返回的格式是，用户id，用户的行为序列[seq0,seq1,seq2,seq3,....,seq(maxlen-1)],对应的时间戳序列[t0,t1,t2,t3,...,t(maxlen-1)]
    #用户这个时间序列对应的时间间隔矩阵，对应的正样本序列[seq1,seq2,seq3,....,seq(maxlen)],负采样序列

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            user = np.random.randint(1, usernum + 1)#随机取样一个用户id
            while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1) #如果这个用户的行为序列小于等于1的话不要，再采样一个另外的用户
            one_batch.append(sample(user))#给他加入到里面
        result_queue.put(zip(*one_batch))#把这个batch的数据放入队列
        #zip(*one_batch)就是(user1, seq1, time_seq1, time_matrix1, pos1, neg1)(user2, seq2, time_seq2, time_matrix2, pos2, neg2)
        #变成了(user1, user2),(seq1, seq2),(time_seq1, time_seq2),(time_matrix1, time_matrix2),(pos1, pos2),(neg1, neg2)

class WarpSampler(object):
    def __init__(self, Users_seq, usernum, itemnum, relation_matrix, batch_size=64, maxlen=10,n_workers=1):
        #Users_seq用户行为序列
        #usernum 用户总数
        #itemnum 物品总数
        #relation——matrix 相对时间间隔矩阵
        #n_workers启动多少子进程进行采样
        self.result_queue = Queue(maxsize=n_workers * 10)#创建一个队列，最大容量为进程数的十倍
        self.processors = []#创建一个空列表用于保存所有子进程对象
        for i in range(n_workers):#启动n_workers个并行的子进程
            self.processors.append(
                Process(target=sample_function, args=(Users_seq,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      relation_matrix,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True#这样设置之后主进程退出，子进程也会结束
            self.processors[-1].start()#启动子进程，每个子进程都会采样数据然后送入队列中

    def next_batch(self):
        return self.result_queue.get()#从队列中取出一个batch——size的元素

    def close(self):#关闭产生batch的进程
        for p in self.processors:
            p.terminate()
            p.join()

# def timeSlice(time_set):#将一组原始时间戳，转化为从0开始的以整数表示的相对时间偏移量
#     time_min = min(time_set)
#     time_map = dict()
#     for time in time_set: # float as map key?
#         time_map[time] = int(round(float(time-time_min)))
#     return time_map
#time_map形状是{t1:t1相对于最小时间的偏移，t2:t2相对于最小时间的偏移}

def cleanAndsort(Users_seq, time_map):#数据处理并排序  把用户物品原始id给映射到内部id
    #User_seq={u1:[[ui1,ut1],[ui2,ut2],......],u2:....}
    #time_map={t1:ts1,t2:ts2,....}
    Users_filted = dict()#存储用户对应的行为序列
    user_set = set()#用于存储用户原始id
    item_set = set()#用于存储物品原始id
    for user, items in Users_seq.items():
        user_set.add(user)
        Users_filted[user] = items
        for item in items:#items的形状是【itemid,timestamp】
            item_set.add(item[0])
    user_map = dict()#用户原始id到从1到n的映射
    item_map = dict()#物品原始id从1到n的映射
    for u_index, user in enumerate(user_set):#把用户原始id映射到1开始的连续id
        user_map[user] = u_index+1
    for i_index, item in enumerate(item_set):#把物品原始id映射到1开始的连续id
        item_map[item] = i_index+1
    
    for user, items in Users_filted.items():#User_filted的形状是{user1:[[item1:timestamp1],[item2:timestamp2],....],....}
        Users_filted[user] = sorted(items, key=lambda x: x[1])#按照时间戳进行排序
    #现在User_filted存储的都是按照时间戳排好序的用户行为序列
    Users_res = dict()#存储处理好的所用数据
    for user, items in Users_filted.items():
        Users_res[user_map[user]] = [[item_map[x[0]],time_map[x[1]]] for x in items]
        #(list(map(lambda x: [item_map[x[0]], time_map[x[1]]], items)))#映射好的用户id：映射好的行为序列
        #现在User_res存储的是映射后，排好序的用户行为序列

    time_max = set()
    for user, items in Users_res.items():
        time_scale=1
        time_list =[x[1] for x in items]
        #list(map(lambda x: x[1], items))
        time_diff = set()
        for i in range(len(time_list)-1):#遍历用户的行为序列的时间戳，然后计算两两之间的差值
            if time_list[i+1]-time_list[i] != 0:
                time_diff.add(time_list[i+1]-time_list[i])
        if len(time_diff)!=0:#如果用户的所有行为都是在一个时间戳进行的话，那就默认这个用户的最小时间间隔为1
            time_scale = min(time_diff)
        time_min = min(time_list)
        Users_res[user] = list(map(lambda x: [x[0], int(round((x[1]-time_min)/time_scale)+1)], items))
        time_max.add(max(set(map(lambda x: x[1], Users_res[user]))))#存储所有用户的一共有多少个不同的时间间隔

    return Users_res, len(user_set), len(item_set), max(time_max)
#User_res:映射之后的用户id：按照时间戳排好序之后的对应的用户行为序列[[映射之后的物品id，按照用户个性化归一之后的时间戳],....]
#用户人数，物品个数，用户最大的时间差值

def data_partition(fname):#fname就是文件存储位置
    #先定义好变量，用户总数，物品总数，
    # 用户字典：用于存储userid：seq[....](按时间戳排序)
    #训练集的用户，验证集的用户，测试集的用户
    interactions=[]#用于存储从数据集中读出来的每一条交互记录（user_id,item_id,timestamp）

    usernum = 0
    itemnum = 0
    Users_seq = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    
    print('Preparing data...')
    f = open('data/%s.txt' % fname, 'r')#从给定的路径中读取文件
    time_set = set()

    user_count = defaultdict(int)#用于记录每个用户有多少记录
    item_count = defaultdict(int)#用于记录每个物品有多少条记录
    #这两个变量用于处理数据过滤
    for line in f:
        try:
            u, i, rating, timestamp = line.rstrip().split('\t')
            #要么格式是，用户id，物品id，评分，时间戳
        except:
            #要么格式是，用户id，物品id，时间戳
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        timestamp=float(timestamp)
        interactions.append((u,i,timestamp))
        user_count[u]+=1
        item_count[i]+=1
    f.close()
    for u,i,timestamp in interactions:
        if user_count[u]>=5 and item_count[i]>=5:
            Users_seq[u].append([i,timestamp])
            time_set.add(timestamp)

    # f = open('data/%s.txt' % fname, 'r') # try?...ugly data pre-processing code
    # for line in f:
    #     try:
    #         u, i, rating, timestamp = line.rstrip().split('\t')
    #     except:
    #         u, i, timestamp = line.rstrip().split('\t')
    #     u = int(u)
    #     i = int(i)
    #     timestamp = float(timestamp)
    #     if user_count[u]<5 or item_count[i]<5: # hard-coded
    #         continue
    #     time_set.add(timestamp)
    #     User[u].append([i, timestamp])
    # f.close()
    #time_map = timeSlice(time_set)#把时间集合给送到时间移动函数中
    #将我们要使用的时间戳集合进行处理，按照最小时间戳进行偏移，得到所有时间戳相对于最小时间戳的相对时间间隔
    time_min=min(time_set)
    time_map=dict()
    for time in time_set:
        time_map[time]=int(round(float(time-time_min)))
    Users_res, usernum, itemnum, timenum = cleanAndsort(Users_seq, time_map)

    for user in Users_res:#开始划分数据集
        nfeedback = len(Users_res[user])
        if nfeedback < 3:#如果一个用户的行为序列小于三个
            user_train[user] = Users_res[user]#那就都划分到训练集中
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = Users_res[user][:-2]
            user_valid[user] = [Users_res[user][-2]]
            user_test[user] = [Users_res[user][-1]]
    print('Preparing done...')
    return [user_train, user_valid, user_test, usernum, itemnum, timenum]


def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)#把传入的dataset给深度拷贝一下
    #dataset就是一个列表，里面有这些东西[user_train, user_valid, user_test, usernum, itemnum, timenum]

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:#如果用户数量大于一万个的话
        users = random.sample(range(1, usernum + 1), 10000)#从所有人中只抽一万个检测
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue#如果这个人的训练集中的行为序列长度小于一又或者他的测试集中的行为序列长度
        #小于一的话，就不用他

        seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1

        #把验证集中的那个物品item作为最近的交互
        seq[idx] = valid[u][0][0]
        time_seq[idx] = valid[u][0][1]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break
        #现在里面就是test的item之前的序列了


        #构建已交互物品集合 rated，用于负采样时避免选到正样本
        rated = set(map(lambda x: x[0],train[u]))
        rated.add(valid[u][0][0])
        rated.add(test[u][0][0])
        rated.add(0)
        item_idx = [test[u][0][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            #从所有物品中选一个
            while t in rated: t = np.random.randint(1, itemnum + 1)#如果选的这个是交互过的正样本的话就重新选
            item_idx.append(t)
        #最后的情况就是item_idx中第一个是正样本，然后后面跟着100个负样本

        time_matrix = computeRePos(time_seq, args.time_span)#根据这个序列构建时间间隔矩阵

        # 👇 关键修改：转为 tensor 并放到 GPU
        u= torch.tensor([u], dtype=torch.long, device=args.device)
        seq= torch.tensor([seq], dtype=torch.long, device=args.device)
        time_matrix= torch.tensor([time_matrix], dtype=torch.long, device=args.device)
        item_idx= torch.tensor(item_idx, dtype=torch.long, device=args.device)

        predictions = -model.predict(u, seq, time_matrix, item_idx)
        # predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix],item_idx]])
        predictions = predictions[0]#因为返回的形状是[1,L]

        rank = predictions.argsort().argsort()[0].item()
        #计算每个元素的排名

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:#每测评完一百个用户就打一个点
            print('.',end='')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break

        rated = set(map(lambda x: x[0], train[u]))
        rated.add(valid[u][0][0])
        rated.add(0)
        item_idx = [valid[u][0][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        time_matrix = computeRePos(time_seq, args.time_span)

        # 👇 关键：转为 PyTorch 张量并放到 GPU
        u= torch.tensor([u], dtype=torch.long, device=args.device)
        seq= torch.tensor([seq], dtype=torch.long, device=args.device)
        time_matrix= torch.tensor([time_matrix], dtype=torch.long, device=args.device)
        item_idx= torch.tensor(item_idx, dtype=torch.long, device=args.device)

        predictions = -model.predict(u, seq, time_matrix, item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.',end='')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
