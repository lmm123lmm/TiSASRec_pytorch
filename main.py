import os
import time
import torch
import pickle
import argparse

from model import TiSASRec
from tqdm import tqdm
from utils import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--train_dir', required=True)#指定训练结果保存的子目录名称
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=50, type=int)#用户行为序列最大长度，超出截断，不够补零
    parser.add_argument('--hidden_units', default=50, type=int)#隐藏层维度，其实就是embedding的维度
    parser.add_argument('--num_blocks', default=2, type=int)#transformer的block的堆叠层数
    parser.add_argument('--num_epochs', default=201, type=int)
    parser.add_argument('--num_heads', default=1, type=int)#多头注意力多少个头
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.00005, type=float)#对embedding层使用的L2正则化参数
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', default=False, type=str2bool)#只推理选线，如果为True就是直接加载之前预训练好的模型进行推理
    parser.add_argument('--state_dict_path', default=None, type=str)#预训练模型的权重存取地址
    parser.add_argument('--time_span', default=256, type=int)#时间间隔的最大值

    args = parser.parse_args()
    if not os.path.isdir(args.dataset + '_' + args.train_dir):#如果这个目录{dataset}_{train_dir}不存在就创建一个
        os.makedirs(args.dataset + '_' + args.train_dir)
    with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:#往这个文件上写参数
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    f.close()
    #写进去的效果就是这样
    # batch_size, 128
    # dataset, ml - 1
    # m
    # lr, 0.001

    dataset = data_partition(args.dataset)
    #现在dataset就是一个列表，里面有这些东西[user_train, user_valid, user_test, usernum, itemnum, timenum]
    [user_train, user_valid, user_test, usernum, itemnum, timenum] = dataset
    num_batch = len(user_train) // args.batch_size#num_batch就是一共有多少个batch的数据
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    #计算并打印这个用户训练集中平均每个用户有多少个行为

    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')#打开或者创建这个log文件

    try:
        #从data/relation_matrix_{dataset}_{maxlen}_{time_span} 这个文件中读取关系矩阵
        relation_matrix = pickle.load(open('data/relation_matrix_%s_%d_%d.pickle'%(args.dataset, args.maxlen, args.time_span),'rb'))
    except:
        #如果这个文件不存在在的话，就调取Relation函数生成关系矩阵，然后将其序列化保存下来，以便下次使用
        relation_matrix = Relation(user_train, usernum, args.maxlen, args.time_span)
        pickle.dump(relation_matrix, open('data/relation_matrix_%s_%d_%d.pickle'%(args.dataset, args.maxlen, args.time_span),'wb'))

    sampler = WarpSampler(user_train, usernum, itemnum, relation_matrix, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = TiSASRec(usernum, itemnum, timenum, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_uniform_(param.data)
        except:
            pass # just ignore those failed init layers

    model.train() # enable model training

    epoch_start_idx = 1
    if args.state_dict_path is not None:#如果有之间训练好并保存下来的参数的话，就加载之间的参数以及训练轮次
        try:
            model.load_state_dict(torch.load(args.state_dict_path))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)

    if args.inference_only:#如果是只推断不训练的话
        model.eval()#把模型设置成评估模式
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()

    def to_gpu_tensor(data, dtype=torch.long):
        return torch.tensor(np.array(data), dtype=dtype, device=args.device)

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, time_seq, time_matrix, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            time_seq, time_matrix = np.array(time_seq), np.array(time_matrix)
            u,seq,time_matrix,pos,neg=to_gpu_tensor(u),to_gpu_tensor(seq),to_gpu_tensor(time_matrix),to_gpu_tensor(pos),to_gpu_tensor(neg)


            pos_logits, neg_logits = model(u, seq, time_matrix, pos, neg)#得到预测结果
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            #真正的标签

            adam_optimizer.zero_grad()#先梯度清零
            indices = torch.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            #可以改成自动正则化，到时候问问大模型即可
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.abs_pos_K_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.abs_pos_V_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.time_matrix_K_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.time_matrix_V_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs

        if epoch % 20 == 0:#每训练20个epoch就进行一次评估
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
            model.train()

        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'TiSASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))

    f.close()
    sampler.close()
    print("Done")


# 👇 关键：用 if __name__ == '__main__' 保护主入口
if __name__ == '__main__':
    # 可选：如果你以后要打包成 exe，取消下面两行注释
    # from multiprocessing import freeze_support
    # freeze_support()

    main()
