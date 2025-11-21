import numpy as np
import torch
import sys

FLOAT_MIN = -sys.float_info.max

class PointWiseFeedForward(torch.nn.Module):#前反馈神经网络，用于数据增强
    def __init__(self, hidden_units, dropout_rate): # wried, why fusion X 2?

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class TimeAwareMultiHeadAttention(torch.nn.Module):#多头时间注意力感知层
    # required homebrewed mha layer for Ti/SASRec experiments
    def __init__(self, hidden_size, head_num, dropout_rate, dev):
        super(TimeAwareMultiHeadAttention, self).__init__()
        self.Q_w = torch.nn.Linear(hidden_size, hidden_size)
        self.K_w = torch.nn.Linear(hidden_size, hidden_size)
        self.V_w = torch.nn.Linear(hidden_size, hidden_size)

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = hidden_size // head_num
        self.dropout_rate = dropout_rate
        self.dev = dev

    def forward(self, queries, keys, time_mask, attn_mask, time_matrix_K, time_matrix_V, abs_pos_K, abs_pos_V):
        #quries查询序列
        #keys键序列
        #time_mask时间序列掩码
        #attn_mask注意力掩码
        #time_matrix_K相对时间间隔矩阵对应的K的embedding矩阵
        #time_matrix_V对应的V的embedding矩阵
        #K的绝对位置，
        #V中的绝对位置
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)#根据输入的序列计算出Q,K,V矩阵

        # head dim * batch dim for parallelization (h*N, T, C/h)
        #多头自注意力的做法，先把每个embedding从维度上拆分为head_num份，每一份带线啊哦都是head_size
        #然后为了能过并行计算每个头的结果，把它们从batch——size维度上进行拼接，这样可以一次并行计算所有头
        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

        #对K时间间隔矩阵，V时间间隔矩阵，K绝对位置矩阵，V绝对位置矩阵也进行相应的变换，以适应多头并行计算
        time_matrix_K_ = torch.cat(torch.split(time_matrix_K, self.head_size, dim=3), dim=0)
        time_matrix_V_ = torch.cat(torch.split(time_matrix_V, self.head_size, dim=3), dim=0)
        abs_pos_K_ = torch.cat(torch.split(abs_pos_K, self.head_size, dim=2), dim=0)
        abs_pos_V_ = torch.cat(torch.split(abs_pos_V, self.head_size, dim=2), dim=0)

        # batched channel wise matmul to gen attention weights
        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))#这是计算Q*K^T
        attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))#这是计算绝对位置Q*(P^K)^T
        attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)#这是计算Q*(R^K)^T
        #由于整个批次的时间间隔矩阵形状是 原来：batch_size*seq_len*seq_len*dimension,经过转化之后就是
        #(batch_size*head_num)*seq_len*seq_len*dimension//head_num

        # seq length adaptive scaling
        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)
        #对自注意力进行归一化

        # key masking, -2^32 lead to leaking, inf lead to nan
        # 0 * inf = nan, then reduce_sum([nan,...]) = nan

        # fixed a bug pointed out in https://github.com/pmixer/TiSASRec.pytorch/issues/2
        # time_mask = time_mask.unsqueeze(-1).expand(attn_weights.shape[0], -1, attn_weights.shape[-1])
        time_mask = time_mask.unsqueeze(-1).repeat(self.head_num, 1, 1)
        time_mask = time_mask.expand(-1, -1, attn_weights.shape[-1])
        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)
        paddings = torch.ones(attn_weights.shape) *  (-2**32+1) # -1e23 # float('-inf')
        paddings = paddings.to(self.dev)
        attn_weights = torch.where(time_mask, paddings, attn_weights) # True:pick padding
        attn_weights = torch.where(attn_mask, paddings, attn_weights) # enforcing causality

        attn_weights = self.softmax(attn_weights) # code as below invalids pytorch backward rules
        # attn_weights = torch.where(time_mask, paddings, attn_weights) # weird query mask in tf impl
        # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/4
        # attn_weights[attn_weights != attn_weights] = 0 # rm nan for -inf into softmax case
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(V_)
        outputs += attn_weights.matmul(abs_pos_V_)
        outputs += attn_weights.unsqueeze(2).matmul(time_matrix_V_).reshape(outputs.shape).squeeze(2)

        # (num_head * N, T, C / num_head) -> (N, T, C)
        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2) # div batch_size

        return outputs


class TiSASRec(torch.nn.Module): # similar to torch.nn.MultiheadAttention
    def __init__(self, user_num, item_num, time_num, args):
        #user_num 用户总数
        #item_num 物品总数
        #time_num 就是时间间隔总数
        #args 就是参数
        super(TiSASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        #在训练时，应该对embedding向量加上L2正则项，正则强度由args.l2_emb控制。
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        #这个hidden_units就是embedding的维度
        self.item_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        #每个embedding层后面都加一个dropout

        self.abs_pos_K_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.abs_pos_V_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        #这个maxlen就是最大序列长度
        self.time_matrix_K_emb = torch.nn.Embedding(args.time_span+1, args.hidden_units)
        self.time_matrix_V_emb = torch.nn.Embedding(args.time_span+1, args.hidden_units)
        #time_span就是最大是时间间隔长度

        self.item_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.abs_pos_K_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.abs_pos_V_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.time_matrix_K_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.time_matrix_V_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        #在标准 Transformer（以及 SASRec/TiSASRec）中，Self-Attention 的 Query (Q) 并不是直接用输入序列，
        #而是先对输入做 Layer Normalization，再作为 Q。
        self.attention_layers = torch.nn.ModuleList()
        #用于存储每一层的自注意力模块
        self.forward_layernorms = torch.nn.ModuleList()
        #用于存储每一层中 Feed-Forward Network（FFN）之前的 LayerNorm
        self.forward_layers = torch.nn.ModuleList()
        #用于存储每一层的前反馈模块

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = TimeAwareMultiHeadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate,
                                                            args.device)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def seq2feats(self, user_ids, log_seqs, time_matrices):
        #把输入的序列转化为每一层要输入的特征
        '''
        :param user_ids: 用户id列表
        :param log_seqs: 用的行为列表
        :param time_matrices: 用户对应是时间间隔矩阵
        :return:
        '''
        seqs = self.item_emb(log_seqs.long())  # 假设 log_seqs 已是 tensor
        #self.item_emb(torch.LongTensor(log_seqs).to(self.dev))#把行为序列中的item_id通过embedding层映射为对应的embedding向量
        seqs *= self.item_emb.embedding_dim ** 0.5
        #进行归一化保持数值稳定性
        seqs = self.item_emb_dropout(seqs)#做一次dropout


        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        #对于batch中的每个行为序列生成相同的序列索引
        positions =torch.LongTensor(positions).to(self.dev)
        abs_pos_K = self.abs_pos_K_emb(positions)
        abs_pos_V = self.abs_pos_V_emb(positions)
        abs_pos_K = self.abs_pos_K_emb_dropout(abs_pos_K)
        abs_pos_V = self.abs_pos_V_emb_dropout(abs_pos_V)

        time_matrices = time_matrices.long().to(self.dev)
        #torch.LongTensor(time_matrices).to(self.dev)
        time_matrix_K = self.time_matrix_K_emb(time_matrices)
        time_matrix_V = self.time_matrix_V_emb(time_matrices)
        time_matrix_K = self.time_matrix_K_dropout(time_matrix_K)
        time_matrix_V = self.time_matrix_V_dropout(time_matrix_V)

        # mask 0th items(placeholder for dry-run) in log_seqs
        # would be easier if 0th item could be an exception for training
        timeline_mask = (log_seqs == 0)#把输入进来的序列的item_id为0的全部设置为True，非0的为False
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim
        #先按位取反，然后在最后增加一个维度，现在是B,L,1 然后seqs现在是B,L,D，让他们广播相乘
        #现在就是只有正常的物品embedidng以及填充的全0embedding了

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        #产生一个L*L的下三角矩阵（包含对角线的位置）为False，这里True表示要遮挡的地方

        for i in range(len(self.attention_layers)):
            # Self-attention, Q=layernorm(seqs), K=V=seqs
            # seqs = torch.transpose(seqs, 0, 1) # (N, T, C) -> (T, N, C)
            Q = self.attention_layernorms[i](seqs) # PyTorch mha requires time first fmt
            mha_outputs = self.attention_layers[i](Q, seqs,
                                            timeline_mask, attention_mask,
                                            time_matrix_K, time_matrix_V,
                                            abs_pos_K, abs_pos_V)
            seqs = Q + mha_outputs
            # seqs = torch.transpose(seqs, 0, 1) # (T, N, C) -> (N, T, C)

            # Point-wise Feed-forward, actually 2 Conv1D for channel wise fusion
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)

        return log_feats 

    def forward(self, user_ids, log_seqs, time_matrices, pos_seqs, neg_seqs): # for training

        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices)
        #这个log_feats其实就是通过transformer预测出来的每个embedding

        pos_embs = self.item_emb(pos_seqs.long().to(self.dev))
        neg_embs = self.item_emb(neg_seqs.long().to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred
    #这个pos_logits就是[第一个相似性分数，第二个相似性分数，。。。。，第n个相似性分数]

    def predict(self, user_ids, log_seqs, time_matrices, item_indices): # for inference
        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices)

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(item_indices.long().to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
