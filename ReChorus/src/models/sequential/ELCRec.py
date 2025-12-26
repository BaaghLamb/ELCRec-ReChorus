""" ELCRec
CMD example:
python main.py `
--model_name ELCRec `
--dataset "Grocery_and_Gourmet_Food" `
--epoch 300 `
--batch_size 256 `
--emb_size 64 `
--num_layers 2 `
--num_heads 4 `
--num_intent_clusters 128 `
--temperature 0.1 `
--contrast_weight 0.1 `
--cluster_weight 0.1 `
--fusion_type add `
--lr 1e-3 `
--l2 1e-6 `
--early_stop 40 `
--history_max 20 `
--num_workers 0 `
--dropout 0.1 `
--save_final_results 1 `
--topk 5,10,20 `
--use_elcm True `
--use_icl True
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseModel import SequentialModel
from utils import layers

class ELCRecBase(object):
    @staticmethod
    def parse_model_args(parser):
        parser = SequentialModel.parse_model_args(parser)
        # 基础参数
        parser.add_argument('--emb_size', type=int, default=64, help='与论文d一致')
        parser.add_argument('--num_layers', type=int, default=2, help='Transformer层数')
        parser.add_argument('--num_heads', type=int, default=4, help='注意力头数')
        # ELCRec专属参数
        parser.add_argument('--num_intent_clusters', type=int, default=256, help='意图聚类数k')
        parser.add_argument('--temperature', type=float, default=0.1, help='对比损失温度')
        parser.add_argument('--contrast_weight', type=float, default=0.1, help='L_icl权重（论文固定0.1）')
        parser.add_argument('--cluster_weight', type=float, default=1.0, help='L_cluster权重α')
        parser.add_argument('--fusion_type', type=str, default='add', help='意图融合方式：add/concat')
        # 消融实验开关
        parser.add_argument('--use_elcm', type=lambda x: (str(x).lower() == 'true'), default=True, help='是否使用ELCM聚类模块')
        parser.add_argument('--use_icl', type=lambda x: (str(x).lower() == 'true'), default=True, help='是否使用ICL对比学习模块')
        return parser

    def _base_init(self, args, corpus):
        self.emb_size = args.emb_size
        self.max_his = args.history_max  # 论文中的T
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.item_num = corpus.n_items
        self.device = args.device
        self.k = args.num_intent_clusters  # 聚类数
        self.temp = args.temperature
        self.contrast_weight = args.contrast_weight
        self.cluster_weight = args.cluster_weight
        self.fusion_type = args.fusion_type
        self.use_elcm = args.use_elcm  # 消融开关：ELCM
        self.use_icl = args.use_icl    # 消融开关：ICL
        
        # 打印配置信息以确认参数正确接收
        print("=" * 60)
        print("ELCRec Configuration:")
        print(f"  use_elcm: {self.use_elcm} (cluster_weight: {self.cluster_weight})")
        print(f"  use_icl: {self.use_icl} (contrast_weight: {self.contrast_weight})")
        print(f"  num_intent_clusters: {self.k}")
        print(f"  fusion_type: {self.fusion_type}")
        print("=" * 60)

        # 可学习聚类中心（论文核心：替换faiss离线聚类）
        # 只有当use_elcm=True时才需要梯度
        requires_grad = self.use_elcm
        self.cluster_centers = nn.Parameter(
            torch.randn(self.k, self.emb_size, device=self.device),
            requires_grad=requires_grad
        )
        nn.init.normal_(self.cluster_centers.data, mean=0.0, std=0.01)
        
        # 保存初始值用于调试
        self.initial_centers = self.cluster_centers.data.clone()

        # 定义网络参数
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)
        self.transformer_block = nn.ModuleList([
            layers.TransformerLayer(
                d_model=self.emb_size,
                d_ff=self.emb_size * 4,  # 论文常用4倍隐藏层
                n_heads=self.num_heads,
                dropout=args.dropout,
                kq_same=False
            ) for _ in range(self.num_layers)
        ])
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight.data, mean=0.0, std=0.01)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    # 实现L_cluster损失
    def compute_cluster_loss(self, behavior_emb):
        """
        behavior_emb: [batch_size, emb_size] 行为嵌入
        return: L_cluster = IntentDecoupling + Intent-behavior Alignment
        """
        # L2归一化
        normed_behavior = F.normalize(behavior_emb, p=2, dim=-1)
        normed_centers = F.normalize(self.cluster_centers, p=2, dim=-1)

        # 第一项：IntentDecoupling（推开不同聚类中心）
        center_sim = torch.matmul(normed_centers, normed_centers.T)  # [k, k]
        intent_decouple = -torch.sum(center_sim) / (self.k * (self.k - 1))  # 论文公式简化

        # 第二项：Intent-behavior Alignment（拉近行为与所有聚类中心）
        behavior_center_sim = torch.matmul(normed_behavior, normed_centers.T)  # [batch_size, k]
        intent_alignment = torch.mean(torch.sum(-behavior_center_sim, dim=1)) / self.k

        return intent_decouple + intent_alignment

    # 实现L_intent_cl损失
    def compute_intent_contrast_loss(self, behavior_emb, cluster_centers):
        """
        behavior_emb: [batch_size, emb_size] 增强视图的行为嵌入
        cluster_centers: [batch_size, emb_size] 对应最近的聚类中心
        return: L_intent_cl
        """
        batch_size = behavior_emb.shape[0]
        # 正样本：行为嵌入与自身最近的聚类中心
        pos_sim = torch.sum(behavior_emb * cluster_centers, dim=-1) / self.temp  # [batch_size]
        # 负样本：行为嵌入与其他所有聚类中心
        all_centers = F.normalize(self.cluster_centers, p=2, dim=-1)  # [k, emb_size]
        neg_sim = torch.matmul(behavior_emb, all_centers.T) / self.temp  # [batch_size, k]
        
        # 计算损失
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [batch_size, 1+k]
        labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)  # 正样本在第0列
        loss = F.cross_entropy(logits, labels)
        return loss

    # 实现L_seq_cl损失
    def compute_seq_contrast_loss(self, view1_emb, view2_emb):
        """序列对比损失：同一序列的两个视图为正样本"""
        batch_size = view1_emb.shape[0]
        # 计算所有 pairwise 相似度
        sim_matrix = torch.matmul(view1_emb, view2_emb.T) / self.temp  # [batch_size, batch_size] - 修复这里
        # 正样本掩码：对角线（自身视图对）
        pos_mask = torch.eye(batch_size, device=self.device)
        # 负样本：非对角线
        neg_mask = 1 - pos_mask
        # 损失计算
        exp_sim = torch.exp(sim_matrix)
        exp_sim = exp_sim * neg_mask
        log_prob = sim_matrix - torch.log(torch.sum(exp_sim, dim=1, keepdim=True))
        loss = -torch.sum(pos_mask * log_prob) / batch_size
        return loss

    # 意图融合（论文两种策略）
    def fuse_intent(self, behavior_emb, intent_center):
        if self.fusion_type == 'add':
            return behavior_emb + intent_center
        elif self.fusion_type == 'concat':
            return torch.cat([behavior_emb, intent_center], dim=-1)

class ELCRec(SequentialModel, ELCRecBase):
    reader = 'ELCRecReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'num_layers', 'num_heads', 'num_intent_clusters', 'fusion_type', 'use_elcm', 'use_icl']

    @staticmethod
    def parse_model_args(parser):
        return ELCRecBase.parse_model_args(parser)

    def __init__(self, args, corpus):
        SequentialModel.__init__(self, args, corpus)
        self._base_init(args, corpus)
        # 若为concat融合，调整输出层维度
        if self.fusion_type == 'concat':
            self.out = nn.Linear(self.emb_size * 2, self.emb_size, bias=True)
        else:
            self.out = nn.Linear(self.emb_size, self.emb_size, bias=True)
            
        # 添加batch计数器用于调试
        self.batch_counter = 0

    def forward(self, feed_dict):
        self.batch_counter += 1
        
        history = feed_dict['history_items']  # [batch_size, max_his]
        lengths = feed_dict['lengths']  # [batch_size]
        batch_size = history.shape[0]

        # 辅助函数：获取行为嵌入（复用论文Transformer编码逻辑）
        def get_behavior_emb(seq, seq_lengths):
            valid_his = (seq > 0).long()
            seq_len = seq.shape[1]
            # 物品嵌入 + 位置嵌入
            his_emb = self.i_embeddings(seq)  # [batch_size, seq_len, emb_size]
            position = (seq_lengths[:, None] - torch.arange(seq_len, device=self.device)[None, :]) * valid_his
            pos_emb = self.p_embeddings(position)
            his_emb = his_emb + pos_emb
            # Transformer编码（因果掩码）
            causality_mask = torch.tril(torch.ones(1, 1, seq_len, seq_len, device=self.device)).bool()
            for block in self.transformer_block:
                his_emb = block(his_emb, causality_mask)
            his_emb = his_emb * valid_his[:, :, None].float()
            # 取最后一个有效行为作为嵌入（论文聚合方式）
            behavior_emb = his_emb[torch.arange(batch_size), seq_lengths - 1, :]  # [batch_size, emb_size]
            return behavior_emb

        # 原始序列行为嵌入（用于推荐损失和聚类损失）
        original_emb = get_behavior_emb(history, lengths)  # [batch_size, emb_size]

        # 训练阶段：计算完整损失
        if 'label' in feed_dict:
            # 推荐损失L_next_item（始终计算）
            item_ids = feed_dict['item_id']  # 训练时：[batch_size]
            item_emb = self.i_embeddings(item_ids)  # [batch_size, emb_size]
            rec_score = torch.sum(original_emb * item_emb, dim=-1)  # [batch_size]
            rec_loss = F.binary_cross_entropy_with_logits(rec_score, feed_dict['label'].float())

            # 对比损失L_icl（仅当use_icl=True时计算）
            icl_loss = 0.0
            if self.use_icl:
                # 获取两个增强视图的行为嵌入
                history1 = feed_dict['history_view1']
                history2 = feed_dict['history_view2']
                lengths1 = (history1 > 0).sum(dim=1)
                lengths2 = (history2 > 0).sum(dim=1)
                view1_emb = get_behavior_emb(history1, lengths1)  # [batch_size, emb_size]
                view2_emb = get_behavior_emb(history2, lengths2)

                # 找到每个行为嵌入最近的聚类中心（论文式(5)）
                normed_centers = F.normalize(self.cluster_centers, p=2, dim=-1)
                view1_dist = torch.matmul(view1_emb, normed_centers.T)  # [batch_size, k]
                view2_dist = torch.matmul(view2_emb, normed_centers.T)
                view1_intent_idx = torch.argmax(view1_dist, dim=1)  # [batch_size]
                view2_intent_idx = torch.argmax(view2_dist, dim=1)
                view1_intent_center = self.cluster_centers[view1_intent_idx]  # [batch_size, emb_size]
                view2_intent_center = self.cluster_centers[view2_intent_idx]

                # 意图融合
                view1_fused = self.fuse_intent(view1_emb, view1_intent_center)
                view2_fused = self.fuse_intent(view2_emb, view2_intent_center)
                # 若concat，通过线性层降维到emb_size
                if self.fusion_type == 'concat':
                    view1_fused = self.out(view1_fused)
                    view2_fused = self.out(view2_fused)

                # 计算对比损失
                seq_cl_loss = self.compute_seq_contrast_loss(view1_fused, view2_fused)
                intent_cl_loss1 = self.compute_intent_contrast_loss(view1_fused, view1_intent_center)
                intent_cl_loss2 = self.compute_intent_contrast_loss(view2_fused, view2_intent_center)
                icl_loss = (seq_cl_loss + intent_cl_loss1 + intent_cl_loss2) / 3  # 平均避免过拟合
                
                # 调试输出：每50个batch打印一次
                if self.batch_counter % 50 == 0:
                    print(f"[Batch {self.batch_counter}] ICL loss: {icl_loss.item():.6f}")

            # 聚类损失L_cluster（仅当use_elcm=True时计算）
            cluster_loss = 0.0
            if self.use_elcm and self.cluster_centers.requires_grad:
                cluster_loss = self.compute_cluster_loss(original_emb)
                
                # 检查聚类中心是否在更新
                if self.batch_counter % 100 == 0:
                    centers_change = (self.cluster_centers.data - self.initial_centers).norm().item()
                    print(f"[Batch {self.batch_counter}] Cluster centers change: {centers_change:.6f}")
                    print(f"[Batch {self.batch_counter}] Cluster loss: {cluster_loss.item():.6f}")

            # 总损失（论文式(8)）
            total_loss = rec_loss + self.contrast_weight * icl_loss + self.cluster_weight * cluster_loss
            
            # 调试输出：每100个batch打印一次汇总
            if self.batch_counter % 100 == 0:
                print(f"\n[Batch {self.batch_counter}] Loss Summary:")
                print(f"  Rec loss: {rec_loss.item():.6f}")
                print(f"  ICL loss (weighted): {self.contrast_weight * icl_loss:.6f}")
                print(f"  Cluster loss (weighted): {self.cluster_weight * cluster_loss:.6f}")
                print(f"  Total loss: {total_loss.item():.6f}")
                print("-" * 50)

            return {'prediction': torch.sigmoid(rec_score), 'loss': total_loss}
        
        # 测试阶段：仅计算推荐分数
        else:
            item_ids = feed_dict['item_id']  # 测试时：[batch_size, k]
            item_emb = self.i_embeddings(item_ids)  # [batch_size, k, emb_size]
            original_emb_expanded = original_emb.unsqueeze(1)  # [batch_size, 1, emb_size]
            rec_score = torch.sum(original_emb_expanded * item_emb, dim=-1)  # [batch_size, k]
            return {'prediction': torch.sigmoid(rec_score)}