# -*- coding: UTF-8 -*-
import logging
import pandas as pd
import torch
import random
from helpers.BaseReader import BaseReader

class ELCRecReader(BaseReader):
    def __init__(self, args):
        super().__init__(args)
        self.max_his = args.history_max  # 只保留max_his，删除model_name相关
        self.aug_strategies = ['crop', 'mask', 'reorder']
        self.mask_ratio = 0.1
        self.reorder_ratio = 0.3
        self._append_his_info()

    def _append_his_info(self):
        logging.info('Appending history info...')
        sort_df = self.all_df.sort_values(by=['time', 'user_id'], kind='mergesort')
        position = list()
        self.user_his = dict()
        for uid, iid, t in zip(sort_df['user_id'], sort_df['item_id'], sort_df['time']):
            if uid not in self.user_his:
                self.user_his[uid] = list()
            position.append(len(self.user_his[uid]))
            self.user_his[uid].append((iid, t))
        sort_df['position'] = position
        for key in ['train', 'dev', 'test']:
            self.data_df[key] = pd.merge(
                left=self.data_df[key], right=sort_df, how='left',
                on=['user_id', 'item_id', 'time'])
        del sort_df

    def _get_feed_dict(self, data, is_train):
        feed_dict = super()._get_feed_dict(data, is_train)
        # 直接删除 self.model_name 判断（Reader只给ELCRec用，训练时必执行增强）
        if is_train:
            history = feed_dict['history_items']  # [batch_size, max_his]
            view1_list, view2_list = [], []
            for seq in history:
                valid_seq = seq[seq != 0].tolist()
                strategy1 = random.choice(self.aug_strategies)
                strategy2 = random.choice(self.aug_strategies)
                v1 = self.augment_sequence(valid_seq, self.max_his, strategy1)
                v2 = self.augment_sequence(valid_seq, self.max_his, strategy2)
                view1_list.append(v1)
                view2_list.append(v2)
            feed_dict['history_view1'] = torch.tensor(view1_list, dtype=torch.long)
            feed_dict['history_view2'] = torch.tensor(view2_list, dtype=torch.long)
        return feed_dict

    def augment_sequence(self, valid_seq, max_his, strategy):
        if len(valid_seq) <= 1:
            pad = [0] * (max_his - len(valid_seq))
            return valid_seq + pad

        if strategy == 'crop':
            start = random.randint(0, len(valid_seq) - 1)
            crop_seq = valid_seq[start:]
            pad = [0] * (max_his - len(crop_seq))
            return (crop_seq + pad)[:max_his]
        
        elif strategy == 'mask':
            mask_num = int(len(valid_seq) * self.mask_ratio)
            mask_pos = random.sample(range(len(valid_seq)), mask_num)
            mask_seq = [0 if i in mask_pos else valid_seq[i] for i in range(len(valid_seq))]
            pad = [0] * (max_his - len(mask_seq))
            return (mask_seq + pad)[:max_his]
        
        elif strategy == 'reorder':
            reorder_len = max(2, int(len(valid_seq) * self.reorder_ratio))
            start = random.randint(0, len(valid_seq) - reorder_len)
            sub_seq = valid_seq[start:start+reorder_len]
            random.shuffle(sub_seq)
            reorder_seq = valid_seq[:start] + sub_seq + valid_seq[start+reorder_len:]
            pad = [0] * (max_his - len(reorder_seq))
            return (reorder_seq + pad)[:max_his]