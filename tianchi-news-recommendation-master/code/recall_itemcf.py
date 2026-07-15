import argparse
import math
import os
import pickle
import random
import signal
from collections import defaultdict
from random import shuffle

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate

# 固定随机种子
random.seed(2020)
np.random.seed(2020)

# 命令行参数
parser = argparse.ArgumentParser(description='itemcf 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')
# 新增：分块大小（控制内存占用）
parser.add_argument('--chunk_size', type=int, default=1000)

args = parser.parse_args()

mode = args.mode
logfile = args.logfile
chunk_size = args.chunk_size  # 每个批次处理的用户数

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'itemcf 召回，mode: {mode}，分块大小: {chunk_size}')


def cal_sim(df):
    """计算物品相似度（保留原逻辑，仅优化内存打印）"""
    user_item_ = df.groupby('user_id')['click_article_id'].agg(
        lambda x: list(x)).reset_index()
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))

    item_cnt = defaultdict(int)
    sim_dict = {}

    # 进度条加描述，方便观察
    for _, items in tqdm(user_item_dict.items(), desc="计算物品共现"):
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            sim_dict.setdefault(item, {})

            for loc2, relate_item in enumerate(items):
                if item == relate_item:
                    continue

                sim_dict[item].setdefault(relate_item, 0)

                # 位置信息权重
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                loc_weight = loc_alpha * (0.9**(np.abs(loc2 - loc1) - 1))

                sim_dict[item][relate_item] += loc_weight / math.log(1 + len(items))

    # 归一化（分批处理，减少内存峰值）
    sim_items = list(sim_dict.keys())
    for idx in tqdm(range(0, len(sim_items), 1000), desc="归一化相似度"):
        batch_items = sim_items[idx:idx+1000]
        for item in batch_items:
            relate_items = sim_dict[item]
            for relate_item, cij in relate_items.items():
                sim_dict[item][relate_item] = cij / math.sqrt(item_cnt[item] * item_cnt[relate_item])

    log.info(f'相似度计算完成，物品数: {len(sim_dict)}')
    return sim_dict, user_item_dict


def recall_single_chunk(df_chunk, item_sim, user_item_dict, worker_id):
    """单批次召回（单进程，避免多进程拷贝）"""
    data_list = []

    for user_id, item_id in tqdm(df_chunk.values, desc=f'批次{worker_id}召回'):
        rank = {}

        if user_id not in user_item_dict:
            continue

        # 取最近点击的2个物品（原逻辑）
        interacted_items = user_item_dict[user_id]
        interacted_items = interacted_items[::-1][:2]

        for loc, item in enumerate(interacted_items):
            # 物品无相似度则跳过，避免KeyError
            if item not in item_sim:
                continue
            # 取前200个相似物品（原逻辑）
            top_relate = sorted(item_sim[item].items(), key=lambda d: d[1], reverse=True)[:200]
            for relate_item, wij in top_relate:
                if relate_item not in interacted_items:
                    rank.setdefault(relate_item, 0)
                    rank[relate_item] += wij * (0.7**loc)

        # 取前100个相似物品
        sim_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:100]
        if not sim_items:
            continue  # 无结果则跳过

        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        df_temp = pd.DataFrame({
            'article_id': item_ids,
            'sim_score': item_sim_scores,
            'user_id': user_id
        })

        # 标记标签（原逻辑）
        if item_id == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)

    # 保存批次结果（避免内存累积）
    if data_list:
        df_data = pd.concat(data_list, ignore_index=True)
        os.makedirs('../user_data/tmp/itemcf', exist_ok=True)
        df_data.to_pickle(f'../user_data/tmp/itemcf/{worker_id}.pkl')
        log.info(f'批次{worker_id}完成，处理用户数: {len(df_chunk)}, 召回结果数: {len(df_data)}')
    else:
        log.info(f'批次{worker_id}无有效结果')


def batch_recall_single_process(df_query, item_sim, user_item_dict):
    """单进程分批次召回（核心修复）"""
    # 清空临时文件夹
    tmp_dir = '../user_data/tmp/itemcf'
    os.makedirs(tmp_dir, exist_ok=True)
    for file_name in os.listdir(tmp_dir):
        if file_name.endswith('.pkl'):
            os.remove(os.path.join(tmp_dir, file_name))

    # 拆分用户为多个批次
    all_users = df_query['user_id'].unique()
    shuffle(all_users)  # 打乱用户顺序（原逻辑保留）
    total_users = len(all_users)
    log.info(f'总用户数: {total_users}，批次大小: {chunk_size}，总批次: {math.ceil(total_users/chunk_size)}')

    # 逐批次处理
    for i in range(0, total_users, chunk_size):
        part_users = all_users[i:i + chunk_size]
        df_temp = df_query[df_query['user_id'].isin(part_users)]
        recall_single_chunk(df_temp, item_sim, user_item_dict, i)

    # 合并所有批次结果
    log.info('开始合并所有批次结果')
    df_data = pd.DataFrame()
    for file_name in tqdm(os.listdir(tmp_dir), desc='合并结果'):
        if file_name.endswith('.pkl'):
            file_path = os.path.join(tmp_dir, file_name)
            df_temp = pd.read_pickle(file_path)
            df_data = pd.concat([df_data, df_temp], ignore_index=True)
            # 删除临时文件，释放磁盘
            os.remove(file_path)

    # 排序（原逻辑保留）
    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                  ascending=[True, False]).reset_index(drop=True)
    return df_data


if __name__ == '__main__':
    # 加载数据（原逻辑保留）
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')
        sim_pkl_file = '../user_data/sim/offline/itemcf_sim.pkl'
        recall_save_path = '../user_data/data/offline/recall_itemcf.pkl'
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')
        sim_pkl_file = '../user_data/sim/online/itemcf_sim.pkl'
        recall_save_path = '../user_data/data/online/recall_itemcf.pkl'

    # 创建目录
    os.makedirs(os.path.dirname(sim_pkl_file), exist_ok=True)
    os.makedirs(os.path.dirname(recall_save_path), exist_ok=True)

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'df_click head: {df_click.head()}')

    # 计算/加载相似度（避免重复计算）
    if os.path.exists(sim_pkl_file):
        log.info(f'加载已保存的相似度文件: {sim_pkl_file}')
        with open(sim_pkl_file, 'rb') as f:
            item_sim = pickle.load(f)
        # 重新构建user_item_dict（避免加载超大文件）
        user_item_ = df_click.groupby('user_id')['click_article_id'].agg(lambda x: list(x)).reset_index()
        user_item_dict = dict(zip(user_item_['user_id'], user_item_['click_article_id']))
    else:
        log.info('开始计算物品相似度（首次运行，耗时较长）')
        item_sim, user_item_dict = cal_sim(df_click)
        # 保存相似度（序列化优化）
        with open(sim_pkl_file, 'wb') as f:
            pickle.dump(item_sim, f, protocol=4)  # protocol=4适配大对象
        log.info(f'相似度文件已保存: {sim_pkl_file}')

    # 单进程分批次召回（核心修复）
    log.info('开始分批次召回（单进程，避免内存溢出）')
    df_data = batch_recall_single_process(df_query, item_sim, user_item_dict)

    # 计算召回指标（原逻辑保留）
    if mode == 'valid':
        log.info(f'计算召回指标')
        valid_df = df_data[df_data['label'].notnull()]
        total_valid_users = df_query[df_query['click_article_id'] != -1]['user_id'].nunique()
        
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            valid_df, total_valid_users)

        log.info(
            f'itemcf 召回指标: \n'
            f'hitrate@5: {hitrate_5:.4f}, mrr@5: {mrr_5:.4f}\n'
            f'hitrate@10: {hitrate_10:.4f}, mrr@10: {mrr_10:.4f}\n'
            f'hitrate@20: {hitrate_20:.4f}, mrr@20: {mrr_20:.4f}\n'
            f'hitrate@40: {hitrate_40:.4f}, mrr@40: {mrr_40:.4f}\n'
            f'hitrate@50: {hitrate_50:.4f}, mrr@50: {mrr_50:.4f}'
        )

    # 保存最终结果
    df_data.to_pickle(recall_save_path)
    log.info(f'召回完成，结果保存至: {recall_save_path}，最终数据形状: {df_data.shape}')