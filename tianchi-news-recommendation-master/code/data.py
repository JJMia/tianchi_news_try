### 数据预（处理数据划分）
''' 
---------主要负责--------
1. 读取原始点击日志
2. 按 valid / online 模式切分数据
3. 生成训练需要的中间文件
4. 输出比如：click.pkl，query.pkl

可以理解成整个项目的“原料加工厂”。
'''
import argparse # 命令行参数解析库
import os # 操作系统接口库
import random # 随机数生成库
from random import sample # 随机采样库

import pandas as pd # 数据处理库
from tqdm import tqdm # 进度条库，导入tqdm函数

from utils import Logger # 日志工具库
'''
Logger 是作者自己封装的一个日志工具类，用来：

1. 记录程序运行过程
2. 把重要信息同时输出到控制台和日志文件
3. 方便排查错误、查看训练过程
'''

random.seed(2020) # 设置随机种子
# 给随机数生成器设置一个固定“种子”，让每次运行时产生的随机结果都一样

## 命令行参数
parser = argparse.ArgumentParser(description='数据处理') # 创建命令行参数解析器
# 先准备一个工具，专门用来接收你运行程序时输入的参数。可以在运行时指定一些选项，而非每次都写死在代码里。
# description='数据处理' 只是给这个脚本加个说明
parser.add_argument('--mode', default='valid') # 添加模式参数，默认值为 valid
# 给程序增加一个参数 --mode。可以在命令行里写 --mode valid或 --mode online，不写就默认valid
'''
--mode online：让程序以“线上预测模式”运行，用于生成最终提交结果（真正出提交结果）
--mode valid：让程序以“离线验证模式”运行，用于验证模型效果（模拟线上预测）
因为需要先在本地测试效果。程序会去走验证集流程，一般做这些事：
读取离线点击数据；生成验证用的 click.pkl 和 query.pkl；生成离线召回文件；生成离线特征文件；训练时还会计算指标，比如 hitrate、mrr
'''
parser.add_argument('--logfile', default='test.log') # 添加日志文件参数
# 给程序增加一个参数 --logfile。可以在命令行里写 --logfile test.log，不写就默认test.log
# 用来指定日志文件名，方便查看运行过程
# 日志文件里会记录：程序开始运行；每一步的输出；中间的调试信息；报错信息；训练进度；指标结果
# 这样就不用每次只看终端输出，也可以事后翻日志
args = parser.parse_args() # 解析命令行参数，把在命令行里输入的参数解析出来，存到 args 里。
# 即把命令行里输入的配置--mode和--logfile 收集起来，后面程序要用

mode = args.mode # 获取模式参数
logfile = args.logfile # 获取日志文件参数

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True) # 创建这个日志目录，如果目录已经存在，就不要报错
# os.makedirs(...)：创建文件夹路径。可以一次创建多层目录
log = Logger(f'../user_data/log/{logfile}').logger # 创建日志记录器
# 创建一个日志对象，并把它保存到 log 变量里，后面就可以用它来写日志了。
'''
Logger(f'../user_data/log/{logfile}')：创建一个日志记录器，文件名是 logfile，把日志同时输出到终端和文件，方便调试和保存运行记录
.logger：获取记录器对象，后面可以用它来写日志
Logger(...) 创建出来的是一个自定义对象，.logger 是它里面真正的日志记录器。
即取出这个日志记录器，赋值给 log
这句代码就是：“创建一个日志器，日志文件名用 logfile 指定，然后把它拿来记录程序运行过程。”
'''
log.info(f'数据处理，mode: {mode}') # 记录日志
'''
# log.info(...)：写一条信息级别的日志，通常用来记录一些重要的运行状态、参数等
# f'数据处理，mode: {mode}'：格式化字符串，把 mode 的值插进去
# 这样运行时，你会在终端看到类似这样的输出：
# [2026-07-16 10:00:00] INFO: 数据处理，mode: valid
# 同时也会在 test.log 文件里看到这条记录
'''

## 离线数据处理：从训练集里挑一批用户出来，专门留作离线验证。
# 这样就能模拟“训练完模型后去预测没见过的用户行为”。
def data_offline(df_train_click, df_test_click): # 离线数据处理
    train_users = df_train_click['user_id'].values.tolist()
    # 把 df_train_click 这个表(训练点击数据)里的 user_id 列取出来，转成一个 Python 列表
    # 因为sample() 需要一个列表。
    # 意思是：先拿到所有训练用户,方便后面随机抽样
    '''
    # df_train_click['user_id']：取出点击数据表里的 user_id 这一列。
    # .values：把这一列变成底层的数组形式。作者习惯，先把 Series 变成更“原始”的数据结构，操作更直接
      在 pandas 里：df['col'].values，通常返回的是一个 NumPy 数组，也就是 np.ndarray。
      所以它其实和 numpy 的数组本质上很接近，很多情况下就是 NumPy 数组。
    # .tolist()：再把数组转成普通 Python 列表。
    有时候性能会更好一点，便于后面做 sample() 这种列表操作
    '''
    val_users = sample(train_users, 50000) # 随机采样出一部分样本做验证集
    # 从训练用户里随机抽 50000 个用户,被拿来做：验证集；看模型效果
    # 可能有重复用户，因为一个用户会有多次点击记录。
    log.debug(f'val_users num: {len(set(val_users))}') # 打印验证用户 val_users 里有多少个去重后的唯一用户。
    # 因为 train_users 里可能有重复用户，所以这里用 set() 看唯一用户数量。
    '''
    log.debug(...) ：把一条用于排查问题的详细信息写进日志。
    会记录到日志里,通常也会显示在终端,可以长期保存，方便后面查问题,属于“比较详细的调试信息”
    set() 是 Python 里的集合，会自动去重。
    len(...)：求去重后有多少个不同用户。
    '''

    '''
    这段后面通常会：
    1. 把这些验证用户最后一次点击当作标签
    2. 前面的点击当作历史行为
    3. 形成 query.pkl 和 click.pkl
    这样就能做离线评估了。
    '''

    # 把每个验证用户的最后一次点击拿出来当答案，前面的点击当作历史，构造离线验证集
    '''
    这段的本质是在做一个“时间切分”的离线验证：
    把用户历史点击作为输入，把用户最后一次点击作为预测目标。
    这样就能评估推荐系统在“下一次点击预测”上的效果。
    '''
    click_list = [] # 创建点击列表，存放训练点击记录
    valid_query_list = [] # 创建验证集列表，存放验证集答案

    groups = df_train_click.groupby('user_id') # 把训练集点击数据按用户分开，每个用户一组
    for user_id, g in tqdm(groups): # 遍历用户组
        if user_id in val_users: # 如果这个用户属于验证用户
            valid_query = g.tail(1) # 取出这个用户最后一次点击作为验证集答案
            valid_query_list.append(valid_query[['user_id', 'click_article_id']]) # 把验证集答案添加到验证集列表中
            # 这里的最后一条点击就是：真实要预测的目标，相当于“label”
            train_click = g.head(g.shape[0] - 1) # 取出这个用户除最后一次点击外的所有点击作为训练集点击记录
            # 前面的点击就是：历史行为，相当于“features”
            click_list.append(train_click) # 把训练集点击记录添加到点击列表中
            # 如果用户在验证集里，就把他的最后一次点击拿走，剩下的点击留给训练
            # 这样做是为了模拟真实推荐场景：已知用户以前点击过什么，预测他下一次会点什么
        else:
            click_list.append(g) # 如果这个用户不在验证集，说明这个用户全部点击都保留在训练集中。
            # 把其他用户点击记录添加到点击列表中，用于训练模型

    df_train_click = pd.concat(click_list, sort=False) # 训练用点击历史,把点击列表中的所有点击记录合并成一个数据表
    df_valid_query = pd.concat(valid_query_list, sort=False) # 验证集查询目标,把验证集列表中的所有验证集答案合并成一个数据表

    test_users = df_test_click['user_id'].unique() # 把测试集点击数据按用户分开，每个用户一组
    test_query_list = [] # 创建测试集列表，存放测试集答案

    for user in tqdm(test_users):
        test_query_list.append([user, -1])

    df_test_query = pd.DataFrame(test_query_list,
                                 columns=['user_id', 'click_article_id'])

    df_query = pd.concat([df_valid_query, df_test_query],
                         sort=False).reset_index(drop=True)
    df_click = pd.concat([df_train_click, df_test_click],
                         sort=False).reset_index(drop=True)
    df_click = df_click.sort_values(['user_id',
                                     'click_timestamp']).reset_index(drop=True)

    log.debug(
        f'df_query shape: {df_query.shape}, df_click shape: {df_click.shape}')
    log.debug(f'{df_query.head()}')
    log.debug(f'{df_click.head()}')

    # 保存文件
    os.makedirs('../user_data/data/offline', exist_ok=True)

    df_click.to_pickle('../user_data/data/offline/click.pkl')
    df_query.to_pickle('../user_data/data/offline/query.pkl')


def data_online(df_train_click, df_test_click):
    test_users = df_test_click['user_id'].unique()
    test_query_list = []

    for user in tqdm(test_users):
        test_query_list.append([user, -1])

    df_test_query = pd.DataFrame(test_query_list,
                                 columns=['user_id', 'click_article_id'])

    df_query = df_test_query
    df_click = pd.concat([df_train_click, df_test_click],
                         sort=False).reset_index(drop=True)
    df_click = df_click.sort_values(['user_id',
                                     'click_timestamp']).reset_index(drop=True)

    log.debug(
        f'df_query shape: {df_query.shape}, df_click shape: {df_click.shape}')
    log.debug(f'{df_query.head()}')
    log.debug(f'{df_click.head()}')

    # 保存文件
    os.makedirs('../data/online', exist_ok=True)

    df_click.to_pickle('../user_data/data/online/click.pkl')
    df_query.to_pickle('../user_data/data/online/query.pkl')


if __name__ == '__main__':
    df_train_click = pd.read_csv('../tcdata/train_click_log.csv')
    df_test_click = pd.read_csv('../tcdata/testB_click_log_Test_B.csv')

    log.debug(
        f'df_train_click shape: {df_train_click.shape}, df_test_click shape: {df_test_click.shape}'
    )

    if mode == 'valid':
        data_offline(df_train_click, df_test_click)
    else:
        data_online(df_train_click, df_test_click)
