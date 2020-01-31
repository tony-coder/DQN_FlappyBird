#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import tensorflow as tf
from game import wrapped_flappy_bird as game
import numpy as np
import cv2
import random
from collections import deque

GAME = "flappy bird"
ACTIONS = 2  # 动作数量
ACTION_NAME = ["不动如山", "拍拍翅膀"]

GAMMA = 0.99  # 未来奖励的衰减
EPSILON = 0.01  # 算法每次选择动作的贪心策略值

OBSERVE = 200  # 算法中不训练只进行观测的观测的轮数  训练前先观测OBSERVE轮，积累经验
REPLAY_MEMORY = 50000  # 保存训练记忆的队列的容量

BATCH = 32  # 每次随机抽取训练集的大小
FRAME_PER_ACTION = 1  # 每次动作所用的时间


# 定义神经网络基本方法

# 初始化指定形状的权重w
# 定义权重w
def get_weight(shape):
    # w = tf.truncated_normal(shape, stddev=0.01)
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.01))
    return w


# 定义偏置b 神经网络模型中使用
def get_bias(shape):
    b = tf.Variable(tf.constant(0.01, shape=shape))
    return b


# 定义卷积操作 步长为[1,1,1,1],边界填充
# padding: 全零填充 'SAME'表示使用
# tf.nn.conv2d(输入描述，卷积核描述，核滑动步长，padding)
# eg. yf.nn.conv2d([BATCH(一次喂入多少图片),5,5(分辨率),1(通道数)]，[3,3(行列分辨率),1(通道数)],[3,3(行列分辨率),1(通道数),16(核数),
# [1,1,1,1],padding='VALID')
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


# 定义最大池化操作
# tf.nn.max_pool(输入描述，池化核描述(仅大小)，池化核滑动步长，padding)
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# 搭建神经网络
def createNetwork():
    # 输入层
    # 定义80*80*4的输入层s（处理过的连续4帧的游戏图像）
    s = tf.placeholder('float', [None, 80, 80, 4])

    # 隐藏层

    # 在达到相同感受野的情况下，卷积核越小，所需要的参数和计算量越小
    # 卷积 池化过程
    w_conv1_1 = get_weight([3, 3, 4, 16])  # 第一个卷积核 3*3 4通道 16核
    b_conv1_1 = get_bias([16])  # 偏置b
    # 卷积
    h_conv1_1 = tf.nn.relu(conv2d(s, w_conv1_1) + b_conv1_1)
    # 池化
    h_pool1 = max_pool_2x2(h_conv1_1)

    # 连续两个卷积+一个池化过程
    w_conv2_1 = get_weight([3, 3, 16, 32])
    b_conv2_1 = get_bias([32])
    # 卷积2_1
    h_conv2_1 = tf.nn.relu(conv2d(h_pool1, w_conv2_1) + b_conv2_1)
    w_conv2_2 = get_weight([3, 3, 32, 32])
    b_conv2_2 = get_bias([32])
    h_conv2_2 = tf.nn.relu(conv2d(h_conv2_1, w_conv2_2) + b_conv2_2)
    # 池化
    h_pool2 = max_pool_2x2(h_conv2_2)

    # 连续三个卷积+一个池化过程
    w_conv3_1 = get_weight([3, 3, 32, 64])
    b_conv3_1 = get_bias([64])
    # 卷积3_1
    h_conv3_1 = tf.nn.relu(conv2d(h_pool2, w_conv3_1) + b_conv3_1)
    w_conv3_2 = get_weight([3, 3, 64, 64])
    b_conv3_2 = get_bias([64])
    h_conv3_2 = tf.nn.relu(conv2d(h_conv3_1, w_conv3_2) + b_conv3_2)
    w_conv3_3 = get_weight([3, 3, 64, 64])
    b_conv3_3 = get_bias([64])
    h_conv3_3 = tf.nn.relu(conv2d(h_conv3_2, w_conv3_3) + b_conv3_3)
    # 池化
    h_pool3 = max_pool_2x2(h_conv3_3)  # 10*10*64

    # 扁平化
    # tf.reshape 函数原型为
    # def reshape(tensor, shape, name=None)
    # 第1个参数为被调整维度的张量 第2个参数为要调整为的形状
    # 返回一个shape形状的新tensor
    # 注意shape里最多有一个维度的值可以填写为 - 1，表示自动计算此维度
    h_conv3_flat = tf.reshape(h_pool3, [-1, 6400])  # 1*6400

    # 全连接层 得到小鸟每个动作对应的Q值out
    # 第一层
    w_fc1 = get_weight([6400, 512])
    b_fc1 = get_bias([512])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, w_fc1) + b_fc1)

    # 输出层
    W_fc2 = get_weight([512, ACTIONS])
    b_gc2 = get_bias([ACTIONS])
    out = tf.matmul(h_fc1, W_fc2) + b_gc2  # 第二层输出：结果out 前向传播

    return s, out  # 返回输入和网络输出


# 定义训练过程的函数 反向传播
# s:网络输入 out:网络输出 sess:会话
def trainNetwork(s, out, sess, istrain):
    # 定义损失函数
    x = tf.placeholder(float, [None, ACTIONS])
    y = tf.placeholder(float, [None])
    # tf.reduce_sum 计算一个张量的各个维度上元素的总和
    # api: reduce_sum(input_tensor , axis = None , keep_dims = False , name = None , reduction_indices = None)
    out_action = tf.reduce_sum(tf.multiply(out, x), reduction_indices=1)  # Q估计
    # 计算实际和预测结果的均方误差
    loss = tf.reduce_mean(tf.square(y - out_action))  # Q现实-Q估计
    # 定义反向传播方法
    # 学习率：决定参数每次更新的幅度 1e-6
    train_step = tf.train.AdamOptimizer(1e-6).minimize(loss)  # Adam优化器

    # 初始化游戏环节
    game_state = game.GameState()

    # 定义双向队列保存每轮的训练数据
    # 将每一轮观测存在D中，之后训练从D中随机抽取batch个数据训练，以打破时间连续导致的相关性，保证神经网络训练所需的随机性
    D = deque()

    # 初始化状态并且预处理图片，把连续的4帧图像作为一个输入
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1  # 初始化小鸟动作为不拍动翅膀
    # 将初始状态输入到游戏中 获取相应的反馈: 游戏图像 x_t,动作的奖励 r_0,游戏是否结束的标志 terminal
    x_t, r_0, terminal = game_state.frame_step(do_nothing)

    # 通过cv2模块的resize,cvtColor,threshold 将游戏图片转换为80*80的二值黑白图片
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    # threshold：固定阈值二值化
    # 图像的二值化就是将图像上的像素点的灰度值设置为0或255，这样将使整个图像呈现出明显的黑白效果
    # 图像的二值化使图像中数据量大为减少，从而能凸显出目标的轮廓
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)

    # 将连续4帧的图片作为神经网络的输入
    # np.stack函数是一个用于numpy数组堆叠的函数
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)  # 增加一维，新维度的下标为2 理解为将4张图片堆叠起来 维度变成三维

    # 加载保存的网络参数
    saver = tf.train.Saver()  # 实例化Saver对象
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # 开始训练
    # 初始化贪婪策略值epsilon
    epsilon = EPSILON
    t = 0  # 初始化时间戳
    while True:
        # 根据输入的s_t选择一个动作a_t
        out_t = out.eval(feed_dict={s: [s_t]})[0]  # 将s_t--初始输入作为参数喂入神经网络
        a_t = np.zeros([ACTIONS])  # 选择的动作
        action_index = 0

        # 每隔FRAME_PER_ACTION小鸟选择一次动作
        if t % FRAME_PER_ACTION == 0:
            # 贪心策略 ，有epsilon的几率随机选择动作去探索，否则选取Q值最大的动作
            if random.random() <= epsilon:  # epsilon几率下随机选择动作执行
                print("----------Random Action----------")  # 随机选择
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:  # 否则选取Q值最大的执行
                action_index = np.argmax(out_t)  # 返回最大值所在的索引号
                a_t[action_index] = 1
        else:
            a_t[0] = 1  # do nothing

        # 将选择的动作输入到游戏中，获取下一步游戏图像x_t1_colored，奖励r_t和结果terminal
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # 把这次行为的观测值（输入的图像s_t,执行的动作a_t,得到的奖励r_t，得到的图像s_t1和结果terminal存入队列D中
        D.append((s_t, a_t, r_t, s_t1, terminal))

        # 如果D满了则替换最早的数据
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # 若训练轮数超过观察轮数且istrain=true(允许训练),开始对数据进行训练
        if t > OBSERVE and istrain:
            # 随机抽取minibatch个数据进行训练
            # 从存储器中随机抽取BATCH组数据
            minibatch = random.sample(D, BATCH)

            # 获取BATCH个变量
            s_j_batch = [d[0] for d in minibatch]  # 图像
            a_batch = [d[1] for d in minibatch]  # 动作
            r_batch = [d[2] for d in minibatch]  # 奖励
            s_j1_batch = [d[3] for d in minibatch]  # 得到的图像

            # 估计奖励
            y_batch = []
            out_j1_batch = out.eval(feed_dict={s: s_j1_batch})

            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # 若terminal=true 则游戏结束，奖励值为r_batch[i]
                # 若terminal=false 则游戏继续，奖励值为r_batch[i]加上GAMMA*最大Q值 Q值推导式
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(out_j1_batch[i]))

            # 将估计奖励y_batch，动作a_batch和图像s_j_batch传入train_step进行训练
            # sess.run(train_step, feed_dict={y: y_batch, x: a_batch, s: s_j_batch})
            train_step.run(feed_dict={
                y: y_batch,  # 估计奖励
                x: a_batch,  # 动作
                s: s_j_batch
            })

        # 更新状态
        s_t = s_t1
        t += 1

        # 每1000轮保存一次网络数据
        if t % 1000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step=t)

        # 打印信息
        # 训练轮数、执行的奖励r_t、最大Q值
        print("TIMESTEP ", t, "| ACTION ", ACTION_NAME[action_index], " | REWARD ", r_t, " | Q_MAX %e" % np.max(out_t))


def playGame():
    sess = tf.InteractiveSession()  # 初始化会话
    s, out = createNetwork()

    # 进行观察和训练
    # istrain=True 为训练
    # istrain-False 为观察
    trainNetwork(s, out, sess, istrain=True)


def main():
    playGame()


if __name__ == "__main__":
    main()
