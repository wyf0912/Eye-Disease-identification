from skimage import io, transform
import glob
import os
import tensorflow as tf
import numpy as np
import time
import math

# 数据集地址
path = 'E:/校级创新项目/datasets/'
# 模型保存地址
model_path = './saved_model/model.ckpt'

# 将所有的图片resize成100*100
w = 200
h = 200
c = 3
kind=4 #分类的种类
learning_rate=0.001 #before0.001


# 计算全链接神经网络的节点数，输入图片的长宽像素
def cal_nodes(w, h):
    for i in range(4):
        w = math.ceil((w - 1) / 2)
        h = math.ceil((h - 1) / 2)
    return w * h


nodes_num = cal_nodes(w, h)


def read_img(path):
    "读取图片和标签 并写为tfrecord格式"
    tfrecords_filename = './tfdata/data.tfrecods'
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            print('reading the images:%s' % (im))
            img = io.imread(im)
            if img.shape != (w, h, 3):
                img = transform.resize(img, (w, h), mode='constant')
                io.imsave(im, img)
            imgs.append(img)
            labels.append(idx)
            # print(labels)
            img_str = img.tostring()
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[idx])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_str]))
                }))
            writer.write(example.SerializeToString())
    writer.close()
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


def read_test_img(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            print('reading the images:%s' % (im))
            img = io.imread(im)
            if img.shape != (w, h, 3):
                img = transform.resize(img, (w, h))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


# -----------------构建网络----------------------
x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')


# 当不使用全0填充时
# out_length=向上取整（in_length-fil_length+1）/slid_length)

def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [5, 5, 3, 32],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))  # 卷积层尺寸5*5*3 深度32
        conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))  # 偏置项深度32
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')  # strides为过滤器的步长
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))  # Relu激活函数去线性化 使用tf.nn.add保证维数一样

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable("weight", [5, 5, 32, 64],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope("layer5-conv3"):
        conv3_weights = tf.get_variable("weight", [3, 3, 64, 128],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    with tf.name_scope("layer6-pool3"):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope("layer7-conv4"):
        conv4_weights = tf.get_variable("weight", [3, 3, 128, 128],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

    with tf.name_scope("layer8-pool4"):
        pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        nodes = nodes_num * 128  # 原来是6*6*128
        reshaped = tf.reshape(pool4, [-1, nodes])

    with tf.variable_scope('layer9-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, 1024],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer10-fc2'):
        fc2_weights = tf.get_variable("weight", [1024, 512],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))

        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train: fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('layer11-fc3'):
        fc3_weights = tf.get_variable("weight", [512, kind],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [kind], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases

    return logit


# ---------------------------网络结束---------------------------
regularizer = tf.contrib.layers.l2_regularizer(0.0001)
logits = inference(x, False, regularizer)

# (小处理)将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
b = tf.constant(value=1, dtype=tf.float32)
logits_eval = tf.multiply(logits, b, name='logits_eval')

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


# 训练和测试数据
# config = tf.ConfigProto(device_count = {'GPU': 0})
n_epoch = 10
batch_size = 40  # CPU时是64 GPU10

# tensorboard可视化部分
trainloss = tf.Variable(0.0)
trainacc = tf.Variable(0.0)
valloss = tf.Variable(0.0)
valacc = tf.Variable(0.0)
tf.summary.scalar('train loss', trainloss)
tf.summary.scalar('train acc', trainacc)
tf.summary.scalar('valid loss', valloss)
tf.summary.scalar('valid acc', valacc)
mergeall = tf.summary.merge_all()
saver = tf.train.Saver()


def read_tfrecord():
    filename_queue = tf.train.string_input_producer(['./tfdata/data.tfrecods'])
    reader = tf.TFRecordReader()
    # 从文件中读出一个样例
    _, serialized_example = reader.read(filename_queue)
    # 解析读入的一个样例
    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'img_raw': tf.FixedLenFeature([], tf.string)
    })
    # 将字符串解析成图像对应的像素数组
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)

    image.set_shape([w * h * 3])
    image = tf.reshape(image, [w, h, 3])

    return image, label


def shuffle():
    '''读取数据打乱顺序'''
    data, label = read_img(path)
    # 打乱顺序
    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data = data[arr]
    label = label[arr]

    # 将所有数据分为训练集和验证集
    ratio = 0.8
    s = np.int(num_example * ratio)
    x_train = data[:s]
    y_train = label[:s]
    x_val = data[s:]
    y_val = label[s:]
    return x_train, y_train, x_val, y_val


def train(continue_train=False):
    x_train, y_train, x_val, y_val = shuffle()
    sess = tf.Session()
    writer = tf.summary.FileWriter("./tensorboard_data", sess.graph)
    '''
    img,label=read_tfrecord()
    x_train, y_train = tf.train.shuffle_batch([img, label],
                                                    batch_size=30, capacity=2000,
                                                    min_after_dequeue=1000)
    x_val=x_train
    y_val=y_train
    '''
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epoch):
        start_time = time.time()
        # training
        train_loss, train_acc, n_batch = 0, 0, 0
        for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
            _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
            train_loss += err;
            train_acc += ac;
            n_batch += 1
        trainloss_ = (np.sum(train_loss) / n_batch)
        trainacc_ = (np.sum(train_acc) / n_batch)
        print("epoch:",epoch)
        print("train loss:", trainloss_)
        print("train acc:", trainacc_, "\n")

        # validation
        val_loss, val_acc, n_batch = 0, 0, 0
        for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
            err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
            val_loss += err;
            val_acc += ac;
            n_batch += 1
        valloss_ = (np.sum(val_loss) / n_batch)
        valacc_ = (np.sum(val_acc) / n_batch)

        print("validation loss:", valloss_)
        print("validation acc:", valacc_, "\n\n")
        # tf.summary.scalar('validation loss', trainloss)
        # tf.summary.scalar('validation acc', trainacc)
        # merged=tf.summary.merge_all()
        result = sess.run(mergeall)
        sess.run(tf.assign(trainacc, trainacc_))  # 更新tensorboard数据
        sess.run(tf.assign(trainloss, trainloss_))
        sess.run(tf.assign(valloss, valloss_))
        sess.run(tf.assign(valacc, valacc_))
        writer.add_summary(result, epoch)
    saver.save(sess, model_path)
    sess.close()


def forecast(img_path):
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        img = io.imread(img_path)

        img = transform.resize(img, (w, h))
        img = [img]
        logit = sess.run(logits_eval, feed_dict={x: img})
        result = tf.nn.softmax(logit)
        return sess.run(result)

if __name__=='__main__':
    train()
    #read_img(path)
    #print(forecast('./test/3/3333.jpg'))
