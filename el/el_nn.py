# coding: utf-8

import random
import numpy as np
import time
import tensorflow as tf
import input_data
import math
import gensim
import jieba
import os
import random
import cPickle as pickle

# mnist = input_data.read_data_sets("/tmp/data",one_hot=False)
# load word2vec
print "load word2vec model"
t_start = time.time()
model = gensim.models.Word2Vec.load_word2vec_format(
    "./GoogleNews-vectors-negative300.bin", binary=True)
print "load word2vec model finsh, use time %.3fs" % (time.time() - t_start)

sen_fix_len = 500 

def get_sentence_vector(sentence):
    global model
    global sen_fix_len
    words = jieba.cut(sentence)
    sentence_vec = []
    cnt = 0
    for word in words:
        try:
            word_vec = model[word]
        except:  # 将词向量中不存在的词填充
            continue
        cnt += 1
        if cnt == sen_fix_len:
            continue
        # word_vec = [2] * 300
        sentence_vec.append(word_vec)

    if len(sentence_vec) < sen_fix_len:  
       sentence_vec.extend([[2]*300]*(sen_fix_len - len(sentence_vec)))

    return np.array(sentence_vec).reshape(-1)

def gen_entity_text_vec():
    entity_text_dict_vec = {}
    data_dir = "./data/entity_text/"
    filelist = os.listdir(data_dir) 
    for filename in filelist:
        #print filename
        filepath = data_dir + filename
        with open(filepath) as fr:
            text = fr.read()
            text_vec = get_sentence_vector(text)
            entity_text_dict_vec[filename] = text_vec
            
    pickle.dump(entity_text_dict_vec,open("entity_text_dict_vec.pk","w"))     


def gen_mention_text_vec():
    mention_text_dict_vec = {} 
    data_dir = "./data/mention_text/"
    filelist = os.listdir(data_dir)
    for filename in filelist:
        # print filename
        filepath = data_dir + filename
        with open(filepath) as fr:
            text = fr.read()
            text_vec = get_sentence_vector(text)
            mention_text_dict_vec[filename] = text_vec
    pickle.dump(mention_text_dict_vec, open("mention_text_dict_vec.pk","w"))




print "gen entity_text_dict"
t_start = time.time()
if os.path.isfile("entity_text_dict_vec.pk"):
    entity_text_dict_vec = pickle.load(open("entity_text_dict_vec.pk"))
else:
    gen_entity_text_vec()
    entity_text_dict_vec = pickle.load(open("entity_text_dict_vec.pk"))
print "gen entity_text_dict_vec ok, use time %.3fs" % (time.time() - t_start)

print "gen mention_text_dict"
t_start = time.time()
if os.path.isfile("mention_text_dict_vec.pk"):
    mention_text_dict_vec = pickle.load(open("mention_text_dict_vec.pk"))
else:
    gen_mention_text_vec()
    mention_text_dict_vec = pickle.load(open("mention_text_dict_vec.pk"))
print "gen mention_text_dict_vec ok, use time %.3fs" % (time.time() - t_start)

del model
del jieba

def load_entity_text():
    entity_text_dict = {}
    data_dir = "./data/entity_text/"
    filelist = os.listdir(data_dir) 
    for filename in filelist:
        filepath = data_dir + filename
        entity_text_dict[filename] = open(filepath).read()
    return entity_text_dict

'''
print "load entity_text_dict"
t_start = time.time()
entity_text_dict = load_entity_text()
print "load entity_text_dict ok, use time %.3fs" % (time.time() - t_start)
'''

def load_mention_text():
    mention_text_dict = {} 
    data_dir = "./data/mention_text/"
    filelist = os.listdir(data_dir)
    for filename in filelist:
        # print filename
        filepath = data_dir + filename
        mention_text_dict[filename] = open(filepath).read()
    return mention_text_dict

'''
print "load mention_text_dict"
t_start = time.time()
mention_text_dict = load_mention_text()
print "load mention_text_dict ok, use time %.3fs" % (time.time() - t_start)
'''

def create_mention_entity_pairs(mention_text_dict_vec, entity_text_dict_vec):
    pairs = []
    labels = []
    with open("./data/eng_gold.tab") as fr:
        cnt = 0
        for line in fr:
            cnt += 1
            if cnt == 1000:
                break
            #print line
            print cnt
            tokens = line.split("\t")
            mention_text_id = tokens[3].split(":")[0]
            entity_text_id = tokens[4]
            m_type = tokens[6]
            if "NOM" in m_type:
                continue
            if "NIL" in entity_text_id:
                continue
            mention_text_vec = mention_text_dict_vec[mention_text_id+".xml"]
            #mention_text_vec = get_sentence_vector(mention_text)
            entity_text_vec = entity_text_dict_vec[entity_text_id]
            #entity_text_vec = get_sentence_vector(entity_text)
            pairs += [[mention_text_vec, entity_text_vec]]
            labels += [1]

            # error pairs.
    '''
            entity_keys = entity_text_dict.keys()
            entity_random_id = entity_keys[random.randint(0, len(entity_keys)-1)]
            while entity_random_id == entity_text_id:
                entity_random_id = entity_keys[random.randint(0, len(entity_keys) - 1)]
                entity_text_vec = get_sentence_vector(entity_text_dict[entity_random_id])

            pairs+= [[mention_text_vec, entity_text_vec]]
            labels += [0]
    '''

    return np.array(pairs), np.array(labels)

print "create_mention_entity_pairs"
t_start = time.time()
pairs, labels = create_mention_entity_pairs(mention_text_dict_vec, entity_text_dict_vec) 
print "create_mention_entity_pairs ok, use time%.3fs" % (time.time() - t_start)

import pdb
def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
            return np.array(pairs), np.array(labels)



# weight initialization
def weight_variable(shape, name="W"):
    with tf.variable_scope(name):
        return tf.get_variable("W",shape)

def bias_variable(shape,name="b"):
    with tf.variable_scope(name):
        return tf.get_variable("b",shape)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def mclnet(x,_dropout):
    # first convolutional layer
    x_image = tf.reshape(x, [-1,500,300,1])
    W_conv1 = weight_variable([5,5,1,32],"W_conv1")
    b_conv1 = bias_variable([32],"b_conv1")   
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # seconed convolutional layer
    W_conv2 = weight_variable([5,5,32,64], "W_conv2")
    b_conv2 = bias_variable([64], "b_conv2")
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)


    # densely connected layer
    W_fc1 = weight_variable([125*75*64, 125*75*64],"W_fc1")
    b_fc1 = bias_variable([125*75*64],"b_fc1")

    h_pool2_flat = tf.reshape(h_pool2,[-1,125*75*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # droput 
    h_fc1_drop = tf.nn.dropout(h_fc1, _dropout)
    #     h_fc1_drop = tf.nn.dropout(h_fc1, )

    # readout layer
    W_fc2 = weight_variable([125*75*64,100],"W_fc2")
    b_fc2 = bias_variable([100],"b_fc2")

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

    return y_conv


# In[4]:

def mlp(input_,input_dim,output_dim,name="mlp"):
    with tf.variable_scope(name):
        w = tf.get_variable('w',[input_dim,output_dim],tf.float32,tf.random_normal_initializer(mean = 0.001,stddev=0.02))
        return tf.nn.relu(tf.matmul(input_,w))

def build_model_mlp(X_,_dropout):

    model = mlpnet(X_,_dropout)
    return model

def mlpnet(image,_dropout):
    l1 = mlp(image,784,128,name='l1')
    l1 = tf.nn.dropout(l1,_dropout)
    l2 = mlp(l1,128,128,name='l2')
    l2 = tf.nn.dropout(l2,_dropout)
    l3 = mlp(l2,128,128,name='l3')
    return l3

def contrastive_loss(y,d):
    tmp= y *tf.square(d)
    # tmp= tf.mul(y,tf.square(d))
    tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
    return tf.reduce_sum(tmp +tmp2)/batch_size/2

def compute_accuracy(prediction,labels):
    return labels[prediction.ravel() < 0.5].mean()
# return tf.reduce_mean(labels[prediction.ravel() < 0.5])
def next_batch(s,e,inputs,labels):
    input1 = inputs[s:e,0]
    input2 = inputs[s:e,1]
    y= np.reshape(labels[s:e],(len(range(s,e)),1))
    return input1,input2,y
# Initializing the variables
init = tf.initialize_all_variables()
# the data, shuffled and split between train and test sets
X_train = mnist.train._images
y_train = mnist.train._labels
X_test = mnist.test._images
y_test = mnist.test._labels
batch_size =128
global_step = tf.Variable(0,trainable=False)
starter_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,10,0.1,staircase=True)
# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(10)]
tr_pairs, tr_y = create_pairs(X_train, digit_indices)
digit_indices = [np.where(y_test == i)[0] for i in range(10)]
te_pairs, te_y = create_pairs(X_test, digit_indices)

images_L = tf.placeholder(tf.float32,shape=([None,784]),name='L')
images_R = tf.placeholder(tf.float32,shape=([None,784]),name='R')
labels = tf.placeholder(tf.float32,shape=([None,1]),name='gt')
dropout_f = tf.placeholder("float")

# with tf.variable_scope("siamese") as scope:
#     model1= build_model_mlp(images_L,dropout_f)
#     scope.reuse_variables()
#     model2 = build_model_mlp(images_R,dropout_f)

with tf.variable_scope("siamese") as scope:
model1 = mclnet(images_L, dropout_f)
scope.reuse_variables()
model2 = mclnet(images_R, dropout_f)



distance  = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(model1,model2),2),1,keep_dims=True))
loss = contrastive_loss(labels,distance)


# contrastice loss
t_vars = tf.trainable_variables()
d_vars  = [var for var in t_vars if 'l' in var.name]
batch = tf.Variable(0)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss)
# optimizer = tf.train.RMSPropOptimizer(0.0001,momentum=0.9,epsilon=1e-6).minimize(loss)
# Launch the graph
with tf.Session() as sess:
# sess.run(init)
tf.initialize_all_variables().run()
# Training cycle
for epoch in range(30):
avg_loss = 0.
avg_acc = 0.
total_batch = int(X_train.shape[0]/batch_size)
start_time = time.time()
# Loop over all batches
for i in range(total_batch):
# print i
s  = i * batch_size
e = (i+1) *batch_size
# Fit training using batch data
input1,input2,y =next_batch(s,e,tr_pairs,tr_y)
_,loss_value,predict=sess.run([optimizer,loss,distance], feed_dict={images_L:input1,images_R:input2 ,labels:y,dropout_f:0.9})
feature1=model1.eval(feed_dict={images_L:input1,dropout_f:0.9})
feature2=model2.eval(feed_dict={images_R:input2,dropout_f:0.9})
tr_acc = compute_accuracy(predict,y)
if math.isnan(tr_acc) and epoch != 0:
print('tr_acc %0.2f' % tr_acc)
pdb.set_trace()
avg_loss += loss_value
avg_acc +=tr_acc*100
# print('epoch %d loss %0.2f' %(epoch,avg_loss/total_batch))
duration = time.time() - start_time
print('epoch %d  time: %f loss %0.5f acc %0.2f' %(epoch,duration,avg_loss/(total_batch),avg_acc/total_batch))

    total_train_batch = int(X_train.shape[0]/batch_size)
    temp_acc = 0.0
    for i in range(total_train_batch):
    s = i * batch_size
    e = (i+1)*batch_size
    input1,input2,y = next_batch(s,e,tr_pairs,tr_y)

        # y = np.reshape(tr_y,(tr_y.shape[0],1))
        predict=distance.eval(feed_dict={images_L:input1,images_R:input2,labels:y,dropout_f:1.0})
        tr_acc = compute_accuracy(predict,y)
        temp_acc += tr_acc

    print('Accuract training set %0.2f' % (100 * temp_acc/total_train_batch))

    # test 
    total_test_batch = int(X_test.shape[0]/batch_size)
    temp_acc = 0.0
    for i in range(total_test_batch):
    s = i * batch_size
    e = (i+1)*batch_size
    input1,input2,y = next_batch(s,e,te_pairs,te_y)

        # y = np.reshape(tr_y,(tr_y.shape[0],1))
        predict=distance.eval(feed_dict={images_L:input1,images_R:input2,labels:y,dropout_f:1.0})
        te_acc = compute_accuracy(predict,y)
        temp_acc += te_acc

    print('Accuract test set %0.2f' % (100 * temp_acc/total_test_batch))


    # Test model
    # predict=distance.eval(feed_dict={images_L:te_pairs[:,0],images_R:te_pairs[:,1],labels:y,dropout_f:1.0})
    # y = np.reshape(te_y,(te_y.shape[0],1))
    # te_acc = compute_accuracy(predict,y)
    # print('Accuract test set %0.2f' % (100 * te_acc))
    '''
