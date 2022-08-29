batch_size = 32    #批处理大小
lambda_gen = 1e-5  #生成器正则化
lambda_dis = 1e-5  #判别器正则化
n_sample = 1    #样本大小
lr_gen = 0.0001#1e-3  学习率
lr_dis = 0.0001#1e-4
n_epoch = 20  #总共训练次数
saves_step = 10
sig = 1.0    #生成器高斯分布的方差
t_size = 5 #节点类型的数目
t_dimension = 3#节点的嵌入维度

label_smooth = 0.  #平滑

d_epoch = 15 #每次判别器迭代次数
g_epoch = 5#每次生成器迭代次数

n_emb = 64  #嵌入大小

dataset = 'yelp' #使用的数据集

#graph_filename = '../data/' + dataset + '/' + dataset + '_triple.dat'
graph_filename = '../data/' + dataset + '_triple.dat'


pretrain_node_emb_filename_d = '../pre_train/node_clustering/' + dataset + '_pre_train.emb'
pretrain_node_emb_filename_g = '../pre_train/node_clustering/' + dataset + '_pre_train.emb'
#pretrain_rel_emb_filename_d = '../data/' + dataset + '/rel_embeddings.txt'
#depretrain_rel_emb_filename_g = '../data/' + dataset + '/rel_embeddings.txt'

emb_filenames = ['../results/' + dataset + '_gen.emb',
                 '../results/' + dataset + '_dis.emb']

model_log = '../log/'
