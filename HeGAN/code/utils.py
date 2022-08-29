import numpy as np
import config

t_size = config.t_size

def read_graph(graph_filename):
    # dblp                          ----------1----------
    # p -> a : 0  p: paper, t : term, a : author
    # a -> p : 1
    # p -> c : 2
    # c -> p : 3
    # p -> t : 4
    # t -> p : 5
    # p : 0, a : 1, c : 2, t : 3
    # self.t_relation = np.array([[-1, 0, 2, 4], [1, -1, -1, -1], [3, -1, -1, -1], [5, -1, -1, -1]])

    #yelp
    # 0 -> 1: 0
    # 0 -> 2: 2
    # 0 -> 3: 4
    # 0 -> 4: 6
    # 1 -> 0: 1
    # 2 -> 0: 3
    # 3 -> 0: 5
    # 7 -> 0: 7
    # self.t_relation = np.array([[-1, 0, 2, 4, 6], [1, -1, -1, -1, -1], [3, -1, -1, -1, -1], [5, -1, -1, -1, -1], [7, -1, -1, -1, -1]])

    graph_filename = '../data/yelp_lp/yelp_triple.dat'

    relations = set()
    nodes = set()
    graph = {}
    g_type = np.zeros([t_size, t_size])
    m_type = {}

    #map_type() 用于区分两节点类型的

    with open(graph_filename) as infile:
        for line in infile.readlines():
            source_node, target_node, relation = line.strip().split(' ')
            source_node = int(source_node)
            target_node = int(target_node)
            relation = int(relation)

            #dblp            ------------2-----------
            # if(relation == 0):
            #     g_type[0][1] += 1
            #     m_type[source_node] = 0
            # elif(relation == 1):
            #     g_type[1][0] += 1
            #     m_type[source_node] = 1
            # elif(relation == 2):
            #     g_type[0][2] += 1
            #     m_type[source_node] = 0
            # elif(relation == 3):
            #     g_type[2][0] += 1
            #     m_type[source_node] = 2
            # elif(relation == 4):
            #     g_type[0][3] += 1
            #     m_type[source_node] = 0
            # elif(relation == 5):
            #     g_type[3][0] += 1
            #     m_type[source_node] = 3
            # else:
            #     pass

            # yelp
            # 0 -> 1: 0
            # 0 -> 2: 2
            # 0 -> 3: 4
            # 0 -> 4: 6
            # 1 -> 0: 1
            # 2 -> 0: 3
            # 3 -> 0: 5
            # 7 -> 0: 7

            if(relation == 0):
                g_type[0][1] += 1
                m_type[source_node] = 0
            elif(relation == 1):
                g_type[1][0] += 1
                m_type[source_node] = 1
            elif(relation == 2):
                g_type[0][2] += 1
                m_type[source_node] = 0
            elif(relation == 3):
                g_type[2][0] += 1
                m_type[source_node] = 2
            elif(relation == 4):
                g_type[0][3] += 1
                m_type[source_node] = 0
            elif(relation == 5):
                g_type[3][0] += 1
                m_type[source_node] = 3
            elif(relation == 6):
                g_type[0][4] += 1
                m_type[source_node] = 0
            elif (relation == 7):
                g_type[4][0] += 1
                m_type[source_node] = 4
            else:
                pass

            nodes.add(source_node)
            nodes.add(target_node)
            relations.add(relation)

            if source_node not in graph:
                graph[source_node] = {}

            if relation not in graph[source_node]:
                graph[source_node][relation] = []

            graph[source_node][relation].append(target_node)

    norm_g_type = Normalized(g_type)
    #print relations
    n_node = len(nodes)
    return n_node, len(relations), graph, g_type, norm_g_type, m_type

def all_Normalized(g_type):
    sum = np.sum(g_type)
    norm_g_type = np.zeros(g_type.shape)
    for i in range(t_size):
        for j in range(t_size):
            norm_g_type[i][j] = float(g_type[i][j]) / sum
    return norm_g_type

def Normalized(g_type):
    sum = np.sum(g_type,axis=1)
    norm_g_type = np.zeros(g_type.shape)
    for i in range(t_size):
        for j in range(t_size):
            norm_g_type[i][j] = float(g_type[i][j]) / sum[i]
    return norm_g_type

def map_type():  #将节点编号映射成节点类型
    type_path = []
    m_type = {}

    #加入节点类型的文件

    for item in type_path:
        with open(item) as infile:
            for line in infile.readlines():
                source_node, type = line.strip().split(' ')
                m_type[source_node] = type

    return m_type

def str_list_to_float(str_list):
    return [float(item) for item in str_list]

def read_embeddings(filename, n_node, n_embed):

    embedding_matrix = np.random.rand(n_node, n_embed)
    i = -1
    with open(filename) as infile:
        for line in infile.readlines()[1:]:
            i += 1
            emd = line.strip().split()
            embedding_matrix[int(emd[0]), :] = str_list_to_float(emd[1:])
    return embedding_matrix

if __name__ == '__main__':
    n_node, n_relation, graph  = read_graph()

    #embedding_matrix = read_embeddings('../data/dblp/rel_embeddings.txt', 6, 64)
    print(graph[1][1])
