import os
import numpy as np
np.random.seed(0)
from collections import defaultdict, Counter
# import pickle5 as pickle
import pickle
import torch
import torch.nn as nn
import scipy.sparse as sp
from utils import process
import sys
from itertools import combinations
import networkx as nx

def except_patterns(nodes,adj):
    a = nodes[0]
    b = nodes[1]
    c = nodes[2]
    exc=[]
    if adj[a,b]==adj[a,c]==adj[b,c]==0:
        pass
    elif adj[a,b]==1 and adj[a,c]==adj[b,c]==0:
        exc.extend([0,2,3,6])
    elif adj[a,c]==1 and adj[a,b]==adj[b,c]==0:
        exc.extend([0,1,3,5])
    elif adj[b,c]==1 and adj[a,b]==adj[a,c]==0:
        exc.extend([0,1,2,4])
    elif adj[a,b]==adj[a,c] and adj[a,b]==1 and adj[b,c]==0:
        exc.extend([0,1,2,3,5,6])
    elif adj[a,b]==adj[b,c] and adj[a,b]==1 and adj[a,c]==0:
        exc.extend([0,1,2,3,4,6])
    elif adj[a,c]==adj[b,c] and adj[a,c]==1 and adj[a,b]==0:
        exc.extend([0,1,2,3,4,5])
    elif adj[a,b]==adj[a,c] and adj[a,c]==adj[b,c] and adj[a,b]==1:
        exc.extend([0,1,2,3,4,5,6])
    return exc

class embedder:
    def __init__(self, args, rewrite=False):
        args.sparse = True
        if args.gpu == "cpu":
            args.device = "cpu"
        else:
            args.device = torch.device("cuda:"+ args.gpu if torch.cuda.is_available() else "cpu")

        path = "./dataset/"+args.dataset
        adj_norm = True
        norm = True

        with open(path+'/meta_data.pkl', 'rb') as f:
            data = pickle.load(f)
        idx ={}
        self.node2type={}
        for t in data['t_info'].keys():
            idx[t] = torch.LongTensor(data['t_info'][t]['ind'])
            for i in data['t_info'][t]['ind']:
                self.node2type[i]=t
        node2id = data['node2gid']
        edge_types=[]
        for edge_type in combinations(list(idx.keys()), 2):
            edge_types.append(edge_type)
        edge_type_dict={}
        edge_type_num=0
        for i in edge_types:
            if i not in edge_type_dict:
                edge_type_dict[i]=edge_type_num
                edge_type_num+=1
            if (i[1],i[0]) not in edge_type_dict:
                edge_type_dict[(i[1],i[0])] = edge_type_dict[i]

        with open(path+'/edges.pkl', 'rb') as f:
            edges = pickle.load(f)


        all_x=np.loadtxt(path+'/train.txt',dtype=int)
        all_y=np.loadtxt(path+'/label.txt',dtype=int)
        if os.path.exists(path+'/train_x.npy'):
            train_x = np.load(path+'/train_x.npy')
            train_y = np.load(path+'/train_y.npy')
            val_x = np.load(path+'/val_x.npy')
            val_y = np.load(path+'/val_y.npy')
            test_x = np.load(path+'/test_x.npy')
            test_y = np.load(path+'/test_y.npy')

        else:
            train_rate = 0.6
            val_rate = 0.2
            counts = dict(Counter(all_y))
            train_data_index = [[] for i in range(8)]
            val_data_index = [[] for i in range(8)]
            test_data_index = [[] for i in range(8)]
            for id, tp in enumerate(all_y):
                if len(train_data_index[tp]) < train_rate * counts[tp]:
                    train_data_index[tp].append(id)
                    continue
                if len(val_data_index[tp]) < val_rate * counts[tp]:
                    val_data_index[tp].append(id)
                    continue
                test_data_index[tp].append(id)


            train_x = [all_x[j] for i in train_data_index for j in i]
            train_y = [all_y[j] for i in train_data_index for j in i]
            np.save(path+'/train_x.npy', train_x)
            np.save(path+'/train_y.npy', train_y)


            val_x = [all_x[j] for i in val_data_index for j in i]
            val_y = [all_y[j] for i in val_data_index for j in i]
            np.save(path+'/val_x.npy', val_x)
            np.save(path+'/val_y.npy', val_y)


            test_x = [all_x[j] for i in test_data_index for j in i]
            test_y = [all_y[j] for i in test_data_index for j in i]
            np.save(path+'/test_x.npy', test_x)
            np.save(path+'/test_y.npy', test_y)

        train_x = torch.tensor(train_x).to(args.device)
        train_y = torch.tensor(train_y).to(args.device)
        val_x = torch.tensor(val_x).to(args.device)
        val_y = torch.tensor(val_y).to(args.device)
        test_x = torch.tensor(test_x).to(args.device)
        test_y = torch.tensor(test_y).to(args.device)



        full_graph = nx.Graph()
        with open(path+'/graph.txt', 'r', encoding='utf-8')as f:
            temp_edges = f.read()
        temp_edges = temp_edges.split('\n')[:-1]
        for edge in temp_edges:
            edge = edge.split('\t')
            full_graph.add_edge(int(edge[0]), int(edge[1]),weight=1)
        adj_train = nx.adjacency_matrix(full_graph, nodelist=range(0,len(node2id)))
        full_graph = nx.Graph(adj_train)
        for i, edge in enumerate(full_graph.edges()):
            t = edge_type_dict[(self.node2type[int(edge[0])], self.node2type[int(edge[1])])]
            full_graph[edge[0]][edge[1]]['type'] = t


        excepts = []
        for x in train_x:
            temp_excepts = except_patterns(x.cpu().numpy(), adj_train)
            excepts.append(temp_excepts)



        node_rel = defaultdict(list)
        
        rel_types = set()
        neighbors = defaultdict(set)
       
        if args.model in ['ANSEHGN']:
            subgraph = {}
            subgraph_nb = {}
            edge_index = np.array([[],[]])
            
            for rel in edges:
                s, t = rel.split('-')
                vu = t + '-' + s
                node_rel[s].append(rel)
                rel_types.add(rel)
                x,y = edges[rel].nonzero()
                for i,j in zip(x,y):
                    neighbors[i].add(j)
            for nt, rels in node_rel.items():
                rel_list = []
                nb_neighbor = []
                for rel in rels:
                    s, t = rel.split('-')
                    e = edges[rel][idx[s],:][:,idx[t]]
                    nb = e.sum(1)
                    nb_neighbor.append(torch.FloatTensor(nb))
                    if adj_norm:
                        e = process.normalize_adj(e)
                    e = process.sparse_to_tuple(e)  # coords, values, shape
                    rel_list.append(torch.sparse_coo_tensor(torch.LongTensor(e[0]),torch.FloatTensor(e[1]), torch.Size(e[2])))
                    edge_index = np.concatenate([edge_index, e[0]],-1)
                    
                subgraph[nt] = rel_list
                subgraph_nb[nt] = nb_neighbor
   


                
        neighbors_list = []
        for i in range(len(node2id)):
            if len(neighbors[i]) == 0:
                print('Node %s has no neighbor'%(str(i)))
            neighbors_list.append(neighbors[i].union(set([i])))
            

        with open(path+"/node_features.pkl", "rb") as f:
            features =torch.FloatTensor(pickle.load(f))
        ft = features.shape[1]
        # self.features = process.preprocess_features(features, norm=norm)
        self.features = features
        self.graph = subgraph
        self.full_graph=full_graph
        self.neighbor_list = neighbors_list
        self.graph_nb_neighbor = subgraph_nb
        self.A=sp.csr_matrix(adj_train)
        args.edge_type_num=edge_type_num
        args.node2id = node2id
        args.nt_rel = node_rel
        args.node_cnt = idx
        args.node_type = list(args.node_cnt)
        args.ft_size = ft
        args.node_size = len(node2id)
        args.rel_types = rel_types
        self.args = args
        self.edge_index = edge_index
        self.args.nb_edge = self.edge_index.shape[1]
        self.train_x=train_x
        self.train_y=train_y
        self.val_x=val_x
        self.val_y=val_y
        self.test_x=test_x
        self.test_y=test_y
        self.excepts=excepts
        args.a=0.5
        args.b=0.5

        print("Dataset: %s"%args.dataset)
        print("node_type num_node:")
        for t, num in self.args.node_cnt.items():
            print("\n%s\t %s"%(t, len(num)))
        print("Graph prepared!")
        print("Model setup:")
        print("learning rate: %s"%args.lr)
        
        print("model: %s" % args.model)
        if args.gpu == "cpu":
            print("use cpu")
        else:
            print("use cuda")
        if args.isAtt:
            print("use attention")
        else:
            print("use mean pool")


