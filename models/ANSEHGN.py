import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from embedder import embedder
from torch_scatter import scatter_add
from torch_sparse import SparseTensor
from layers import *
from evaluation import evaluation_metrics
from layers.mlp import MLP
import time
class SubHIN(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args
        self.data={'train_x':self.train_x,'train_y':self.train_y,
                   'val_x':self.val_x,'val_y':self.val_y,
                   'test_x':self.test_x,'test_y':self.test_y,
                   'except':self.excepts,'A':self.A,'G':self.full_graph}

    def training(self):
        self.features = self.features.to(self.args.device)
        self.graph = {t: [m.to(self.args.device) for m in ms] for t, ms in self.graph.items()}
        model = modeler(self.args,self.data).to(self.args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr)
        cnt_wait = 0; best = 1e9; best_auc = 0
        test_ev=0
        file = '.npy'

        for epoch in range(self.args.nb_epochs):
            model.train()
            optimizer.zero_grad()

            embs,train_logits = model(self.graph, self.features)
            loss = model.loss2(embs, train_logits,self.full_graph)
            loss.backward()
            optimizer.step()
   
            train_loss = loss.item()
    
            # validation
            model.eval()
            val_struct=model.gen_struct(self.val_x)
            val_logits = model.mlp(embs[self.val_x[:, 0]], embs[self.val_x[:, 1]], embs[self.val_x[:, 2]],val_struct)
            ev = evaluation_metrics(val_logits, self.data['val_y'])
            auc = ev.auc
            
            if (train_loss < best) and (auc >= best_auc):
                best = train_loss
                best_auc = auc
                cnt_wait = 0
                test_struct=model.gen_struct(self.test_x)
                test_logits = model.mlp(embs[self.test_x[:, 0]], embs[self.test_x[:, 1]], embs[self.test_x[:, 2]],test_struct)
                test_ev = evaluation_metrics(test_logits, self.data['test_y'])
                test_auc = test_ev.auc
                print("Epoch {}, loss: {:.4}, val_auc: {:.4}, test_auc:{:.4}".format(epoch, train_loss, best_auc,test_auc))
                outs = embs.detach().cpu().numpy()

            else:
                cnt_wait += 1

            if cnt_wait == self.args.patience:
                np.save("output_emb/new/"+self.args.dataset+file, outs) 
                break

        print("test_macro_f1:{:.4},test_micro_f1:{:.4},test_nmi:{:.4}".format(test_ev.macro_f1,test_ev.micro_f1,test_ev.nmi))


class modeler(nn.Module):
    def __init__(self, args,data):
        super().__init__()
        self.args = args
        self.bnn = nn.ModuleDict()
        self.fc = nn.ModuleDict()

        self.semanticatt = nn.ModuleDict()

        self.train_x=data['train_x']
        self.train_y=data['train_y']
        self.val_x=data['val_x']
        self.val_y=data['val_y']
        self.test_x=data['test_x']
        self.test_y=data['test_y']
        self.excepts=data['except']
        self.A=data['A']
        self.A2 = self.A.dot(self.A)
        self.G=data['G']
        rows, cols =self.A.nonzero()
        self.edge_types = [self.G[i][j]['type'] for i, j in zip(rows, cols)]
        self.f_edge=nn.ModuleDict()
        for i in range(self.args.edge_type_num):
            self.f_edge[str(i)] = torch.nn.Sequential(torch.nn.Linear(1, args.out_ft),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(args.out_ft, 1))

        self.f_node = torch.nn.Sequential(torch.nn.Linear(1, args.out_ft),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(args.out_ft, 1))
        self.g_phi = torch.nn.Sequential(torch.nn.Linear(1, args.out_ft),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(args.out_ft, 1))
        self.alpha = torch.nn.Parameter(torch.FloatTensor([0, 0]))
        for t, rels in self.args.nt_rel.items():

            self.fc[t] = FullyConnect(args.hid_units2+args.ft_size, args.out_ft, drop_prob=self.args.drop_prob)

            for rel in rels: 
                self.bnn['0'+rel] = GCN(args.ft_size, args.hid_units, act=nn.ReLU(), isBias=args.isBias)
                self.bnn['1'+rel] = GCN(args.hid_units, args.hid_units2, act=nn.ReLU(), isBias=args.isBias)

            self.semanticatt['0'+t] = SemanticAttention(args.hid_units, args.hid_units//4)
            self.semanticatt['1'+t] = SemanticAttention(args.hid_units2, args.hid_units2//4)
        self.mlp=MLP(args.out_ft,args.out_ft//2)
        self.struct_mlp=MLP(1,args.out_ft)
        self.liner = FullyConnect(3 * args.out_ft, args.out_ft)

    def forward(self, graph, features):
        embs1 = torch.zeros((self.args.node_size, self.args.hid_units)).to(self.args.device)
        embs2 = torch.zeros((self.args.node_size, self.args.out_ft)).to(self.args.device)
        for n, rels in self.args.nt_rel.items():
            vec = []
            for j, rel in enumerate(rels):
                t = rel.split('-')[1]
                    
                mean_neighbor = torch.spmm(graph[n][j], features[self.args.node_cnt[t]])
                v = self.bnn['0'+rel](mean_neighbor)
                vec.append(v)

            vec = torch.stack(vec, 0)
            
            if self.args.isAtt:
                v_summary = self.semanticatt['0'+n](vec.view(-1, self.args.hid_units), len(rels))
            else:
                v_summary = torch.mean(vec, 0)

            embs1[self.args.node_cnt[n]] = v_summary
        
        for n, rels in self.args.nt_rel.items():
            vec = []
            for j, rel in enumerate(rels):
                t = rel.split('-')[-1]
                mean_neighbor = torch.spmm(graph[n][j], embs1[self.args.node_cnt[t]])
                v = self.bnn['1'+rel](mean_neighbor)  
                vec.append(v)

            vec = torch.stack(vec, 0)
            if self.args.lamb_lp:
                v_summary = self.semanticatt['1'+n](vec.view(-1, self.args.hid_units2), len(rels))
            else:
                v_summary = torch.mean(vec, 0)

            v_cat = torch.hstack((v_summary, features[self.args.node_cnt[n]]))
            v_summary = self.fc[n](v_cat)
            
            embs2[self.args.node_cnt[n]] = v_summary



        row_A, col_A = self.A.nonzero()
        tmp_A = torch.stack([torch.from_numpy(row_A), torch.from_numpy(col_A)]).type(torch.LongTensor).to(self.args.device)
        row_A, col_A = tmp_A[0], tmp_A[1]
        edge_weight_A = torch.from_numpy(self.A.data).unsqueeze(-1).float().to(self.args.device)
        edge_weight_A = [self.f_edge[str(self.edge_types[i])](edge_weight) for i, edge_weight in enumerate(edge_weight_A)]
        edge_weight_A = torch.stack(edge_weight_A)
        self.node_struct_feat = scatter_add(edge_weight_A, col_A, dim=0, dim_size=len(self.G.nodes))
        out_batch = []
        batches = torch.chunk(self.train_x, chunks=1, dim=0)
        for batch in batches:
            out_struct = self.gen_struct(batch)
            out_batch.append(self.mlp(embs2[batch[:, 0]], embs2[batch[:, 1]], embs2[batch[:, 2]], out_struct))
        out = torch.cat(out_batch, dim=0)
        return embs2,out

    def gen_struct(self,x):

        indexes_src = x[:, 0].cpu().numpy()
        row_src1, col_src1 = self.A[indexes_src].nonzero()
        edge_index_src1 = torch.stack([torch.from_numpy(row_src1), torch.from_numpy(col_src1)]).type(torch.LongTensor).to(
            self.args.device)
        edge_weight_src1 = torch.from_numpy(self.A[indexes_src].data).to(self.args.device)
        edge_weight_src1 = edge_weight_src1 * self.f_node(self.node_struct_feat[col_src1]).squeeze()

        indexes_mid = x[:, 1].cpu().numpy()
        row_mid1, col_mid1 = self.A[indexes_mid].nonzero()
        edge_index_mid1 = torch.stack([torch.from_numpy(row_mid1), torch.from_numpy(col_mid1)]).type(torch.LongTensor).to(
            self.args.device)
        edge_weight_mid1 = torch.from_numpy(self.A[indexes_mid].data).to(self.args.device)
        edge_weight_mid1= edge_weight_mid1 * self.f_node(self.node_struct_feat[col_mid1]).squeeze()

        indexes_dst = x[:, 2].cpu().numpy()
        row_dst1, col_dst1 = self.A[indexes_dst].nonzero()
        edge_index_dst1 = torch.stack([torch.from_numpy(row_dst1), torch.from_numpy(col_dst1)]).type(torch.LongTensor).to(
            self.args.device)
        edge_weight_dst1 = torch.from_numpy(self.A[indexes_dst].data).to(self.args.device)
        edge_weight_dst1 = edge_weight_dst1 * self.f_node(self.node_struct_feat[col_dst1]).squeeze()

        batch_size = x.shape[0]
        mat_src1 = SparseTensor.from_edge_index(edge_index_src1, edge_weight_src1, [batch_size, len(self.G.nodes)])
        mat_mid1 = SparseTensor.from_edge_index(edge_index_mid1, edge_weight_mid1, [batch_size, len(self.G.nodes)])
        mat_dst1 = SparseTensor.from_edge_index(edge_index_dst1, edge_weight_dst1, [batch_size, len(self.G.nodes)])



        indexes_src = x[:, 0].cpu().numpy()
        row_src2, col_src2 = self.A2[indexes_src].nonzero()
        edge_index_src2 = torch.stack([torch.from_numpy(row_src2), torch.from_numpy(col_src2)]).type(torch.LongTensor).to(
            self.args.device)
        edge_weight_src2 = torch.from_numpy(self.A2[indexes_src].data).to(self.args.device)
        edge_weight_src2 = 0.8*edge_weight_src2 * self.f_node(self.node_struct_feat[col_src2]).squeeze()

        indexes_mid = x[:, 1].cpu().numpy()
        row_mid2, col_mid2 = self.A2[indexes_mid].nonzero()
        edge_index_mid2 = torch.stack([torch.from_numpy(row_mid2), torch.from_numpy(col_mid2)]).type(torch.LongTensor).to(
            self.args.device)
        edge_weight_mid2 = torch.from_numpy(self.A2[indexes_mid].data).to(self.args.device)
        edge_weight_mid2 = 0.8 * edge_weight_mid2 * self.f_node(self.node_struct_feat[col_mid2]).squeeze()

        indexes_dst = x[:, 2].cpu().numpy()
        row_dst2, col_dst2 = self.A2[indexes_dst].nonzero()
        edge_index_dst2 = torch.stack([torch.from_numpy(row_dst2), torch.from_numpy(col_dst2)]).type(torch.LongTensor).to(
            self.args.device)
        edge_weight_dst2 = torch.from_numpy(self.A2[indexes_dst].data).to(self.args.device)
        edge_weight_dst2 = 0.8 * edge_weight_dst2 * self.f_node(self.node_struct_feat[col_dst2]).squeeze()

        batch_size = x.shape[0]
        mat_src2 = SparseTensor.from_edge_index(edge_index_src2, edge_weight_src2, [batch_size, len(self.G.nodes)])
        mat_mid2 = SparseTensor.from_edge_index(edge_index_mid2, edge_weight_mid2, [batch_size, len(self.G.nodes)])
        mat_dst2 = SparseTensor.from_edge_index(edge_index_dst2, edge_weight_dst2, [batch_size, len(self.G.nodes)])



        mat_src=mat_src1+mat_src2
        mat_mid=mat_mid1+mat_mid2
        mat_dst=mat_dst1+mat_dst2


        out_struct_temp1 = (mat_src @ mat_mid.to_dense().t()).diag()
        out_struct_temp2 = (mat_src @ mat_dst.to_dense().t()).diag()
        out_struct_temp3 = (mat_mid @ mat_dst.to_dense().t()).diag()

        out_struct1=self.g_phi(out_struct_temp1.unsqueeze(-1))
        out_struct2=self.g_phi(out_struct_temp2.unsqueeze(-1))
        out_struct3=self.g_phi(out_struct_temp3.unsqueeze(-1))
        out_struct = torch.cat((out_struct1, out_struct2, out_struct3),dim=-1)
        return out_struct
    def loss2(self, embs2, logits, graph):
        loss_funcution = nn.CrossEntropyLoss()
        loss_funcution_binary = nn.BCEWithLogitsLoss()
        multi_loss = loss_funcution(logits, self.train_y)
        n={0:0,1:1,2:1,3:1,4:2,5:2,6:2,7:3}
        #The  pattern id(0-7)in paper is not same as that in code data processing.So The the key of p is not same as the paper.
        p={0:0,2:0,1:1,3:1,4:1,6:1,5:2,7:2}
        except_loss=0.0
        for exc, log in zip(self.excepts, logits):
            coefficient=[]
            ex_loss=[]
            if exc:
                for ex in exc:
                    ex_pred = log[ex]
                    ex_label = torch.tensor(0.0).to(self.args.device)
                    coefficient.append(torch.exp(torch.tensor(-(self.args.a*n[ex]+self.args.b*p[ex]))))
                    ex_loss.append(loss_funcution_binary(ex_pred, ex_label))
                coefficient=np.array(coefficient)
                exc_loss=np.dot(coefficient,np.array(ex_loss))
                except_loss +=exc_loss
        totalLoss=multi_loss+except_loss/self.train_x.shape[0]
        return totalLoss



