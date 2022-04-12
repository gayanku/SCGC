from __future__ import print_function, division
import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
#from torch.nn import Linear
from utils import load_data, load_graph
from evaluation import eva, display
from collections import Counter
import sys
import time

import pickle 

from models import *

# torch.cuda.set_device(1)

def get_A_r_flex(adj, r, cumulative=False):
    adj_d = adj.to_dense()
    adj_c = adj_d           # A1, A2, A3 .....
    adj_label = adj_d

    for i in range(r-1):
        adj_c = adj_c@adj_d
        adj_label = adj_label + adj_c if cumulative else adj_c
    return adj_label

################################################################################
##  Source : https://github.com/yanghu819/Graph-MLP
################################################################################
def get_feature_dis(x):
    #x :           batch_size x nhid
    #x_dis(i,j):   item means the similarity between x(i) and x(j).
    x_dis = x@x.T
    mask = torch.eye(x_dis.shape[0]).to(x.device)
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis*(x_sum**(-1))
    x_dis = (1-mask) * x_dis
    return x_dis

def Ncontrast(x_dis, adj_label, tau = 1):
    # compute the Ncontrast loss
    x_dis = torch.exp( tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss

################################################################################
##  Code modified from : https://github.com/bdy9527/SDCN
################################################################################

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_scgc_trim(dataset):
    _model = getattr(sys.modules[__name__], args.model)
    #print(f'Model : {_model}')

    model = _model(500, 500, 2000, 2000, 500, 500,
                n_input=args.n_input,
                n_z=args.n_z,
                n_clusters=args.n_clusters,
                load_from=f'{args.data_path}/data/{args.name}.pkl',
                mode=args.mode,
                v=1.0).to(device)
    if args.verbosity > 1: print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # KNN Graph
    adj = load_graph(args.data_path, args.name, args.k)
    adj = get_A_r_flex(adj, args.order, cumulative=args.influence)
    adj = adj.to(device)

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y
    with torch.no_grad():
        _, _, _, _, z = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'Loaded PAE', verbosity=5)

    # get the value
    kmeans_Q, NMI_Q, ARI_Q, F1_Q = [],[],[],[]
    kmeans_Z, NMI_Z, ARI_Z, F1_Z = [],[],[],[]
    kmeans_P, NMI_P, ARI_P, F1_P = [],[],[],[]

    if args.cuda:  # INIT LOGGERS
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
    start_time = time.time()

    best_Z = None
    for epoch in range(args.epochs):
        if epoch % 1 == 0:
        # update_interval
            _, tmp_q, pred, z, _,_ = model(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
        
            res1 = tmp_q.cpu().numpy().argmax(1)       #Q
            res2 = res1 #pred.data.cpu().numpy().argmax(1)   #Z
            res3 = p.data.cpu().numpy().argmax(1)      #P
            eva(y, res1, str(epoch) + 'Q', kmeans_Q, NMI_Q, ARI_Q, F1_Q, verbosity=args.verbosity)
            eva(y, res2, str(epoch) + 'Z <', kmeans_Z, NMI_Z, ARI_Z, F1_Z, verbosity=args.verbosity)
            eva(y, res3, str(epoch) + 'P', kmeans_P, NMI_P, ARI_P, F1_P, verbosity=args.verbosity)
            #if args.verbosity > 2: print(f' {epoch} Z:{np.max(kmeans_Z):6.4f}    Q:{np.max(kmeans_Q):6.4f}    P:{np.max(kmeans_P):6.4f}')

            #if np.max(kmeans_Z) == kmeans_Z[-1]:
            #   best_Z = z.data.cpu().numpy()
        with record_function("_MODEL_TRAIN"):
            x_bar, q, _, z,  _,_ = model(data, adj)

        with record_function("_MODEL_KL"):
            kl_loss = F.kl_div(q.log(), p, reduction='batchmean')

        with record_function("_MODEL_DIST"):
            x_dis = get_feature_dis(z)
        with record_function("_MODEL_CONTRASTIVE"):
            nContrast_loss = Ncontrast(x_dis, adj, tau = args.tau)

        loss = args.beta * kl_loss + args.alpha * nContrast_loss #+ re_loss #ClusterMLP Loss
        if args.verbosity > 2: print(f' {epoch} {args.model}   Z:{kmeans_Z[-1]:6.4f}  Q:{kmeans_Q[-1]:6.4f}  P:{kmeans_P[-1]:6.4f}    |   Z:{np.max(kmeans_Z):6.4f}  Q:{np.max(kmeans_Q):6.4f}  P:{np.max(kmeans_P):6.4f}   ||   L:{loss.item():6.4f}  > KL:{kl_loss.item():6.4f}  NC:{nContrast_loss.item():6.4f}  RE:___', flush=True)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    gpu_time = 0
    if args.cuda:
        ender.record()
        torch.cuda.synchronize()
        gpu_time = starter.elapsed_time(ender)
    clock_time = time.time() - start_time

    return best_Z, (np.max(kmeans_Q), np.max(NMI_Q), np.max(ARI_Q), np.max(F1_Q), \
        np.max(kmeans_Z), np.max(NMI_Z), np.max(ARI_Z), np.max(F1_Z), \
        np.max(kmeans_P), np.max(NMI_P), np.max(ARI_P), np.max(F1_P), gpu_time, clock_time)



def train_scgc(dataset):
    _model = getattr(sys.modules[__name__], args.model)
    #print(f'Model : {_model}')

    model = _model(500, 500, 2000, 2000, 500, 500,
                n_input=args.n_input,
                n_z=args.n_z,
                n_clusters=args.n_clusters,
                load_from=f'{args.data_path}/data/{args.name}.pkl',
                mode=args.mode,
                v=1.0).to(device)
    if args.verbosity > 1: print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # KNN Graph
    adj = load_graph(args.data_path, args.name, args.k)
    adj = get_A_r_flex(adj, args.order, cumulative=args.influence)
    adj = adj.to(device)

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y
    with torch.no_grad():
        _, _, _, _, z = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'Loaded PAE', verbosity=5)

    # get the value
    kmeans_Q, NMI_Q, ARI_Q, F1_Q = [],[],[],[]
    kmeans_Z, NMI_Z, ARI_Z, F1_Z = [],[],[],[]
    kmeans_P, NMI_P, ARI_P, F1_P = [],[],[],[]

    if args.cuda:  # INIT LOGGERS
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
    start_time = time.time()

    best_Z = None
    for epoch in range(args.epochs):
        if epoch % 1 == 0:
        # update_interval
            _, tmp_q, pred, z, _,_ = model(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
        
            res1 = tmp_q.cpu().numpy().argmax(1)       #Q
            res2 = res1 #pred.data.cpu().numpy().argmax(1)   #Z
            res3 = p.data.cpu().numpy().argmax(1)      #P
            eva(y, res1, str(epoch) + 'Q', kmeans_Q, NMI_Q, ARI_Q, F1_Q, verbosity=args.verbosity)
            eva(y, res2, str(epoch) + 'Z <', kmeans_Z, NMI_Z, ARI_Z, F1_Z, verbosity=args.verbosity)
            eva(y, res3, str(epoch) + 'P', kmeans_P, NMI_P, ARI_P, F1_P, verbosity=args.verbosity)
            #if args.verbosity > 2: print(f' {epoch} Z:{np.max(kmeans_Z):6.4f}    Q:{np.max(kmeans_Q):6.4f}    P:{np.max(kmeans_P):6.4f}')

            #if np.max(kmeans_Z) == kmeans_Z[-1]:
            #   best_Z = z.data.cpu().numpy()
        
        with record_function("_MODEL_TRAIN"):
            x_bar, q, _, z,  _,_ = model(data, adj)

        with record_function("_MODEL_KL"):
            kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        
        with record_function("_MODEL_MSE"):
            re_loss = F.mse_loss(x_bar, data)

        with record_function("_MODEL_DIST"):
            x_dis = get_feature_dis(z)
        with record_function("_MODEL_CONTRASTIVE"):
            nContrast_loss = Ncontrast(x_dis, adj, tau = args.tau)

        loss = args.beta * kl_loss + args.alpha * nContrast_loss + re_loss #ClusterMLP Loss
        if args.verbosity > 2: print(f' {epoch} {args.model}   Z:{kmeans_Z[-1]:6.4f}  Q:{kmeans_Q[-1]:6.4f}  P:{kmeans_P[-1]:6.4f}    |   Z:{np.max(kmeans_Z):6.4f}  Q:{np.max(kmeans_Q):6.4f}  P:{np.max(kmeans_P):6.4f}   ||   L:{loss.item():6.4f}  > KL:{kl_loss.item():6.4f}  NC:{nContrast_loss.item():6.4f}  RE:{re_loss.item():6.4f}', flush=True)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    gpu_time = 0
    if args.cuda:
        ender.record()
        torch.cuda.synchronize()
        gpu_time = starter.elapsed_time(ender)
    clock_time = time.time() - start_time


    return best_Z, (np.max(kmeans_Q), np.max(NMI_Q), np.max(ARI_Q), np.max(F1_Q), \
        np.max(kmeans_Z), np.max(NMI_Z), np.max(ARI_Z), np.max(F1_Z), \
        np.max(kmeans_P), np.max(NMI_P), np.max(ARI_P), np.max(F1_P), gpu_time, clock_time)


def train_sdcn(dataset):
    _model = getattr(sys.modules[__name__], args.model)
    #print(f'Model : {_model}')

    model = _model(500, 500, 2000, 2000, 500, 500,
                n_input=args.n_input,
                n_z=args.n_z,
                n_clusters=args.n_clusters,
                load_from=f'{args.data_path}/data/{args.name}.pkl',
                mode=args.mode,
                v=1.0).to(device)
    if args.verbosity > 1: print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # KNN Graph
    adj = load_graph(args.data_path, args.name, args.k)
    adj = adj.to(device)

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y
    with torch.no_grad():
        _, _, _, _, z = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'Loaded PAE', verbosity=args.verbosity)

    # get the value
    kmeans_Q, NMI_Q, ARI_Q, F1_Q = [],[],[],[]
    kmeans_Z, NMI_Z, ARI_Z, F1_Z = [],[],[],[]
    kmeans_P, NMI_P, ARI_P, F1_P = [],[],[],[]

    if args.cuda:  # INIT LOGGERS
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
    start_time = time.time()

    best_Z = None
    for epoch in range(args.epochs):
        if epoch % 1 == 0:
        # update_interval
            _, tmp_q, pred, z = model(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
        
            res1 = tmp_q.cpu().numpy().argmax(1)       #Q
            res2 = pred.data.cpu().numpy().argmax(1)   #Z
            res3 = p.data.cpu().numpy().argmax(1)      #P
            eva(y, res1, str(epoch) + 'Q', kmeans_Q, NMI_Q, ARI_Q, F1_Q, verbosity=args.verbosity)
            eva(y, res2, str(epoch) + 'Z <', kmeans_Z, NMI_Z, ARI_Z, F1_Z, verbosity=args.verbosity)
            eva(y, res3, str(epoch) + 'P', kmeans_P, NMI_P, ARI_P, F1_P, verbosity=args.verbosity)
            #if args.verbosity > 2: print(f' {epoch} Z:{np.max(kmeans_Z):6.4f}    Q:{np.max(kmeans_Q):6.4f}    P:{np.max(kmeans_P):6.4f}')

            #if np.max(kmeans_Z) == kmeans_Z[-1]:
            #   best_Z = z.data.cpu().numpy()
        
        with record_function("_MODEL_TRAIN"):
            x_bar, q, pred, _ = model(data, adj)

        with record_function("_MODEL_KL_Q"):
            kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        with record_function("_MODEL_KL_Z"):
            ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')

        with record_function("_MODEL_MSE"):
          re_loss = F.mse_loss(x_bar, data)

        #loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss #SDCN Loss DEFAULT
        loss = args.beta * kl_loss + args.alpha * ce_loss + re_loss #SDCN Loss alpha=0.01, beta = 0.1,   AGCN l2=beta l1=alpha
        if args.verbosity > 2: print(f' {epoch} {args.model}   Z:{kmeans_Z[-1]:6.4f}  Q:{kmeans_Q[-1]:6.4f}  P:{kmeans_P[-1]:6.4f}    |   Z:{np.max(kmeans_Z):6.4f}  Q:{np.max(kmeans_Q):6.4f}  P:{np.max(kmeans_P):6.4f}   ||   L:{loss.item():6.4f}  > KL:{kl_loss.item():6.4f}  CE:{ce_loss.item():6.4f}  RE:{re_loss.item():6.4f}', flush=True)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    gpu_time = 0
    if args.cuda:
        ender.record()
        torch.cuda.synchronize()
        gpu_time = starter.elapsed_time(ender)
    clock_time = time.time() - start_time

    return best_Z, (np.max(kmeans_Q), np.max(NMI_Q), np.max(ARI_Q), np.max(F1_Q), \
        np.max(kmeans_Z), np.max(NMI_Z), np.max(ARI_Z), np.max(F1_Z), \
        np.max(kmeans_P), np.max(NMI_P), np.max(ARI_P), np.max(F1_P), gpu_time, clock_time)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='reut')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/001_Clustering/_Dataset_SDCN')

    parser.add_argument('--model', type=str, default='SCGC')                    # AGCN SDCN SCGC, SCGC_TRIM
    parser.add_argument('--mode', type=str, default='full')                     # Full, trim
    parser.add_argument('--influence', default=False, action='store_true',      help='Use Inluence contrastive')


    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--verbosity', type=int, default=0)

    parser.add_argument('--note', type=str, default='-')

    parser.add_argument('--alpha', type=float, default=2.0,                     help='To control the ratio of Ncontrast loss')
    parser.add_argument('--beta', type=float, default=0.1,                      help='To control the ratio of Clustering loss')

    parser.add_argument('--batch_size', type=int, default=2048,                 help='batch size')
    parser.add_argument('--order', type=int, default=2,                         help='to compute order-th power of adj')
    parser.add_argument('--tau', type=float, default=1.0,                       help='temperature for Ncontrast loss')

    # 0 Args Final
    # 1 + Itr_totals 
    # 2 + Model PAE
    # 3 + epoch_totals
    # 4
    # 5 show all

    parser.add_argument('--seed', type=int, default=42, help='Random seed.')    #42    Solution to all problems !
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    dataset = load_data(args.data_path, args.name)

    if args.name == 'usps':
        args.n_clusters = 10
        args.n_input = 256

    if args.name == 'hhar':
        args.k = 5
        args.n_clusters = 6
        args.n_input = 561

    if args.name == 'reut':
        args.n_clusters = 4
        args.n_input = 2000

    if args.name == 'acm':
        args.k = None
        args.n_clusters = 3
        args.n_input = 1870

    if args.name == 'dblp':
        args.k = None
        args.n_clusters = 4
        args.n_input = 334

    if args.name == 'cite':
        args.k = None
        args.n_clusters = 6
        args.n_input = 3703

    print(args)

    kmeans_iter_Q, NMI_iter_Q, ARI_iter_Q, F1_iter_Q = [],[],[],[]
    kmeans_iter_Z, NMI_iter_Z, ARI_iter_Z, F1_iter_Z = [],[],[],[]
    kmeans_iter_P, NMI_iter_P, ARI_iter_P, F1_iter_P = [],[],[],[]
    gpu_time_iter, clock_time_iter = [], []

    for i in range(args.iterations):
        if args.verbosity > 1: print ('iteration____________________________________________', i)

        from torch.profiler import profile, record_function, ProfilerActivity
        print('---------------PROFILING CODE--------------')
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=False, with_flops=True) as prof:
            with record_function("_MODEL_TRAIN_ALL"):
                if args.model =='SDCN': best_Z, vals = train_sdcn(dataset)
                if args.model =='AGCN': best_Z, vals = train_sdcn(dataset)
                if args.model =='SCGC': best_Z, vals = train_scgc(dataset)
                if args.model =='SCGC_TRIM': best_Z, vals = train_scgc_trim(dataset)
                
        #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
        print( '\n'.join([ line for line in prof.key_averages().table(row_limit=1000000).split('\n') if           any(tag in line for tag in    ('---', 'Name', 'Self CPU', 'Self CUDA',  '_MODEL')   )   ]))

        #if best_Z is not None:
        #    pickle.dump(best_Z, open( f'BEST/best_Z_{args.name}_{args.model}_{i}.v1.pkl', 'wb' ) )

        if args.verbosity > 0: print ('iteration', i, 'Q[acc,nmi,ari,f1]ZP GPU CPU     ', '  '.join([f'{v:6.4f}' for v in vals]))
        kmeans_iter_Q.append(vals[0]); NMI_iter_Q.append(vals[1]); ARI_iter_Q.append(vals[2]); F1_iter_Q.append(vals[3]); 
        kmeans_iter_Z.append(vals[4]); NMI_iter_Z.append(vals[5]); ARI_iter_Z.append(vals[6]); F1_iter_Z.append(vals[7]); 
        kmeans_iter_P.append(vals[8]); NMI_iter_P.append(vals[9]); ARI_iter_P.append(vals[10]); F1_iter_P.append(vals[11]); 
        gpu_time_iter.append(vals[12]); clock_time_iter.append(vals[13]); 


    print(f'Z:acc-nmi-ari-F1-gpu-clock: {display(kmeans_iter_Z)},|,{display(NMI_iter_Z)},|,{display(ARI_iter_Z)},|,{display(F1_iter_Z)},|,{display(gpu_time_iter)},|,{display(clock_time_iter)},||, {args}'  )
