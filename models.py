import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import Linear

################################################################################
##  source : https://github.com/bdy9527/SDCN
################################################################################

class SDCN_AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z):
        super(SDCN_AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z

class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if active:
            output = F.relu(output)
        return output

class SDCN_GNN(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z, n_clusters):
        super(SDCN_GNN, self).__init__()
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

    def forward(self, x, adj, tra1, tra2, tra3, z, sigma=0.5):
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1-sigma)*h + sigma*tra1, adj)
        h = self.gnn_3((1-sigma)*h + sigma*tra2, adj)
        h = self.gnn_4((1-sigma)*h + sigma*tra3, adj)
        h = self.gnn_5((1-sigma)*h + sigma*z, adj, active=False)
        predict = F.softmax(h, dim=1)
        return predict

class SDCN(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z, n_clusters, load_from, mode, v=1):
        super(SDCN, self).__init__()

        # autoencoder for intra information
        self.ae = SDCN_AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(load_from, map_location='cpu'))

        # GCN for inter information
        self.gnn = SDCN_GNN(n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        x_bar, tra1, tra2, tra3, z = self.ae(x)
        predict = self.gnn(x, adj, tra1, tra2, tra3, z, sigma=0.5)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z

################################################################################
##  source : https://github.com/ZhihaoPENG-CityU/MM21---AGCN
################################################################################
class MLP_L(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_L, self).__init__()
        self.wl = Linear(n_mlp, 5)
    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.wl(mlp_in)), dim=1)
        return weight_output

class MLP_1(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_1, self).__init__()
        self.w1 = Linear(n_mlp,2)
    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w1(mlp_in)), dim=1) 
        return weight_output

class MLP_2(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_2, self).__init__()
        self.w2 = Linear(n_mlp, 2)
    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w2(mlp_in)), dim=1)
        return weight_output

class MLP_3(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_3, self).__init__()
        self.w3 = Linear(n_mlp, 2)
    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w3(mlp_in)), dim=1)  
        return weight_output

class AGCN_GNN(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z, n_clusters):
        super(AGCN_GNN, self).__init__()
        self.n_all = n_enc_1 + n_enc_2 + n_enc_3 + n_z + n_z

        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(self.n_all, n_clusters)

        # attention on [Z_i || H_i]
        self.mlp1 = MLP_1(2*n_enc_1)
        self.mlp2 = MLP_2(2*n_enc_2)
        self.mlp3 = MLP_3(2*n_enc_3)
        self.mlp = MLP_L(self.n_all)


    def forward(self, x, adj, h1, h2, h3, z, sigma=0.5):
        n_x = x.shape[0]
        # # AGCN-H
        z1 = self.gnn_1(x, adj)
        # z2
        m1 = self.mlp1( torch.cat((h1,z1), 1) )
        m1 = F.normalize(m1,p=2)
        m11 = torch.reshape(m1[:,0], [n_x, 1])
        m12 = torch.reshape(m1[:,1], [n_x, 1])
        m11_broadcast =  m11.repeat(1,500)
        m12_broadcast =  m12.repeat(1,500)
        z2 = self.gnn_2( m11_broadcast.mul(z1)+m12_broadcast.mul(h1), adj)
        # z3
        m2 = self.mlp2( torch.cat((h2,z2),1) )     
        m2 = F.normalize(m2,p=2)
        m21 = torch.reshape(m2[:,0], [n_x, 1])
        m22 = torch.reshape(m2[:,1], [n_x, 1])
        m21_broadcast = m21.repeat(1,500)
        m22_broadcast = m22.repeat(1,500)
        z3 = self.gnn_3( m21_broadcast.mul(z2)+m22_broadcast.mul(h2), adj)
        # z4
        m3 = self.mlp3( torch.cat((h3,z3),1) )# self.mlp3(h2)      
        m3 = F.normalize(m3,p=2)
        m31 = torch.reshape(m3[:,0], [n_x, 1])
        m32 = torch.reshape(m3[:,1], [n_x, 1])
        m31_broadcast = m31.repeat(1,2000)
        m32_broadcast = m32.repeat(1,2000)
        z4 = self.gnn_4( m31_broadcast.mul(z3)+m32_broadcast.mul(h3), adj)

        # # AGCN-S
        u  = self.mlp(torch.cat((z1,z2,z3,z4,z),1))
        u = F.normalize(u,p=2) 
        u0 = torch.reshape(u[:,0], [n_x, 1])
        u1 = torch.reshape(u[:,1], [n_x, 1])
        u2 = torch.reshape(u[:,2], [n_x, 1])
        u3 = torch.reshape(u[:,3], [n_x, 1])
        u4 = torch.reshape(u[:,4], [n_x, 1])

        tile_u0 = u0.repeat(1,500)
        tile_u1 = u1.repeat(1,500)
        tile_u2 = u2.repeat(1,2000)
        tile_u3 = u3.repeat(1,10)
        tile_u4 = u4.repeat(1,10)

        net_output = torch.cat((tile_u0.mul(z1), tile_u1.mul(z2), tile_u2.mul(z3), tile_u3.mul(z4), tile_u4.mul(z)), 1 )   
        net_output = self.gnn_5(net_output, adj, active=False) 
        predict = F.softmax(net_output, dim=1)
        return predict

class AGCN(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z, n_clusters, load_from, mode, v=1):
        super(AGCN, self).__init__()

        # autoencoder for intra information
        self.ae = SDCN_AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(load_from, map_location='cpu'))

        # GCN for inter information
        self.gnn = AGCN_GNN(n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        x_bar, tra1, tra2, tra3, z = self.ae(x)
        predict = self.gnn(x, adj, tra1, tra2, tra3, z, sigma=0.5)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z

################################################################################
################################################################################
################################################################################
class MLP(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z):
        super(MLP, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        return 0, enc_h1, enc_h2, enc_h3, z

class SCGC(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z, n_clusters, load_from, mode, v=1):
        super(SCGC, self).__init__()

        # autoencoder for intra information
        self.ae = SDCN_AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)

        pretrained_state_dict = torch.load(load_from, map_location='cpu')
        trimmed_pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in self.ae.state_dict()}  #filter out unnecessary keys
        self.ae.load_state_dict(trimmed_pretrained_state_dict)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        x_bar, tra1, tra2, tra3, z = self.ae(x)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, None, z, None, None

class SCGC_TRIM(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z, n_clusters, load_from, mode, v=1):
        super(SCGC_TRIM, self).__init__()

        # autoencoder for intra information
        self.ae = MLP(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)

        pretrained_state_dict = torch.load(load_from, map_location='cpu')
        trimmed_pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in self.ae.state_dict()}  #filter out unnecessary keys
        self.ae.load_state_dict(trimmed_pretrained_state_dict)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        x_bar, tra1, tra2, tra3, z = self.ae(x)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, None, z, None, None
