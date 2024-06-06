
import torch
from interfaces import TempTranslationModel, TempBilinearModel
from torchkge.utils import init_embedding
from model_utils import LSTMModel
from torch.nn.functional import normalize
import torch.nn.functional as F
from torch import nn
import numpy as np

from utils import device


class TTransEModel(TempTranslationModel):

    def __init__(self, args):

        super().__init__(args.n_entities, args.n_relations, args.dissimilarity_type)

        self.emb_dim = args.emb_dim
        self.tem_total = args.tem_total
        
        self.ent_emb = init_embedding(self.n_ent, self.emb_dim)
        self.rel_emb = init_embedding(self.n_rel, self.emb_dim)
        self.tem_emb = init_embedding(self.tem_total, self.emb_dim)
        self.lstm = LSTMModel(self.emb_dim, n_layer=1)
        self.time_trans = None
        self.normalize_parameters()
        self.rel_emb.weight.data = normalize(self.rel_emb.weight.data, p=2, dim=1)
        
    def scoring_function(self, h_idx, t_idx, r_idx, start_time_idx, end_time_idx):
        start_time_idx = self.time_trans[start_time_idx]
        h = self.ent_emb(h_idx)
        t = self.ent_emb(t_idx)
        r = self.get_rseq(r_idx, start_time_idx)
        
        return - self.dissimilarity(h + r, t)
        
    def normalize_parameters(self):
        self.ent_emb.weight.data = normalize(self.ent_emb.weight.data,
                                             p=2, dim=1)

    def get_embeddings(self):
        self.normalize_parameters()
        return self.ent_emb.weight.data, self.rel_emb.weight.data

    def get_rseq(self, r, tem):
        r_e = self.rel_emb(r)
        r_e = r_e.unsqueeze(0).transpose(0, 1)

        bs = tem.shape[0]  # batch size
        tem_len = tem.shape[1]
        tem = tem.contiguous()
        tem = tem.view(bs * tem_len)
        token_e = self.tem_emb(tem)
        token_e = token_e.view(bs, tem_len, self.emb_dim)
        seq_e = torch.cat((r_e, token_e), 1)

        hidden_tem = self.lstm(seq_e)
        hidden_tem = hidden_tem[0, :, :]
        rseq_e = hidden_tem

        return rseq_e
    
    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, start_time_idx, end_time_idx, entities=True):
        start_time_idx = self.time_trans[start_time_idx]
        b_size = h_idx.shape[0]

        h = self.ent_emb(h_idx)
        t = self.ent_emb(t_idx)
        r = self.get_rseq(r_idx, start_time_idx)
        
        if entities:
            candidates = self.ent_emb.weight.data.view(1, self.n_ent, self.emb_dim)
            candidates = candidates.expand(b_size, self.n_ent, self.emb_dim)
        else:
            candidates = self.rel_emb.weight.data.view(1, self.n_rel, self.emb_dim)
            candidates = candidates.expand(b_size, self.n_rel, self.emb_dim)

        return h, t, r, candidates
    def inference_scoring_function(self, proj_h, proj_t, r, start_time, end_time):
        b_size = proj_h.shape[0]

        if len(r.shape) == 2:
            if len(proj_t.shape) == 3:
                assert (len(proj_h.shape) == 2)
                # this is the tail completion case in link prediction
                hr = (proj_h + r).view(b_size, 1, r.shape[1])
                return - self.dissimilarity(hr, proj_t)
            else:
                assert (len(proj_h.shape) == 3) & (len(proj_t.shape) == 2)
                # this is the head completion case in link prediction
                r_ = r.view(b_size, 1, r.shape[1])
                t_ = proj_t.view(b_size, 1, r.shape[1])
                return - self.dissimilarity(proj_h + r_, t_)
        elif len(r.shape) == 3:
            # this is the relation prediction case
            # Two cases possible:
            # * proj_ent.shape == (b_size, self.n_rel, self.emb_dim) -> projection depending on relations
            # * proj_ent.shape == (b_size, self.emb_dim) -> no projection
            proj_h = proj_h.view(b_size, -1, self.emb_dim)
            proj_t = proj_t.view(b_size, -1, self.emb_dim)
            return - self.dissimilarity(proj_h + r, proj_t)
        
class TDistMultModel(TempBilinearModel):

    def __init__(self, args):
        super().__init__(args.emb_dim, args.n_entities, args.n_relations)
        self.emb_dim = args.emb_dim
        

        self.ent_emb = init_embedding(self.n_ent, self.emb_dim)
        self.rel_emb = init_embedding(self.n_rel, self.emb_dim)

        self.tem_total = args.tem_total
        self.tem_emb = init_embedding(self.tem_total, self.emb_dim)
        self.lstm = LSTMModel(self.emb_dim, n_layer=1)
        self.time_trans = None
        self.normalize_parameters()

    def scoring_function(self, h_idx, t_idx, r_idx, start_time_idx, end_time_idx):
        start_time_idx = self.time_trans[start_time_idx]
        h = normalize(self.ent_emb(h_idx), p=2, dim=1)
        t = normalize(self.ent_emb(t_idx), p=2, dim=1)
        r = self.get_rseq(r_idx, start_time_idx)

        return (h * r * t).sum(dim=1)

    def get_rseq(self, r, tem):
        r_e = self.rel_emb(r)
        r_e = r_e.unsqueeze(0).transpose(0, 1)

        bs = tem.shape[0]  # batch size
        tem_len = tem.shape[1]
        tem = tem.contiguous()
        tem = tem.view(bs * tem_len)
        token_e = self.tem_emb(tem)
        token_e = token_e.view(bs, tem_len, self.emb_dim)
        seq_e = torch.cat((r_e, token_e), 1)

        hidden_tem = self.lstm(seq_e)
        hidden_tem = hidden_tem[0, :, :]
        rseq_e = hidden_tem

        return rseq_e

    def normalize_parameters(self):
        self.ent_emb.weight.data = normalize(self.ent_emb.weight.data, p=2,
                                             dim=1)


    def get_embeddings(self):
        self.normalize_parameters()
        return self.ent_emb.weight.data, self.rel_emb.weight.data

    def inference_scoring_function(self, h, t, r, start_time, end_time):
        b_size = h.shape[0]

        if len(t.shape) == 3:
            assert (len(h.shape) == 2) & (len(r.shape) == 2)
            # this is the tail completion case in link prediction
            hr = (h * r).view(b_size, 1, self.emb_dim)
            return (hr * t).sum(dim=2)
        elif len(h.shape) == 3:
            assert (len(t.shape) == 2) & (len(r.shape) == 2)
            # this is the head completion case in link prediction
            rt = (r * t).view(b_size, 1, self.emb_dim)
            return (h * rt).sum(dim=2)
        elif len(r.shape) == 3:
            assert (len(h.shape) == 2) & (len(t.shape) == 2)
            # this is the relation prediction case
            hr = (h.view(b_size, 1, self.emb_dim) * r)  # hr has shape (b_size, self.n_rel, self.emb_dim)
            return (hr * t.view(b_size, 1, self.emb_dim)).sum(dim=2)


    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, start_time_idx, end_time_idx, entities=True):
        b_size = h_idx.shape[0]
        start_time_idx = self.time_trans[start_time_idx]
        h = self.ent_emb(h_idx)
        t = self.ent_emb(t_idx)
        r = self.get_rseq(r_idx, start_time_idx)

        if entities:
            candidates = self.ent_emb.weight.data.view(1, self.n_ent, self.emb_dim)
            candidates = candidates.expand(b_size, self.n_ent, self.emb_dim)
        else:
            candidates = self.rel_emb.weight.data.view(1, self.n_rel, self.emb_dim)
            candidates = candidates.expand(b_size, self.n_rel, self.emb_dim)

        return h, t, r, candidates


class DETransEModel(TempTranslationModel):

    def __init__(self, args):

        super().__init__(args.n_entities, args.n_relations, args.dissimilarity_type)

        self.emb_dim = args.emb_dim
        self.s_rep_ratio = args.s_rep_ratio
        self.ent_emb_dim = int(self.emb_dim * self.s_rep_ratio)
        self.time_emb_dim = args.emb_dim - self.ent_emb_dim
        self.time_trans = None
        self.ent_emb = init_embedding(self.n_ent, self.ent_emb_dim)
        self.rel_emb = init_embedding(self.n_rel, self.ent_emb_dim + self.time_emb_dim)
        self.device = device

        self.create_time_embedds()        
        self.normalize_parameters()
        self.rel_emb.weight.data = normalize(self.rel_emb.weight.data, p=2, dim=1)

    def create_time_embedds(self):
        self.m_freq = nn.Embedding(self.n_ent, self.time_emb_dim).to(device)
        self.d_freq = nn.Embedding(self.n_ent, self.time_emb_dim).to(device)
        self.y_freq = nn.Embedding(self.n_ent, self.time_emb_dim).to(device)
        nn.init.xavier_uniform_(self.m_freq.weight)
        nn.init.xavier_uniform_(self.d_freq.weight)
        nn.init.xavier_uniform_(self.y_freq.weight)
        
        
        self.m_phi = nn.Embedding(self.n_ent, self.time_emb_dim).to(device)
        self.d_phi = nn.Embedding(self.n_ent, self.time_emb_dim).to(device)
        self.y_phi = nn.Embedding(self.n_ent, self.time_emb_dim).to(device)
        nn.init.xavier_uniform_(self.m_phi.weight)
        nn.init.xavier_uniform_(self.d_phi.weight)
        nn.init.xavier_uniform_(self.y_phi.weight)

        self.m_amp = nn.Embedding(self.n_ent, self.time_emb_dim).to(device)
        self.d_amp = nn.Embedding(self.n_ent, self.time_emb_dim).to(device)
        self.y_amp = nn.Embedding(self.n_ent, self.time_emb_dim).to(device)
        nn.init.xavier_uniform_(self.m_amp.weight)
        nn.init.xavier_uniform_(self.d_amp.weight)
        nn.init.xavier_uniform_(self.y_amp.weight)
        
    def get_time_embedd(self, entities, start_time_idx, end_time_idx):
        year, month, day = start_time_idx[:,0].unsqueeze(dim=1), start_time_idx[:,1].unsqueeze(dim=1), start_time_idx[:,2].unsqueeze(dim=1)
        pi = 3.14159265359
        y = self.y_amp(entities)*torch.sin(self.y_freq(entities)*year + self.y_phi(entities))
        m = self.m_amp(entities)*torch.sin(self.m_freq(entities)*month + self.m_phi(entities))
        d = self.d_amp(entities)*torch.sin(self.d_freq(entities)*day + self.d_phi(entities))
        return y+m+d

    def getEmbeddings(self, heads_idx, rels_idx, tails_idx, start_time_idx, end_time_idx, intervals = None):
        h,r,t = self.ent_emb(heads_idx), self.rel_emb(rels_idx), self.ent_emb(tails_idx)
        h_t = self.get_time_embedd(heads_idx, start_time_idx, end_time_idx)
        t_t = self.get_time_embedd(tails_idx, start_time_idx, end_time_idx)

        h = torch.cat((h,h_t), 1)
        t = torch.cat((t,t_t), 1)
        return h,r,t

    def scoring_function(self, h_idx, t_idx, r_idx, start_time_idx, end_time_idx):
        start_time_idx = self.time_trans[start_time_idx]
        h,r,t = self.getEmbeddings(heads_idx = h_idx, rels_idx = r_idx, tails_idx= t_idx, start_time_idx=start_time_idx, end_time_idx=end_time_idx, intervals = None)
        h = normalize(h, p=2, dim=1)
        t = normalize(t, p=2, dim=1)

        return - self.dissimilarity(h + r, t)

    def normalize_parameters(self):
        self.ent_emb.weight.data = normalize(self.ent_emb.weight.data,
                                             p=2, dim=1)

    def get_embeddings(self):
        self.normalize_parameters()
        return self.ent_emb.weight.data, self.rel_emb.weight.data

    def inference_scoring_function(self, proj_h, proj_t, r, start_time, end_time):
        b_size = proj_h.shape[0]

        if len(r.shape) == 2:
            if len(proj_t.shape) == 3:
                assert (len(proj_h.shape) == 2)
                # this is the tail completion case in link prediction
                hr = (proj_h + r).view(b_size, 1, r.shape[1])
                return - self.dissimilarity(hr, proj_t)
            else:
                assert (len(proj_h.shape) == 3) & (len(proj_t.shape) == 2)
                # this is the head completion case in link prediction
                r_ = r.view(b_size, 1, r.shape[1])
                t_ = proj_t.view(b_size, 1, r.shape[1])
                return - self.dissimilarity(proj_h + r_, t_)
        elif len(r.shape) == 3:
            # this is the relation prediction case
            # Two cases possible:
            proj_h = proj_h.view(b_size, -1, self.emb_dim)
            proj_t = proj_t.view(b_size, -1, self.emb_dim)
            return - self.dissimilarity(proj_h + r, proj_t)
        
    def get_candidate_time_emb(self, time_amp, time_freq, time_phi, time):
        b_size = time.shape[0]
        time_amp_ = time_amp.weight.data.view(1, self.n_ent, self.time_emb_dim)
        time_amp_ = time_amp_.expand(b_size, self.n_ent, self.time_emb_dim)
        time_freq_ = time_freq.weight.data.view(1, self.n_ent, self.time_emb_dim)
        time_freq_ = time_freq_.expand(b_size, self.n_ent, self.time_emb_dim)
        time_phi_ = time_phi.weight.data.view(1, self.n_ent, self.time_emb_dim)
        time_phi_ = time_phi_.expand(b_size, self.n_ent, self.time_emb_dim)
        return time_amp_*torch.sin(time_freq_*time + time_phi_)
    
    def get_candidate_embedding(self, start_time_idx):
        b_size = start_time_idx.shape[0]
        year, month, day = start_time_idx[:,0].unsqueeze(dim=1), start_time_idx[:,1].unsqueeze(dim=1), start_time_idx[:,2].unsqueeze(dim=1)
        year = year.view(b_size, 1, year.shape[1])
        month = month.view(b_size, 1, month.shape[1])
        day = day.view(b_size, 1, day.shape[1])
        c = self.ent_emb.weight.data.view(1, self.n_ent, self.ent_emb_dim)
        c = c.expand(b_size, self.n_ent, self.ent_emb_dim)
        c_y = self.get_candidate_time_emb(self.y_amp, self.y_freq, self.y_phi, year)
        c_m = self.get_candidate_time_emb(self.m_amp, self.m_freq, self.m_phi, month)
        c_d = self.get_candidate_time_emb(self.d_amp, self.d_freq, self.d_phi, day)
        c = torch.cat((c,c_y + c_m + c_d), -1)
        return c
    
    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, start_time_idx, end_time_idx, entities=True):
        start_time_idx = self.time_trans[start_time_idx]
        b_size = h_idx.shape[0]
        h,r,t = self.getEmbeddings(heads_idx = h_idx, rels_idx = r_idx, tails_idx= t_idx, start_time_idx=start_time_idx, end_time_idx=end_time_idx, intervals = None)
        if entities:
            candidates = self.get_candidate_embedding(start_time_idx)
        else:
            candidates = self.rel_emb.weight.data.view(1, self.n_rel, self.emb_dim)
            candidates = candidates.expand(b_size, self.n_rel, self.emb_dim + self.time_emb_dim)

        return h, t, r, candidates

class DEDistMultModel(TempTranslationModel):

    def __init__(self, args):

        super().__init__(args.n_entities, args.n_relations, args.dissimilarity_type)

        self.emb_dim = args.emb_dim
        self.s_rep_ratio = args.s_rep_ratio
        self.ent_emb_dim = int(self.emb_dim * self.s_rep_ratio)
        self.time_emb_dim = args.emb_dim - self.ent_emb_dim
        self.time_trans = None
        self.ent_emb = init_embedding(self.n_ent, self.ent_emb_dim)
        self.rel_emb = init_embedding(self.n_rel, self.ent_emb_dim + self.time_emb_dim)
        self.device = device
        self.args = args
        self.create_time_embedds()        
        self.normalize_parameters()
        self.rel_emb.weight.data = normalize(self.rel_emb.weight.data, p=2, dim=1)

    def create_time_embedds(self):
        self.m_freq = nn.Embedding(self.n_ent, self.time_emb_dim).to(device)
        self.d_freq = nn.Embedding(self.n_ent, self.time_emb_dim).to(device)
        self.y_freq = nn.Embedding(self.n_ent, self.time_emb_dim).to(device)
        nn.init.xavier_uniform_(self.m_freq.weight)
        nn.init.xavier_uniform_(self.d_freq.weight)
        nn.init.xavier_uniform_(self.y_freq.weight)
        
        
        self.m_phi = nn.Embedding(self.n_ent, self.time_emb_dim).to(device)
        self.d_phi = nn.Embedding(self.n_ent, self.time_emb_dim).to(device)
        self.y_phi = nn.Embedding(self.n_ent, self.time_emb_dim).to(device)
        nn.init.xavier_uniform_(self.m_phi.weight)
        nn.init.xavier_uniform_(self.d_phi.weight)
        nn.init.xavier_uniform_(self.y_phi.weight)

        self.m_amp = nn.Embedding(self.n_ent, self.time_emb_dim).to(device)
        self.d_amp = nn.Embedding(self.n_ent, self.time_emb_dim).to(device)
        self.y_amp = nn.Embedding(self.n_ent, self.time_emb_dim).to(device)
        nn.init.xavier_uniform_(self.m_amp.weight)
        nn.init.xavier_uniform_(self.d_amp.weight)
        nn.init.xavier_uniform_(self.y_amp.weight)
        
    def get_time_embedd(self, entities, start_time_idx, end_time_idx):
        year, month, day = start_time_idx[:,0].unsqueeze(dim=1), start_time_idx[:,1].unsqueeze(dim=1), start_time_idx[:,2].unsqueeze(dim=1)
        pi = 3.14159265359
        y = self.y_amp(entities)*torch.sin(self.y_freq(entities)*year + self.y_phi(entities))
        m = self.m_amp(entities)*torch.sin(self.m_freq(entities)*month + self.m_phi(entities))
        d = self.d_amp(entities)*torch.sin(self.d_freq(entities)*day + self.d_phi(entities))
        return y+m+d

    def getEmbeddings(self, heads_idx, rels_idx, tails_idx, start_time_idx, end_time_idx, intervals = None):
        h,r,t = self.ent_emb(heads_idx), self.rel_emb(rels_idx), self.ent_emb(tails_idx)
        h_t = self.get_time_embedd(heads_idx, start_time_idx, end_time_idx)
        t_t = self.get_time_embedd(tails_idx, start_time_idx, end_time_idx)

        h = torch.cat((h,h_t), 1)
        t = torch.cat((t,t_t), 1)
        return h,r,t

    def scoring_function(self, h_idx, t_idx, r_idx, start_time_idx, end_time_idx):
        start_time_idx = self.time_trans[start_time_idx]
        h,r,t = self.getEmbeddings(heads_idx = h_idx, rels_idx = r_idx, tails_idx= t_idx, start_time_idx=start_time_idx, end_time_idx=end_time_idx, intervals = None)
        # h = normalize(h, p=2, dim=1)
        # t = normalize(t, p=2, dim=1)
        scores = (h * r * t)
        scores = F.dropout(scores, p=self.args.dropout, training=self.training)

        return scores.sum(dim=1)
    
    def normalize_parameters(self):
        self.ent_emb.weight.data = normalize(self.ent_emb.weight.data,
                                             p=2, dim=1)

    def get_embeddings(self):
        self.normalize_parameters()
        return self.ent_emb.weight.data, self.rel_emb.weight.data

    def inference_scoring_function(self, h, t, r, start_time, end_time):
        b_size = h.shape[0]

        if len(t.shape) == 3:
            assert (len(h.shape) == 2) & (len(r.shape) == 2)
            # this is the tail completion case in link prediction
            hr = (h * r).view(b_size, 1, self.emb_dim)
            return (hr * t).sum(dim=2)
        elif len(h.shape) == 3:
            assert (len(t.shape) == 2) & (len(r.shape) == 2)
            # this is the head completion case in link prediction
            rt = (r * t).view(b_size, 1, self.emb_dim)
            return (h * rt).sum(dim=2)
        elif len(r.shape) == 3:
            assert (len(h.shape) == 2) & (len(t.shape) == 2)
            # this is the relation prediction case
            hr = (h.view(b_size, 1, self.emb_dim) * r)  # hr has shape (b_size, self.n_rel, self.emb_dim)
            return (hr * t.view(b_size, 1, self.emb_dim)).sum(dim=2)
        
    def get_candidate_time_emb(self, time_amp, time_freq, time_phi, time):
        b_size = time.shape[0]
        time_amp_ = time_amp.weight.data.view(1, self.n_ent, self.time_emb_dim)
        time_amp_ = time_amp_.expand(b_size, self.n_ent, self.time_emb_dim)
        time_freq_ = time_freq.weight.data.view(1, self.n_ent, self.time_emb_dim)
        time_freq_ = time_freq_.expand(b_size, self.n_ent, self.time_emb_dim)
        time_phi_ = time_phi.weight.data.view(1, self.n_ent, self.time_emb_dim)
        time_phi_ = time_phi_.expand(b_size, self.n_ent, self.time_emb_dim)
        return time_amp_*torch.sin(time_freq_*time + time_phi_)
    
    def get_candidate_embedding(self, start_time_idx):
        b_size = start_time_idx.shape[0]
        year, month, day = start_time_idx[:,0].unsqueeze(dim=1), start_time_idx[:,1].unsqueeze(dim=1), start_time_idx[:,2].unsqueeze(dim=1)
        year = year.view(b_size, 1, year.shape[1])
        month = month.view(b_size, 1, month.shape[1])
        day = day.view(b_size, 1, day.shape[1])
        c = self.ent_emb.weight.data.view(1, self.n_ent, self.ent_emb_dim)
        c = c.expand(b_size, self.n_ent, self.ent_emb_dim)
        c_y = self.get_candidate_time_emb(self.y_amp, self.y_freq, self.y_phi, year)
        c_m = self.get_candidate_time_emb(self.m_amp, self.m_freq, self.m_phi, month)
        c_d = self.get_candidate_time_emb(self.d_amp, self.d_freq, self.d_phi, day)
        c = torch.cat((c,c_y + c_m + c_d), -1)
        return c
    
    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, start_time_idx, end_time_idx, entities=True):
        start_time_idx = self.time_trans[start_time_idx]
        b_size = h_idx.shape[0]
        h,r,t = self.getEmbeddings(heads_idx = h_idx, rels_idx = r_idx, tails_idx= t_idx, start_time_idx=start_time_idx, end_time_idx=end_time_idx, intervals = None)
        if entities:
            candidates = self.get_candidate_embedding(start_time_idx)
        else:
            candidates = self.rel_emb.weight.data.view(1, self.n_rel, self.emb_dim)
            candidates = candidates.expand(b_size, self.n_rel, self.emb_dim + self.time_emb_dim)

        return h, t, r, candidates


class TeRoModel(TempTranslationModel):

    def __init__(self, args):

        super().__init__(args.n_entities, args.n_relations, args.dissimilarity_type)

        self.emb_dim = args.emb_dim
        self.tem_total = args.tem_total
        self.dissimilarity_type = args.dissimilarity_type
        
        self.real_ent_emb = init_embedding(self.n_ent, self.emb_dim)
        self.img_ent_emb = init_embedding(self.n_ent, self.emb_dim)
        
        self.real_rel_emb = init_embedding(self.n_rel, self.emb_dim)
        self.img_rel_emb = init_embedding(self.n_rel, self.emb_dim)

        self.time_emb = init_embedding(self.tem_total, self.emb_dim)
        
        # Initialization
        r = 6 / np.sqrt(self.emb_dim)
        self.real_ent_emb.weight.data.uniform_(-r, r)
        self.img_ent_emb.weight.data.uniform_(-r, r)
        self.real_rel_emb.weight.data.uniform_(-r, r)
        self.img_rel_emb.weight.data.uniform_(-r, r)
        self.time_emb.weight.data.uniform_(-r, r)
        
        self.time_trans = None
        self.device = args.device
        self.out_mult = -1
    def scoring_function(self, h_idx, t_idx, r_idx, start_time_idx, end_time_idx):
        # print(start_time_idx.shape)
        start_time_idx = self.time_trans[start_time_idx]
        # print(start_time_idx.shape)
        
        pi = 3.14159265358979323846
        d_img = torch.sin(self.time_emb(start_time_idx).view(-1, self.emb_dim))#/(6 / np.sqrt(self.embedding_dim)/pi))

        d_real = torch.cos(self.time_emb(start_time_idx).view(-1, self.emb_dim))#/(6 / np.sqrt(self.embedding_dim)/pi))

        h_real = self.real_ent_emb(h_idx).view(-1, self.emb_dim) *d_real-\
                 self.img_ent_emb(h_idx).view(-1,self.emb_dim) *d_img

        t_real = self.real_ent_emb(t_idx).view(-1, self.emb_dim) *d_real-\
                 self.img_ent_emb(t_idx).view(-1, self.emb_dim)*d_img


        r_real = self.real_rel_emb(r_idx).view(-1, self.emb_dim)

        h_img = self.real_ent_emb(h_idx).view(-1, self.emb_dim) *d_img+\
                 self.img_ent_emb(h_idx).view(-1,self.emb_dim) *d_real


        t_img = self.real_ent_emb(t_idx).view(-1, self.emb_dim) *d_img+\
                self.img_ent_emb(t_idx).view(-1,self.emb_dim) *d_real

        r_img = self.img_rel_emb(r_idx).view(-1, self.emb_dim)

        
        if self.dissimilarity_type == 'L1':
            out_real = torch.sum(torch.abs(h_real + r_real - t_real), 1)
            out_img = torch.sum(torch.abs(h_img + r_img + t_img), 1)
            out = out_real + out_img

        else:
            out_real = torch.sum((h_real + r_real + d_img - t_real) ** 2, 1)
            out_img = torch.sum((h_img + r_img + d_img + t_real) ** 2, 1)
            out = torch.sqrt(out_img + out_real)


#         if self.dissimilarity_type == 'L1':
#             out_real = self.dissimilarity(h_real + r_real, t_real)
#             out_img = self.dissimilarity(h_img + r_img, - t_img)
#             out = out_real + out_img

#         else:
#             out_real = self.dissimilarity(h_real + r_real + start_time_idx.unsqueeze(1), t_real)
#             out_img = self.dissimilarity(h_img + r_img + start_time_idx.unsqueeze(1), - t_real)
#             out = torch.sqrt(out_img + out_real)

        return self.out_mult * out

    def normalize_parameters(self):
        self.ent_emb.weight.data = normalize(self.ent_emb.weight.data,
                                             p=2, dim=1)

    def get_embeddings(self):
        self.normalize_parameters()
        return self.ent_emb.weight.data, self.rel_emb.weight.data
    
    def get_candidate_embedding(self, start_time_idx):
        b_size = start_time_idx.shape[0]
        
        candidates_real = self.real_ent_emb.weight.data.view(1, self.n_ent, self.emb_dim)
        candidates_real = candidates_real.expand(b_size, self.n_ent, self.emb_dim)
        candidates_img = self.img_ent_emb.weight.data.view(1, self.n_ent, self.emb_dim)
        candidates_img = candidates_img.expand(b_size, self.n_ent, self.emb_dim)
        
        d_img = torch.sin(self.time_emb(start_time_idx).view(b_size, 1, self.emb_dim))#/(6 / np.sqrt(self.embedding_dim)/pi))

        d_real = torch.cos(self.time_emb(start_time_idx).view(b_size, 1, self.emb_dim))#/(6 / np.sqrt(self.embedding_dim)/pi))

        candidates_real = candidates_real *d_real-\
                 candidates_img *d_img

        candidates_img = candidates_real *d_img+\
                 candidates_img *d_real
        return torch.cat([candidates_real, candidates_img], dim = -1)
        
        
    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, start_time_idx, end_time_idx, entities=True):
        start_time_idx = self.time_trans[start_time_idx]
        b_size = h_idx.shape[0]
        d_img = torch.sin(self.time_emb(start_time_idx).view(-1, self.emb_dim))#/(6 / np.sqrt(self.embedding_dim)/pi))

        d_real = torch.cos(self.time_emb(start_time_idx).view(-1, self.emb_dim))#/(6 / np.sqrt(self.embedding_dim)/pi))

        h_real = self.real_ent_emb(h_idx).view(-1, self.emb_dim) *d_real-\
                 self.img_ent_emb(h_idx).view(-1, self.emb_dim) *d_img

        t_real = self.real_ent_emb(t_idx).view(-1, self.emb_dim) *d_real-\
                 self.img_ent_emb(t_idx).view(-1, self.emb_dim) *d_img

        r_real = self.real_rel_emb(r_idx).view(-1, self.emb_dim)

        h_img = self.real_ent_emb(h_idx).view(-1, self.emb_dim) *d_img+\
                 self.img_ent_emb(h_idx).view(-1,self.emb_dim) *d_real


        t_img = self.real_ent_emb(t_idx).view(-1, self.emb_dim) *d_img+\
                self.img_ent_emb(t_idx).view(-1,self.emb_dim) *d_real

        r_img = self.img_rel_emb(r_idx).view(-1, self.emb_dim)

        h = torch.cat([h_real, h_img], dim = 1)
        t = torch.cat([t_real, t_img], dim = 1)
        r = torch.cat([r_real, r_img], dim = 1)
        
        if entities:
            candidates = self.get_candidate_embedding(start_time_idx)
        else:
            candidates = self.rel_emb.weight.data.view(1, self.n_rel, self.emb_dim)
            candidates = candidates.expand(b_size, self.n_rel, self.emb_dim)

        return h, t, r, candidates

    def inference_scoring_function(self, proj_h, proj_t, r, start_time, end_time):
        b_size = proj_h.shape[0]
        r_real, r_img = torch.chunk(r, 2, -1)
        real_dim_size = r_real.shape[1]
        img_dim_size = r_img.shape[1]
        d_i = self.time_trans[start_time].unsqueeze(1).unsqueeze(1)
        if len(r.shape) == 2:
            if len(proj_t.shape) == 3:
                assert (len(proj_h.shape) == 2)
                proj_t_real, proj_t_img = torch.chunk(proj_t, 2, -1)
                proj_h_real, proj_h_img = torch.chunk(proj_h, 2, -1)
                
                hr_real = (proj_h_real + r_real).view(b_size, 1, r_real.shape[1])
                hr_img = (proj_h_img + r_img).view(b_size, 1, r_img.shape[1])

                if self.dissimilarity_type == 'L1':
                    out_real = torch.sum(torch.abs(hr_real - proj_t_real), -1)
                    out_img = torch.sum(torch.abs(hr_img + proj_t_img), -1)
                    out = out_real + out_img

                else:
                    out_real = torch.sum((hr_real + d_i - proj_t_real) ** 2, -1)
                    out_img = torch.sum((hr_img + d_i + proj_t_real) ** 2, -1)
                    out = torch.sqrt(out_img + out_real)
                return self.out_mult * out
            else:
                assert (len(proj_h.shape) == 3) & (len(proj_t.shape) == 2)
                # this is the head completion case in link prediction
                proj_t_real, proj_t_img = torch.chunk(proj_t, 2, -1)
                proj_h_real, proj_h_img = torch.chunk(proj_h, 2, -1)
                
                r_real = r_real.view(b_size, 1, real_dim_size)
                r_img = r_img.view(b_size, 1, img_dim_size)
                proj_t_real = proj_t_real.view(b_size, 1, real_dim_size)
                proj_t_img = proj_t_img.view(b_size, 1, img_dim_size)
                
                if self.dissimilarity_type == 'L1':
                    out_real = torch.sum(torch.abs(proj_h_real + r_real - proj_t_real), -1)
                    out_img = torch.sum(torch.abs(proj_h_img + r_img + proj_t_img), -1)
                    out = out_real + out_img

                else:
                    out_real = torch.sum((proj_h_real + r_real + d_i - proj_t_real) ** 2, -1)
                    out_img = torch.sum((proj_h_img + r_img + d_i + proj_t_real) ** 2, -1)
                    out = torch.sqrt(out_img + out_real)
                return self.out_mult * out
        elif len(r.shape) == 3:
            # this is the relation prediction case
            # Two cases possible:
            proj_h = proj_h.view(b_size, -1, self.emb_dim)
            proj_t = proj_t.view(b_size, -1, self.emb_dim)
            return - self.dissimilarity(proj_h + r, proj_t)

class ATISEModel(TempTranslationModel):

    def __init__(self, args):

        super().__init__(args.n_entities, args.n_relations, args.dissimilarity_type)

        self.emb_dim = args.emb_dim
        self.cmin = args.cmin
        self.cmax = args.cmax

        self.tem_total = args.tem_total
        self.dissimilarity_type = args.dissimilarity_type

        self.emb_E = init_embedding(self.n_ent, self.emb_dim)
        self.emb_E_var = init_embedding(self.n_ent, self.emb_dim)
        self.emb_R = init_embedding(self.n_rel, self.emb_dim)
        self.emb_R_var = init_embedding(self.n_rel, self.emb_dim)
        self.emb_TE = init_embedding(self.n_ent, self.emb_dim)
        self.alpha_E = init_embedding(self.n_ent, self.emb_dim)
        self.beta_E = init_embedding(self.n_ent, self.emb_dim)
        self.omega_E = init_embedding(self.n_ent, self.emb_dim)
        self.emb_TR = init_embedding(self.n_rel, self.emb_dim)
        self.alpha_R = init_embedding(self.n_rel, self.emb_dim)
        self.beta_R = init_embedding(self.n_rel, self.emb_dim)
        self.omega_R = init_embedding(self.n_rel, self.emb_dim)
        
        self.time_trans = None
        self.device = args.device

        r = 6 / np.sqrt(self.emb_dim)
        self.emb_E.weight.data.uniform_(-r, r)
        self.emb_E_var.weight.data.uniform_(self.cmin, self.cmax)
        self.emb_R.weight.data.uniform_(-r, r)
        self.emb_R_var.weight.data.uniform_(self.cmin, self.cmax)
        self.emb_TE.weight.data.uniform_(-r, r)
        self.alpha_E.weight.data.uniform_(0, 0)
        self.beta_E.weight.data.uniform_(0, 0)
        self.omega_E.weight.data.uniform_(-r, r)
        self.emb_TR.weight.data.uniform_(-r, r)
        self.alpha_R.weight.data.uniform_(0, 0)
        self.beta_R.weight.data.uniform_(0, 0)
        self.omega_R.weight.data.uniform_(-r, r)


    def scoring_function(self, h_idx, t_idx, r_idx, start_time_idx, end_time_idx):
        start_time_idx = self.time_trans[start_time_idx]
        h_i, t_i, r_i, d_i = h_idx, t_idx, r_idx, start_time_idx

        pi = 3.14159265358979323846
        h_mean = self.emb_E(h_i).view(-1, self.emb_dim) + \
            d_i.view(-1, 1) * self.alpha_E(h_i).view(-1, 1) * self.emb_TE(h_i).view(-1, self.emb_dim) \
            + self.beta_E(h_i).view(-1, self.emb_dim) * torch.sin(
            2 * pi * self.omega_E(h_i).view(-1, self.emb_dim) * d_i.view(-1, 1))
            
        t_mean = self.emb_E(t_i).view(-1, self.emb_dim) + \
            d_i.view(-1, 1) * self.alpha_E(t_i).view(-1, 1) * self.emb_TE(t_i).view(-1, self.emb_dim) \
            + self.beta_E(t_i).view(-1, self.emb_dim) * torch.sin(
            2 * pi * self.omega_E(t_i).view(-1, self.emb_dim) * d_i.view(-1, 1))
            
        r_mean = self.emb_R(r_i).view(-1, self.emb_dim) + \
            d_i.view(-1, 1) * self.alpha_R(r_i).view(-1, 1) * self.emb_TR(r_i).view(-1, self.emb_dim) \
            + self.beta_R(r_i).view(-1, self.emb_dim) * torch.sin(
            2 * pi * self.omega_R(r_i).view(-1, self.emb_dim) * d_i.view(-1, 1))


        h_var = self.emb_E_var(h_i).view(-1, self.emb_dim)
        t_var = self.emb_E_var(t_i).view(-1, self.emb_dim)
        r_var = self.emb_R_var(r_i).view(-1, self.emb_dim)

        out1 = torch.sum((h_var+t_var)/r_var, 1)+torch.sum(((r_mean-h_mean+t_mean)**2)/r_var, 1)-self.emb_dim
        out2 = torch.sum(r_var/(h_var+t_var), 1)+torch.sum(((h_mean-t_mean-r_mean)**2)/(h_var+t_var), 1)-self.emb_dim
        out = (out1+out2)/4
        

        return - out

    def normalize_parameters(self):
        self.ent_emb.weight.data = normalize(self.ent_emb.weight.data,
                                             p=2, dim=1)

    def get_embeddings(self):
        self.normalize_parameters()
        return self.ent_emb.weight.data, self.rel_emb.weight.data

    def get_inference_candidates(self):
        self.emb_E
        self.emb_E_var
        self.emb_TE
        self.alpha_E
        self.beta_E
        self.omega_E
        
    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, start_time_idx, end_time_idx, entities=True):
        start_time_idx = self.time_trans[start_time_idx]
        start_time_idx = self.time_trans[start_time_idx]
        h_i, t_i, r_i, d_i = h_idx, t_idx, r_idx, start_time_idx
        b_size = h_idx.shape[0]
        pi = 3.14159265358979323846
        h_mean = self.emb_E(h_i).view(-1, self.emb_dim) + \
            d_i.view(-1, 1) * self.alpha_E(h_i).view(-1, 1) * self.emb_TE(h_i).view(-1, self.emb_dim) \
            + self.beta_E(h_i).view(-1, self.emb_dim) * torch.sin(
            2 * pi * self.omega_E(h_i).view(-1, self.emb_dim) * d_i.view(-1, 1))
            
        t_mean = self.emb_E(t_i).view(-1, self.emb_dim) + \
            d_i.view(-1, 1) * self.alpha_E(t_i).view(-1, 1) * self.emb_TE(t_i).view(-1, self.emb_dim) \
            + self.beta_E(t_i).view(-1, self.emb_dim) * torch.sin(
            2 * pi * self.omega_E(t_i).view(-1, self.emb_dim) * d_i.view(-1, 1))
            
        r_mean = self.emb_R(r_i).view(-1, self.emb_dim) + \
            d_i.view(-1, 1) * self.alpha_R(r_i).view(-1, 1) * self.emb_TR(r_i).view(-1, self.emb_dim) \
            + self.beta_R(r_i).view(-1, self.emb_dim) * torch.sin(
            2 * pi * self.omega_R(r_i).view(-1, self.emb_dim) * d_i.view(-1, 1))


        h_var = self.emb_E_var(h_i).view(-1, self.emb_dim)
        t_var = self.emb_E_var(t_i).view(-1, self.emb_dim)
        r_var = self.emb_R_var(r_i).view(-1, self.emb_dim)

        out1 = torch.sum((h_var+t_var)/r_var, 1)+torch.sum(((r_mean-h_mean+t_mean)**2)/r_var, 1)-self.emb_dim
        out2 = torch.sum(r_var/(h_var+t_var), 1)+torch.sum(((h_mean-t_mean-r_mean)**2)/(h_var+t_var), 1)-self.emb_dim
        out = (out1+out2)/4

        if entities:
            candidates_emb_E = self.emb_E.weight.data.view(1, self.n_ent, self.emb_dim)
            candidates_emb_E = candidates_emb_E.expand(b_size, self.n_ent, self.emb_dim)
            candidates_emb_E_var = self.emb_E_var.weight.data.view(1, self.n_ent, self.emb_dim)
            candidates_emb_E_var = candidates_emb_E_var.expand(b_size, self.n_ent, self.emb_dim)
            candidates_emb_TE = self.emb_TE.weight.data.view(1, self.n_ent, self.emb_dim)
            candidates_emb_TE = candidates_emb_TE.expand(b_size, self.n_ent, self.emb_dim)
            candidates_alpha_E = self.alpha_E.weight.data.view(1, self.n_ent, self.emb_dim)
            candidates_alpha_E = candidates_alpha_E.expand(b_size, self.n_ent, self.emb_dim)
            candidates_beta_E = self.beta_E.weight.data.view(1, self.n_ent, self.emb_dim)
            candidates_beta_E = candidates_beta_E.expand(b_size, self.n_ent, self.emb_dim)
            candidates_omega_E = self.omega_E.weight.data.view(1, self.n_ent, self.emb_dim)
            candidates_omega_E = candidates_omega_E.expand(b_size, self.n_ent, self.emb_dim)
        else:
            candidates = self.rel_emb.weight.data.view(1, self.n_rel, self.emb_dim)
            candidates = candidates.expand(b_size, self.n_rel, self.emb_dim)

        return torch.cat([h_real, h_img], dim = 1), torch.cat([t_real, t_img], dim = 1), torch.cat([r_real, r_img], dim = 1), torch.cat([candidates_real_, candidates_img_], dim = -1)

    def inference_scoring_function(self, proj_h, proj_t, r, start_time, end_time):
        b_size = proj_h.shape[0]
        r_real = r[:, 0:self.emb_dim]
        r_img = r[:, self.emb_dim:]
        real_dim_size = r_real.shape[1]
        img_dim_size = r_img.shape[1]
        if len(r.shape) == 2:
            if len(proj_t.shape) == 3:
                assert (len(proj_h.shape) == 2)
                proj_t_real = proj_t[:,:, 0:self.emb_dim]
                proj_t_img = proj_t[:,:, self.emb_dim:]
                proj_h_real = proj_h[:, 0:self.emb_dim]
                proj_h_img = proj_h[:, self.emb_dim:]

                # this is the tail completion case in link prediction
                hr_real = (proj_h_real + r_real).view(b_size, 1, r_real.shape[1])
                hr_img = (proj_h_img + r_img).view(b_size, 1, r_img.shape[1])
                
                if self.dissimilarity_type == 'L1':
                    out_real = self.dissimilarity(hr_real, proj_t_real)
                    out_img = self.dissimilarity(hr_img, - proj_t_img)
                    out = out_real + out_img

                else:
                    out_real = self.dissimilarity(hr_real + start_time.unsqueeze(1).unsqueeze(1), proj_t_real)
                    out_img = self.dissimilarity(hr_img + start_time.unsqueeze(1).unsqueeze(1), - proj_t_real)
                    out = torch.sqrt(out_img + out_real)
                return - out
            else:
                assert (len(proj_h.shape) == 3) & (len(proj_t.shape) == 2)
                # this is the head completion case in link prediction
                proj_t_real = proj_t[:, 0:self.emb_dim]
                proj_t_img = proj_t[:, self.emb_dim:]
                proj_h_real = proj_h[:,:, 0:self.emb_dim]
                proj_h_img = proj_h[:,:, self.emb_dim:]

                r_real = r_real.view(b_size, 1, real_dim_size)
                r_img = r_img.view(b_size, 1, img_dim_size)
                proj_t_real = proj_t_real.view(b_size, 1, real_dim_size)
                proj_t_img = proj_t_img.view(b_size, 1, img_dim_size)
                if self.dissimilarity_type == 'L1':
                    out_real = self.dissimilarity(proj_h_real + r_real, proj_t_real)
                    out_img = self.dissimilarity(proj_h_img + r_img, - proj_t_img)
                    out = out_real + out_img
                else:
                    out_real = self.dissimilarity(proj_h_real + r_real + start_time.unsqueeze(1).unsqueeze(1), proj_t_real)
                    out_img = self.dissimilarity(proj_h_img + r_img + start_time.unsqueeze(1).unsqueeze(1), - proj_t_real)
                    out = torch.sqrt(out_img + out_real)
                return - out
        elif len(r.shape) == 3:
            # this is the relation prediction case
            # Two cases possible:
            proj_h = proj_h.view(b_size, -1, self.emb_dim)
            proj_t = proj_t.view(b_size, -1, self.emb_dim)
            return - self.dissimilarity(proj_h + r, proj_t)


def get_model(model_name, args):
    if model_name == 'TDistMultModel':
        return TDistMultModel(args)
    elif model_name == 'TTransEModel':
        return TTransEModel(args)
    elif model_name == 'DETransEModel':
        return DETransEModel(args)
    elif model_name == 'DEDistMultModel':
        return DEDistMultModel(args)
    if model_name == 'TeRoModel':
        return TeRoModel(args)
    elif model_name == 'DEDistMultModel':
        return DEDistMultModel(args)