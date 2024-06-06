

from torch.nn import Module

from torchkge.utils.dissimilarities import l1_dissimilarity, l2_dissimilarity, \
    l1_torus_dissimilarity, l2_torus_dissimilarity, el2_torus_dissimilarity



class TempModel(Module):
    def __init__(self, n_entities, n_relations):
        super().__init__()
        self.n_ent = n_entities
        self.n_rel = n_relations

    def forward(self, heads, tails, relations, start_time, end_time, negative_heads, negative_tails, negative_relations=None):
        pos = self.scoring_function(heads, tails, relations, start_time, end_time)

        if negative_relations is None:
            negative_relations = relations

        if negative_heads.shape[0] > negative_relations.shape[0]:
            # in that case, several negative samples are sampled from each fact
            n_neg = int(negative_heads.shape[0] / negative_relations.shape[0])
            # print('pos.shape: ', pos.shape)
            pos = pos.repeat(n_neg)
            neg = self.scoring_function(negative_heads,
                                        negative_tails,
                                        negative_relations.repeat(n_neg),
                                        start_time.repeat(n_neg, 1) if start_time.dim() ==2 & start_time.shape[-1]>1 else start_time.repeat(n_neg),
                                        end_time.repeat(n_neg, 1) if end_time.dim() ==2 & end_time.shape[-1]>1 else end_time.repeat(n_neg))
        else:
            neg = self.scoring_function(negative_heads,
                                        negative_tails,
                                        negative_relations, start_time, end_time)

        return pos, neg

    def scoring_function(self, h_idx, t_idx, r_idx, start_time_idx, end_time_idx):
        raise NotImplementedError

    def normalize_parameters(self):
        raise NotImplementedError

    def get_embeddings(self):
        raise NotImplementedError

    def inference_scoring_function(self, h, t, r, time):
        raise NotImplementedError

    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, entities=True):
        raise NotImplementedError


class TempTranslationModel(TempModel):
    def __init__(self, n_entities, n_relations, dissimilarity_type):
        super().__init__(n_entities, n_relations)

        assert dissimilarity_type in ['L1', 'L2', 'torus_L1', 'torus_L2',
                                      'torus_eL2']

        if dissimilarity_type == 'L1':
            self.dissimilarity = l1_dissimilarity
        elif dissimilarity_type == 'L2':
            self.dissimilarity = l2_dissimilarity
        elif dissimilarity_type == 'torus_L1':
            self.dissimilarity = l1_torus_dissimilarity
        elif dissimilarity_type == 'torus_L2':
            self.dissimilarity = l2_torus_dissimilarity
        else:
            self.dissimilarity = el2_torus_dissimilarity

    def scoring_function(self, h_idx, t_idx, r_idx, start_time_idx, end_time_idx):
        """See torchkge.models.interfaces.Models.
        """
        raise NotImplementedError

    def normalize_parameters(self):
        """See torchkge.models.interfaces.Models.
        """
        raise NotImplementedError

    def get_embeddings(self):
        """See torchkge.models.interfaces.Models.
        """
        raise NotImplementedError

    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, entities=True):
        raise NotImplementedError

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


class TempBilinearModel(TempModel):

    def __init__(self, emb_dim, n_entities, n_relations):
        super().__init__(n_entities, n_relations)
        self.emb_dim = emb_dim

    def scoring_function(self, h_idx, t_idx, r_idx, start_time_idx, end_time_idx):
        """See torchkge.models.interfaces.Models.
        """
        raise NotImplementedError

    def normalize_parameters(self):
        """See torchkge.models.interfaces.Models.
        """
        raise NotImplementedError

    def get_embeddings(self):
        """See torchkge.models.interfaces.Models.
        """
        raise NotImplementedError

    def inference_scoring_function(self, h, t, r, start_time, end_time):
        """See torchkge.models.interfaces.Models.
        """
        raise NotImplementedError

    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, start_time, end_time, entities=True):
        """See torchkge.models.interfaces.Models.
        """
        raise NotImplementedError