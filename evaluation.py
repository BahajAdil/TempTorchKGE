

from torch import empty, zeros, cat
from tqdm import tqdm
from data_utils import TempDataLoader
from utils import device
from eval_utils import filter_scores, get_rank
from torchkge.exceptions import NotYetEvaluatedError

class TemporalLinkPredictionEvaluator(object):

    def __init__(self, model, knowledge_graph):
        self.model = model
        self.kg = knowledge_graph

        self.rank_true_heads = empty(size=(knowledge_graph.n_facts,)).long()
        self.rank_true_tails = empty(size=(knowledge_graph.n_facts,)).long()
        self.filt_rank_true_heads = empty(size=(knowledge_graph.n_facts,)).long()
        self.filt_rank_true_tails = empty(size=(knowledge_graph.n_facts,)).long()

        self.evaluated = False

    def evaluate(self, b_size, verbose=True):
        use_cuda = next(self.model.parameters()).is_cuda

        if use_cuda:
            dataloader = TempDataLoader(self.kg, batch_size=b_size, use_cuda='batch')
            self.rank_true_heads = self.rank_true_heads.to(device)
            self.rank_true_tails = self.rank_true_tails.to(device)
            self.filt_rank_true_heads = self.filt_rank_true_heads.to(device)
            self.filt_rank_true_tails = self.filt_rank_true_tails.to(device)
        else:
            dataloader = TempDataLoader(self.kg, batch_size=b_size)

        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                             unit='batch', disable=(not verbose),
                             desc='Link prediction evaluation'):
            h_idx, t_idx, r_idx, start_time, end_time = batch[0], batch[1], batch[2], batch[3], batch[4]
                        
            h_emb, t_emb, r_emb, candidates = self.model.inference_prepare_candidates(h_idx, t_idx, r_idx, start_time, end_time, entities=True)

            scores = self.model.inference_scoring_function(h_emb, candidates, r_emb, start_time, end_time)
            filt_scores = filter_scores(scores, self.kg.temp_dict_of_tails, h_idx, r_idx, start_time, end_time, t_idx)
            self.rank_true_tails[i * b_size: (i + 1) * b_size] = get_rank(scores, t_idx).detach()
            self.filt_rank_true_tails[i * b_size: (i + 1) * b_size] = get_rank(filt_scores, t_idx).detach()

            scores = self.model.inference_scoring_function(candidates, t_emb, r_emb, start_time, end_time)
            filt_scores = filter_scores(scores, self.kg.temp_dict_of_heads, t_idx, r_idx, start_time, end_time, h_idx)
            self.rank_true_heads[i * b_size: (i + 1) * b_size] = get_rank(scores, h_idx).detach()
            self.filt_rank_true_heads[i * b_size: (i + 1) * b_size] = get_rank(filt_scores, h_idx).detach()

        self.evaluated = True

        if use_cuda:
            self.rank_true_heads = self.rank_true_heads.cpu()
            self.rank_true_tails = self.rank_true_tails.cpu()
            self.filt_rank_true_heads = self.filt_rank_true_heads.cpu()
            self.filt_rank_true_tails = self.filt_rank_true_tails.cpu()

    def mean_rank(self):
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        sum_ = (self.rank_true_heads.float().mean() +
                self.rank_true_tails.float().mean()).item()
        filt_sum = (self.filt_rank_true_heads.float().mean() +
                    self.filt_rank_true_tails.float().mean()).item()
        # return sum_ / 2, filt_sum / 2
        return {'mr':sum_ / 2, 'filt_mr':filt_sum / 2}

    def hit_at_k_heads(self, k=10):
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        head_hit = (self.rank_true_heads <= k).float().mean()
        filt_head_hit = (self.filt_rank_true_heads <= k).float().mean()

        # return head_hit.item(), filt_head_hit.item()
        return {'head_hit_'+str(k):head_hit.item(), 'filt_head_hit_'+str(k): filt_head_hit.item()}

    def hit_at_k_tails(self, k=10):
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        tail_hit = (self.rank_true_tails <= k).float().mean()
        filt_tail_hit = (self.filt_rank_true_tails <= k).float().mean()

        # return tail_hit.item(), filt_tail_hit.item()
        return {'tail_hit_'+str(k):tail_hit.item(), 'filt_tail_hit_'+str(k):filt_tail_hit.item()}

    def hit_at_k(self, k=10):
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')

        head_hit = self.hit_at_k_heads(k=k)
        head_hit, filt_head_hit = head_hit['head_hit_'+str(k)], head_hit['filt_head_hit_'+str(k)]

        tail_hit = self.hit_at_k_tails(k=k)
        tail_hit, filt_tail_hit = tail_hit['tail_hit_'+str(k)], tail_hit['filt_tail_hit_'+str(k)]

        # return (head_hit + tail_hit) / 2, (filt_head_hit + filt_tail_hit) / 2
        return {'hit_'+str(k): (head_hit + tail_hit) / 2, 'filt_hit_'+str(k): (filt_head_hit + filt_tail_hit) / 2}

    def mrr(self):
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        res = {}
        head_mrr = (self.rank_true_heads.float()**(-1)).mean()
        res['head_mrr'] = head_mrr.item()
        tail_mrr = (self.rank_true_tails.float()**(-1)).mean()
        res['tail_mrr'] = tail_mrr.item()
        filt_head_mrr = (self.filt_rank_true_heads.float()**(-1)).mean()
        res['filt_head_mrr'] = filt_head_mrr.item()
        filt_tail_mrr = (self.filt_rank_true_tails.float()**(-1)).mean()    
        res['filt_tail_mrr'] = filt_tail_mrr.item()
        res['mrr'] = (head_mrr + tail_mrr).item() / 2
        res['filt_mrr'] = (filt_head_mrr + filt_tail_mrr).item() / 2
        return res

    def get_results(self, k=None):
        res = {}
        if k is None:
            k = 10

        for i in range(1, k + 1):
            hits_res = self.hit_at_k(k=i)
            res.update(hits_res)

        mr_res = self.mean_rank()
        res.update(mr_res)
        mrr_res = self.mrr()
        res.update(mrr_res)
        return res

    def print_results(self, k=None, n_digits=3):
        if k is None:
            k = 10

        if k is not None and type(k) == int:
            print('Hit@{} : {} \t\t Filt. Hit@{} : {}'.format(
                k, round(self.hit_at_k(k=k)[0], n_digits),
                k, round(self.hit_at_k(k=k)[1], n_digits)))
        if k is not None and type(k) == list:
            for i in k:
                print('Hit@{} : {} \t\t Filt. Hit@{} : {}'.format(
                    i, round(self.hit_at_k(k=i)[0], n_digits),
                    i, round(self.hit_at_k(k=i)[1], n_digits)))

        print('Mean Rank : {} \t Filt. Mean Rank : {}'.format(
            int(self.mean_rank()[0]), int(self.mean_rank()[1])))
        print('MRR : {} \t\t Filt. MRR : {}'.format(
            round(self.mrr()[0], n_digits), round(self.mrr()[1], n_digits)))
