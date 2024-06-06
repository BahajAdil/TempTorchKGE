

from utils import device
from models import get_model
from losses import get_loss
from torch import cuda
from torch.optim import Adam
from torchkge.sampling import BernoulliNegativeSampler, UniformNegativeSampler
from tqdm import tqdm
from data_utils import TempDataLoader
import torch
from os.path import join
import os
from evaluation import TemporalLinkPredictionEvaluator


class TrainLoop():
    def __init__(self, kg_train, kg_val, kg_test, args):
        self.args = args

        self.kg_train = kg_train
        self.kg_val = kg_val
        self.kg_test = kg_test
        self.n_epochs = args.n_epochs
        self.model_name = args.model_name
        self.emb_dim = args.emb_dim
        self.lr = args.lr
        self.n_epochs = args.n_epochs
        self.b_size = args.b_size
        self.margin = args.margin
        self.model_save_name = args.model_save_name
        self.device = device
        self.model_path = args.model_path
        self.n_neg = args.n_neg

#         self.model = TransEModel(self.emb_dim, self.kg_train.n_ent, self.kg_train.n_rel, dissimilarity_type='L2')
        self.model = get_model(args.model_name, args)
        self.model.time_trans = kg_train.time_trans.to(device)
        self.criterion = get_loss(args.loss_name, args)

        # Move everything to CUDA if available
        if cuda.is_available():
            cuda.empty_cache()
            self.model = self.model.to(self.device)
            if hasattr(self.criterion, 'to'):
                self.criterion = self.criterion.to(self.device)

        # Define the torch optimizer to be used
        self.optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0)
#         self.sampler = BernoulliNegativeSampler(self.kg_train, n_neg = self.n_neg)
        self.sampler = UniformNegativeSampler(self.kg_train, n_neg = self.n_neg)
        self.dataloader = TempDataLoader(self.kg_train, batch_size=self.b_size, use_cuda=None)

    def save_model(self,):
        torch.save(self.model.state_dict(), join(self.model_path,self.model_save_name))

    def evaluation(self, ):
        with torch.no_grad():
            evaluator = TemporalLinkPredictionEvaluator(self.model, self.kg_val)
            evaluator.evaluate(b_size=50, verbose=False)
            val_res = evaluator.get_results()
            val_filt_mrr = val_res['filt_mrr']
            # print(val_filt_mrr)
            return val_filt_mrr, val_res
    
    def test(self,):
        with torch.no_grad():
            self.model = self.model.to('cpu')
            del self.model
            torch.cuda.empty_cache()
            self.model = get_model(model_name = self.model_name, args = self.args)
            self.model.load_state_dict(torch.load(join(self.model_path,self.model_save_name)))
            self.model.time_trans = self.kg_train.time_trans.to(device)
            
            os.remove(join(self.model_path,self.model_save_name))
            self.model = self.model.to(device=device)
            evaluator = TemporalLinkPredictionEvaluator(self.model, self.kg_test)
            evaluator.evaluate(b_size=32)
            test_res = evaluator.get_results()
            return test_res

    def run(self):
        
        iterator = tqdm(range(self.n_epochs), unit='epoch')
        eval_epoch = 1
        best_val_mrr = -100
        best_epoch = 0
        patience_max = 50
        patience_counter = 0
        train_evolution = []

        for epoch in iterator:
            running_loss = 0.0
            epoch_log = {}
            for i, batch in enumerate(self.dataloader):
                h, t, r, start_time, end_time = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device)
                # print('h.shape ooooo ',h.shape)
                n_h, n_t = self.sampler.corrupt_batch(h, t, r)                
                self.optimizer.zero_grad()

                pos, neg = self.model(h, t, r, start_time, end_time, n_h, n_t)
                loss = self.criterion(pos, neg)                
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            epoch_log['avg_train_loss'] = running_loss / len(self.dataloader)
            if epoch % eval_epoch == 0:
                val_mrr,  val_res = self.evaluation()
                epoch_log['val_res'] = val_res
                if (best_val_mrr < val_mrr) & (val_mrr<float('inf')):
                    best_val_mrr = val_mrr
                    best_epoch = epoch + 1
                    # save the model
                    self.save_model()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter == patience_max:
                        break
            train_evolution.append(epoch_log)
            iterator.set_description(
                'Epoch {} | mean loss: {:.5f}| best mrr: {:.5f} | Best Epoch {} '.format(epoch + 1,
                                                      running_loss / len(self.dataloader), best_val_mrr, best_epoch))

        # self.model.normalize_parameters()
        return train_evolution
