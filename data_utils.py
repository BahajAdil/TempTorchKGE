

from utils import device
from utils import get_temporal_dictionaries
from torch.utils.data import Dataset
from torchkge.exceptions import SizeMismatchError, WrongArgumentsError, SanityError
from torchkge.utils.operations import get_dictionaries
from collections import defaultdict
import torch
from torch import tensor
from torch import cat, eq, int64, long, randperm, tensor, Tensor, zeros_like
from pandas import DataFrame
from os import environ, makedirs
from os.path import join
from os.path import exists, expanduser, join
import shutil


class TemporalKnowledgeGraph(Dataset):

    def __init__(self, df=None, time_mode = None,kg=None, ent2ix=None, rel2ix=None, time2ix = None,
                 time_trans = None,
                 dict_of_heads=None, dict_of_tails=None, dict_of_rels=None,
                 temp_dict_of_heads=None, temp_dict_of_tails=None, temp_dict_of_rels=None):

        if df is None:
            if kg is None:
                raise WrongArgumentsError("Please provide at least one "
                                          "argument of `df` and kg`")
            else:
                try:
                    assert (type(kg) == dict) & ('heads' in kg.keys()) & \
                           ('tails' in kg.keys()) & \
                           ('relations' in kg.keys())& \
                            ('start_time' in kg.keys())& \
                            ('end_time' in kg.keys())
                    
                except AssertionError:
                    raise WrongArgumentsError("Keys in the `kg` dict should "
                                              "contain `heads`, `tails`, "
                                              "`relations`.")
                try:
                    assert (rel2ix is not None) & (ent2ix is not None)
                except AssertionError:
                    raise WrongArgumentsError("Please provide the two "
                                              "dictionaries ent2ix and rel2ix "
                                              "if building from `kg`.")
        else:
            if kg is not None:
                raise WrongArgumentsError("`df` and kg` arguments should not "
                                          "both be provided.")

        if ent2ix is None:
            self.ent2ix = get_dictionaries(df, ent=True)
        else:
            self.ent2ix = ent2ix

        if rel2ix is None:
            self.rel2ix = get_dictionaries(df, ent=False)
        else:
            self.rel2ix = rel2ix
            
        if time_mode is not None:
            self.time_mode = time_mode
            
        if time2ix is None:
            self.time2ix, self.time_trans = get_temporal_dictionaries(df, mode = self.time_mode)
        else:
            self.time2ix, self.time_trans = time2ix, time_trans
        
        self.n_ent = max(self.ent2ix.values()) + 1
        self.n_rel = max(self.rel2ix.values()) + 1
        time_val = list(self.time2ix.values())
        
        if isinstance(time_val[0], torch.Tensor):
            self.n_time = int(torch.cat(time_val).max()) + 1
        else: 
            self.n_time = max(time_val) + 1
            
#         print('self.n_time: ',self.n_time)
        if df is not None:
            # build kg from a pandas dataframe
            self.n_facts = len(df)
            self.head_idx = tensor(df['from'].map(self.ent2ix).values).long()
            self.tail_idx = tensor(df['to'].map(self.ent2ix).values).long()
            self.relations = tensor(df['rel'].map(self.rel2ix).values).long()
#             self.start_time = tensor(df['start_time'].map(self.rel2ix).values).long()
#             self.end_time = tensor(df['end_time'].map(self.rel2ix).values).long()
#             print(self.time2ix)
            self.start_time = list(df['start_time'].map(self.time2ix).values)
#             print(self.start_time)
#             print('-'*22)
#             print("type(self.start_time[0]): ",type(self.start_time[0]))
#             print('type(self.start_time): ',type(self.start_time))
#             if isinstance(self.start_time[0], torch.Tensor):
#                 print('llllll')
            if isinstance(self.start_time, list) & isinstance(self.start_time[0], torch.Tensor):
                self.start_time = torch.stack(self.start_time)
            else:
                self.start_time = torch.tensor(self.start_time)
            
            self.start_time = self.start_time.long()
            
            self.end_time = list(df['end_time'].map(self.time2ix).values)
            if isinstance(self.end_time, list) & isinstance(self.end_time[0], torch.Tensor):
                self.end_time = torch.stack(self.end_time)
            else:
                self.end_time = torch.tensor(self.end_time)   
            self.end_time = self.end_time.long()
            
        else:
            # build kg from another kg
            self.n_facts = kg['heads'].shape[0]
            self.head_idx = kg['heads']
            self.tail_idx = kg['tails']
            self.relations = kg['relations']
            self.start_time = kg['start_time']
            self.end_time = kg['end_time']

        if dict_of_heads is None or dict_of_tails is None or dict_of_rels is None:
            self.dict_of_heads = defaultdict(set)
            self.dict_of_tails = defaultdict(set)
            self.dict_of_rels = defaultdict(set)
            self.temp_dict_of_heads = defaultdict(set)
            self.temp_dict_of_tails = defaultdict(set)
            self.temp_dict_of_rels = defaultdict(set)
#             self.dict_of_start_time = defaultdict(set)
#             self.dict_of_end_time = defaultdict(set)
            self.evaluate_dicts()

        else:
            self.dict_of_heads = dict_of_heads
            self.dict_of_tails = dict_of_tails
            self.dict_of_rels = dict_of_rels
            self.temp_dict_of_heads = temp_dict_of_heads
            self.temp_dict_of_tails = temp_dict_of_tails
            self.temp_dict_of_rels = temp_dict_of_rels
#             self.dict_of_start_time = dict_of_start_time
#             self.dict_of_end_time = dict_of_end_time
        try:
            self.sanity_check()
        except AssertionError:
            raise SanityError("Please check the sanity of arguments.")

    def __len__(self):
        return self.n_facts

    def __getitem__(self, item):
        return (self.head_idx[item].item(),
                self.tail_idx[item].item(),
                self.relations[item].item())

    def sanity_check(self):
        assert (type(self.dict_of_heads) == defaultdict) & \
               (type(self.dict_of_tails) == defaultdict) & \
               (type(self.dict_of_rels) == defaultdict) & \
                (type(self.temp_dict_of_heads) == defaultdict) & \
               (type(self.temp_dict_of_tails) == defaultdict) & \
               (type(self.temp_dict_of_rels) == defaultdict)
        assert (type(self.ent2ix) == dict) & (type(self.rel2ix) == dict)
        assert (len(self.ent2ix) == self.n_ent) & \
               (len(self.rel2ix) == self.n_rel)
        assert (type(self.head_idx) == Tensor) & \
               (type(self.tail_idx) == Tensor) & \
               (type(self.relations) == Tensor)
        assert (self.head_idx.dtype == int64) & \
               (self.tail_idx.dtype == int64) & (self.relations.dtype == int64)
        assert (len(self.head_idx) == len(self.tail_idx) == len(self.relations))

    def split_kg(self, share=0.8, sizes=None, validation=False):
        if sizes is not None:
            try:
                if len(sizes) == 3:
                    try:
                        assert (sizes[0] + sizes[1] + sizes[2] == self.n_facts)
                    except AssertionError:
                        raise WrongArgumentsError('Sizes should sum to the '
                                                  'number of facts.')
                elif len(sizes) == 2:
                    try:
                        assert (sizes[0] + sizes[1] == self.n_facts)
                    except AssertionError:
                        raise WrongArgumentsError('Sizes should sum to the '
                                                  'number of facts.')
                else:
                    raise SizeMismatchError('Tuple `sizes` should be of '
                                            'length 2 or 3.')
            except AssertionError:
                raise SizeMismatchError('Tuple `sizes` should sum up to the '
                                        'number of facts in the knowledge '
                                        'graph.')
        else:
            assert share < 1

        if ((sizes is not None) and (len(sizes) == 3)) or \
                ((sizes is None) and validation):
            # return training, validation and a testing graphs

            if (sizes is None) and validation:
                mask_tr, mask_val, mask_te = self.get_mask(share,
                                                           validation=True)
            else:
                mask_tr = cat([tensor([1 for _ in range(sizes[0])]),
                               tensor([0 for _ in range(sizes[1] + sizes[2])])]).bool()
                mask_val = cat([tensor([0 for _ in range(sizes[0])]),
                                tensor([1 for _ in range(sizes[1])]),
                                tensor([0 for _ in range(sizes[2])])]).bool()
                mask_te = ~(mask_tr | mask_val)

            return (TemporalKnowledgeGraph(
                        kg={'heads': self.head_idx[mask_tr],
                            'tails': self.tail_idx[mask_tr],
                            'relations': self.relations[mask_tr],
                           'start_time': self.start_time[mask_tr],
                           'end_time': self.end_time[mask_tr]},
                            ent2ix=self.ent2ix, rel2ix=self.rel2ix, time2ix = self.time2ix,
                            time_trans = self.time_trans,
                            dict_of_heads=self.dict_of_heads,
                            dict_of_tails=self.dict_of_tails,
                            dict_of_rels=self.dict_of_rels,
                            temp_dict_of_heads = self.temp_dict_of_heads,
                            temp_dict_of_tails = self.temp_dict_of_tails,
                            temp_dict_of_rels = self.temp_dict_of_rels
                            ),
                    TemporalKnowledgeGraph(
                        kg={'heads': self.head_idx[mask_val],
                            'tails': self.tail_idx[mask_val],
                            'relations': self.relations[mask_val],
                           'start_time':  self.start_time[mask_val],
                           'end_time': self.end_time[mask_val]},
                        ent2ix=self.ent2ix, rel2ix=self.rel2ix, time2ix = self.time2ix,
                        time_trans = self.time_trans,
                        dict_of_heads=self.dict_of_heads,
                        dict_of_tails=self.dict_of_tails,
                        dict_of_rels=self.dict_of_rels,
                        temp_dict_of_heads = self.temp_dict_of_heads,
                        temp_dict_of_tails = self.temp_dict_of_tails,
                        temp_dict_of_rels = self.temp_dict_of_rels),
                    TemporalKnowledgeGraph(
                        kg={'heads': self.head_idx[mask_te],
                            'tails': self.tail_idx[mask_te],
                            'relations': self.relations[mask_te],
                           'start_time': self.start_time[mask_te],
                           'end_time': self.end_time[mask_te]},
                        ent2ix=self.ent2ix, rel2ix=self.rel2ix, time2ix = self.time2ix,
                        time_trans = self.time_trans,
                        dict_of_heads=self.dict_of_heads,
                        dict_of_tails=self.dict_of_tails,
                        dict_of_rels=self.dict_of_rels,
                        temp_dict_of_heads = self.temp_dict_of_heads,
                        temp_dict_of_tails = self.temp_dict_of_tails,
                        temp_dict_of_rels = self.temp_dict_of_rels
                        ))
        else:
            # return training and testing graphs

            assert (((sizes is not None) and len(sizes) == 2) or
                    ((sizes is None) and not validation))
            if sizes is None:
                mask_tr, mask_te = self.get_mask(share, validation=False)
            else:
                mask_tr = cat([tensor([1 for _ in range(sizes[0])]),
                               tensor([0 for _ in range(sizes[1])])]).bool()
                mask_te = ~mask_tr
            return (TemporalKnowledgeGraph(
                        kg={'heads': self.head_idx[mask_tr],
                            'tails': self.tail_idx[mask_tr],
                            'relations': self.relations[mask_tr],
                           'start_time': self.start_time[mask_tr],
                           'end_time': self.end_time[mask_tr]},
                        ent2ix=self.ent2ix, rel2ix=self.rel2ix, time2ix = self.time2ix,
                        time_trans = self.time_trans,
                        dict_of_heads=self.dict_of_heads,
                        dict_of_tails=self.dict_of_tails,
                        dict_of_rels=self.dict_of_rels,
                        temp_dict_of_heads = self.temp_dict_of_heads,
                        temp_dict_of_tails = self.temp_dict_of_tails,
                        temp_dict_of_rels = self.temp_dict_of_rels
                        ),
                    TemporalKnowledgeGraph(
                        kg={'heads': self.head_idx[mask_te],
                            'tails': self.tail_idx[mask_te],
                            'relations': self.relations[mask_te],
                           'start_time': self.start_time[mask_te],
                           'end_time': self.end_time[mask_te]},
                        ent2ix=self.ent2ix, rel2ix=self.rel2ix, time2ix = self.time2ix,
                        time_trans = self.time_trans,
                        dict_of_heads=self.dict_of_heads,
                        dict_of_tails=self.dict_of_tails,
                        dict_of_rels=self.dict_of_rels,
                        temp_dict_of_heads = self.temp_dict_of_heads,
                        temp_dict_of_tails = self.temp_dict_of_tails,
                        temp_dict_of_rels = self.temp_dict_of_rels
                        ))

    def get_mask(self, share, validation=False):

        uniques_r, counts_r = self.relations.unique(return_counts=True)
        uniques_e, _ = cat((self.head_idx,
                            self.tail_idx)).unique(return_counts=True)

        mask = zeros_like(self.relations).bool()
        if validation:
            mask_val = zeros_like(self.relations).bool()

        # splitting relations among subsets
        for i, r in enumerate(uniques_r):
            rand = randperm(counts_r[i].item())

            # list of indices k such that relations[k] == r
            sub_mask = eq(self.relations, r).nonzero(as_tuple=False)[:, 0]

            assert len(sub_mask) == counts_r[i].item()

            if validation:
                train_size, val_size, test_size = self.get_sizes(counts_r[i].item(),
                                                                 share=share,
                                                                 validation=True)
                mask[sub_mask[rand[:train_size]]] = True
                mask_val[sub_mask[rand[train_size:train_size + val_size]]] = True

            else:
                train_size, test_size = self.get_sizes(counts_r[i].item(),
                                                       share=share,
                                                       validation=False)
                mask[sub_mask[rand[:train_size]]] = True

        # adding missing entities to the train set
        u = cat((self.head_idx[mask], self.tail_idx[mask])).unique()
        if len(u) < self.n_ent:
            missing_entities = tensor(list(set(uniques_e.tolist()) -
                                           set(u.tolist())), dtype=long)
            for e in missing_entities:
                sub_mask = ((self.head_idx == e) |
                            (self.tail_idx == e)).nonzero(as_tuple=False)[:, 0]
                rand = randperm(len(sub_mask))
                sizes = self.get_sizes(mask.shape[0],
                                       share=share,
                                       validation=validation)
                mask[sub_mask[rand[:sizes[0]]]] = True
                if validation:
                    mask_val[sub_mask[rand[:sizes[0]]]] = False

        if validation:
            assert not (mask & mask_val).any().item()
            return mask, mask_val, ~(mask | mask_val)
        else:
            return mask, ~mask

    @staticmethod
    def get_sizes(count, share, validation=False):
        if count == 1:
            if validation:
                return 1, 0, 0
            else:
                return 1, 0
        if count == 2:
            if validation:
                return 1, 1, 0
            else:
                return 1, 1

        n_train = int(count * share)
        assert n_train < count
        if n_train == 0:
            n_train += 1

        if not validation:
            return n_train, count - n_train
        else:
            if count - n_train == 1:
                n_train -= 1
                return n_train, 1, 1
            else:
                n_val = int(int(count - n_train) / 2)
                return n_train, n_val, count - n_train - n_val

    def evaluate_dicts(self):
        for i in range(self.n_facts):
            self.dict_of_heads[(self.tail_idx[i].item(),
                                self.relations[i].item())].add(self.head_idx[i].item())
            self.dict_of_tails[(self.head_idx[i].item(),
                                self.relations[i].item())].add(self.tail_idx[i].item())
            self.dict_of_rels[(self.head_idx[i].item(),
                               self.tail_idx[i].item())].add(self.relations[i].item())
            
            self.temp_dict_of_heads[(self.tail_idx[i].item(),
                                    self.relations[i].item(),
                                     self.start_time[i].item(),
                                     self.end_time[i].item())].add(self.head_idx[i].item())
            self.temp_dict_of_tails[(self.head_idx[i].item(),
                                self.relations[i].item(),
                                     self.start_time[i].item(),
                                     self.end_time[i].item())].add(self.tail_idx[i].item())
            self.temp_dict_of_rels[(self.head_idx[i].item(),
                                   self.tail_idx[i].item(),
                                    self.start_time[i].item(),
                                     self.end_time[i].item())].add(self.relations[i].item())

    def get_df(self):
        ix2ent = {v: k for k, v in self.ent2ix.items()}
        ix2rel = {v: k for k, v in self.rel2ix.items()}

        df = DataFrame(cat((self.head_idx.view(1, -1),
                            self.tail_idx.view(1, -1),
                            self.relations.view(1, -1))).transpose(0, 1).numpy(),
                       columns=['from', 'to', 'rel'])

        df['from'] = df['from'].apply(lambda x: ix2ent[x])
        df['to'] = df['to'].apply(lambda x: ix2ent[x])
        df['rel'] = df['rel'].apply(lambda x: ix2rel[x])

        return df


class SmallKG(Dataset):
    def __init__(self, heads, tails, relations):
        assert heads.shape == tails.shape == relations.shape
        self.head_idx = heads
        self.tail_idx = tails
        self.relations = relations
        self.length = heads.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.head_idx[item].item(), self.tail_idx[item].item(), self.relations[item].item()



def get_data_home(data_home=None):
    if data_home is None:
        data_home = environ.get('TORCHKGE_DATA',
                                join('~', 'torchkge_data'))
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home


def clear_data_home(data_home=None):
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


def get_n_batches(n, b_size):
    n_batch = n // b_size
    if n % b_size > 0:
        n_batch += 1
    return n_batch


class TempDataLoader:
    """This class is inspired from :class:`torch.utils.dataloader.DataLoader`.
    It is however way simpler.

    """
    def __init__(self, kg, batch_size, use_cuda=None):

        self.h = kg.head_idx
        self.t = kg.tail_idx
        self.r = kg.relations
        self.start_time = kg.start_time
        self.end_time = kg.end_time

        self.use_cuda = use_cuda
        self.batch_size = batch_size

        if use_cuda is not None and use_cuda == 'all':
            self.h = self.h.to(device)
            self.t = self.t.to(device)
            self.r = self.r.to(device)
            self.start_time = self.start_time.to(device)
            self.end_time = self.end_time.to(device)
            
    def __len__(self):
        return get_n_batches(len(self.h), self.batch_size)

    def __iter__(self):
        return _TempDataLoaderIter(self)


class _TempDataLoaderIter:
    def __init__(self, loader):
        self.h = loader.h
        self.t = loader.t
        self.r = loader.r
        self.start_time = loader.start_time
        self.end_time = loader.end_time
        
        self.use_cuda = loader.use_cuda
        self.batch_size = loader.batch_size

        self.n_batches = get_n_batches(len(self.h), self.batch_size)
        self.current_batch = 0

    def __next__(self):
        if self.current_batch == self.n_batches:
            raise StopIteration
        else:
            i = self.current_batch
            self.current_batch += 1

            tmp_h = self.h[i * self.batch_size: (i + 1) * self.batch_size]
            tmp_t = self.t[i * self.batch_size: (i + 1) * self.batch_size]
            tmp_r = self.r[i * self.batch_size: (i + 1) * self.batch_size]
            tmp_start_time = self.start_time[i * self.batch_size: (i + 1) * self.batch_size]
            tmp_end_time = self.end_time[i * self.batch_size: (i + 1) * self.batch_size]
            
            if self.use_cuda is not None and self.use_cuda == 'batch':
                return tmp_h.to(device), tmp_t.to(device), tmp_r.to(device), tmp_start_time.to(device), tmp_end_time.to(device)
            else:
                return tmp_h, tmp_t, tmp_r, tmp_start_time, tmp_end_time

    def __iter__(self):
        return self
