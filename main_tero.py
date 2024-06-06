



from train_loop import TrainLoop
from utils import Params
from data import get_data
from utils import device

dataset  = 'icews14'
model_name = 'TeRoModel' # 'TTransEModel','TDistMultModel'
args = Params()
args.n_epochs = 1
args.device = device
args.data_home = '/home/pctp/Desktop/Projects/tempTorchKGE/data/'
args.model_save_name = model_name + '.bt'
args.loss_name = 'MarginLoss'
args.model_path = '/home/pctp/Desktop/Projects/tempTorchKGE/best_models'
args.time_mode = 'simple'
args.dissimilarity_type = 'L2'
args.dropout = 0.2
args.model_name = model_name 
args.dataset = dataset
kg_train, kg_val, kg_test = get_data(args.dataset, args.time_mode, args.data_home)
args.n_entities = kg_train.n_ent
args.n_relations = kg_train.n_rel
args.tem_total = kg_train.n_time
args.s_rep_ratio = 0.2
args.lr = 0.001
args.margin = 5
args.n_neg = 10
args.b_size = 2000
args.emb_dim = 100
print('|'.join(str(k)+':'+str(v) for k, v in args.__dict__.items()))
tmp_trainlp = TrainLoop(kg_train, kg_val, kg_test, args)
train_evolution = tmp_trainlp.run()
test_res = tmp_trainlp.test()

print('results: ', test_res)
