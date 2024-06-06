

from pandas import read_csv, concat
import pandas as pd
from data_utils import TemporalKnowledgeGraph

def load_icews14(data_home=None, time_mode=None):
    data_path = data_home + '/icews14'

    df1 = read_csv(data_path + '/train.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to', 'start_time', 'end_time'])
    df2 = read_csv(data_path + '/valid.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to', 'start_time', 'end_time'])
    df3 = read_csv(data_path + '/test.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to', 'start_time', 'end_time'])
    df = concat([df1, df2, df3])
    kg = TemporalKnowledgeGraph(df, time_mode = time_mode)

    return kg.split_kg(sizes=(len(df1), len(df2), len(df3)))


def load_edukg(data_home=None, time_mode=None):
    data_path = data_home + '/edukg'
    ent_df = pd.read_csv(data_path + '/entity2id.txt', sep='\t',
                         header=None, names=['ent', 'id'])
    # print('ent_df.dtypes: ',ent_df.dtypes)
    rel_df = pd.read_csv(data_path + '/relation2id.txt', sep='\t',
                         header=None, names=['rel', 'id'])
    # print('rel_df.dtypes: ',rel_df.dtypes)
    ent_df = dict(zip(ent_df['ent'], ent_df['id']))
    rel_df = dict(zip(rel_df['rel'], rel_df['id']))
    
    df1 = read_csv(data_path + '/train.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to', 'start_time', 'end_time'])
    df1['end_time'] = df1['start_time']
    # print(df1)
    # print(df1.isna().any())
    # print('df1.dtypes: ',df1.dtypes)
    df2 = read_csv(data_path + '/val.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to', 'start_time', 'end_time'])
    df2['end_time'] = df2['start_time']
    # print('df2.dtypes: ',df2.dtypes)
    df3 = read_csv(data_path + '/test.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to', 'start_time', 'end_time'])
    df3['end_time'] = df3['start_time']
    # print('df3.dtypes: ',df3.dtypes)
    df = concat([df1, df2, df3])
    kg = TemporalKnowledgeGraph(df, time_mode = time_mode, ent2ix=ent_df, rel2ix=rel_df)

    return kg.split_kg(sizes=(len(df1), len(df2), len(df3)))

def load_gdelt(data_home=None, time_mode=None):
    data_path = data_home + '/gdelt'

    df1 = read_csv(data_path + '/train.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to', 'start_time'])
    df1['end_time'] = '1111-11-11'
    df2 = read_csv(data_path + '/valid.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to', 'start_time'])
    df2['end_time'] = '1111-11-11'
    df3 = read_csv(data_path + '/test.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to', 'start_time'])
    df3['end_time'] = '1111-11-11'
    df = concat([df1, df2, df3])
    kg = TemporalKnowledgeGraph(df, time_mode = time_mode)

    return kg.split_kg(sizes=(len(df1), len(df2), len(df3)))

def load_icews15(data_home=None, time_mode=None):
    data_path = data_home + '/icews05-15'

    df1 = read_csv(data_path + '/train.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to', 'start_time', 'end_time'])
    df2 = read_csv(data_path + '/valid.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to', 'start_time', 'end_time'])
    df3 = read_csv(data_path + '/test.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to', 'start_time', 'end_time'])
    df = concat([df1, df2, df3])
    kg = TemporalKnowledgeGraph(df, time_mode = time_mode)

    return kg.split_kg(sizes=(len(df1), len(df2), len(df3)))

def load_yago11k(data_home=None, time_mode=None):
    data_path = data_home + '/YAGO11k'

    df1 = read_csv(data_path + '/train.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to', 'start_time', 'end_time'])
    df2 = read_csv(data_path + '/valid.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to', 'start_time', 'end_time'])
    df3 = read_csv(data_path + '/test.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to', 'start_time', 'end_time'])
    df = concat([df1, df2, df3])
    df['start_time'] = df['start_time'].str.replace('##','01')
    df['end_time'] = df['end_time'].str.replace('##','01')
    kg = TemporalKnowledgeGraph(df, time_mode = time_mode)

    return kg.split_kg(sizes=(len(df1), len(df2), len(df3)))

def load_wikidata12k(data_home=None, time_mode=None):
    data_path = data_home + '/WIKIDATA12k'

    df1 = read_csv(data_path + '/train.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to', 'start_time', 'end_time'])
    df2 = read_csv(data_path + '/valid.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to', 'start_time', 'end_time'])
    df3 = read_csv(data_path + '/test.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to', 'start_time', 'end_time'])
    df = concat([df1, df2, df3])
    df['start_time'] = df['start_time'].str.replace('##','01')
    df['end_time'] = df['end_time'].str.replace('##','01')
    kg = TemporalKnowledgeGraph(df, time_mode = time_mode)

    return kg.split_kg(sizes=(len(df1), len(df2), len(df3)))


def get_data(data_name, time_mode, data_home='data'):
    if data_name == 'icews14':
        return load_icews14(data_home=data_home, time_mode = time_mode)
    elif data_name == 'edukg':
        return load_edukg(data_home=data_home, time_mode=time_mode)
    elif data_name == 'gdelt':
        return load_gdelt(data_home=data_home, time_mode = time_mode)
    elif data_name == 'icews15':
        return load_icews15(data_home=data_home, time_mode = time_mode)
    elif data_name == 'yago11k':
        return load_yago11k(data_home=data_home, time_mode = time_mode)
    elif data_name == 'wikidata12k':
        return load_wikidata12k(data_home=data_home, time_mode = time_mode)
    else:
        datas = ['icews14']
        
        print('Choose One of the Following Datasets: ',datas)
