
import json
import pandas as pd
import torch
import datetime
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

tem_dict = {
    '0y': 0, '1y': 1, '2y': 2, '3y': 3, '4y': 4, '5y': 5, '6y': 6, '7y': 7, '8y': 8, '9y': 9,
    '0m': 10, '1m': 11, '2m': 12, '3m': 13, '4m': 14, '5m': 15, '6m': 16, '7m': 17, '8m': 18, '9m': 19,
    '0d': 20, '1d': 21, '2d': 22, '3d': 23, '4d': 24, '5d': 25, '6d': 26, '7d': 27, '8d': 28, '9d': 29
}

class Params():
    def __init__(self):
        pass


def get_expriment_tests(results):
    res = []
    for exp in results:
        train_params = exp['train_params']
        test_results = exp['test_results']
        train_params.update(test_results)
        res.append(train_params)
    return pd.DataFrame(res)

def read_json_lines(file_name):
    lines = []
    with open(file_name) as file_in:
        for line in file_in:
            lines.append(json.loads(line))
    return lines

def write_json_lines(file_name,dict_data):
    json_string = json.dumps(dict_data)
    with open(file_name, 'a') as f:
        f.write(json_string+"\n")

def experiment_exists(all_exp_data_df, exp_data_dict, important_fields = None):
    if important_fields is None:
        search_keys = exp_data_dict
    else:
        search_keys = important_fields
    for k in search_keys:
        all_exp_data_df = all_exp_data_df[all_exp_data_df[k]==exp_data_dict[k]]
        if all_exp_data_df.shape[0]==0:
            return False
    return True
def check_experiment(exp_file_name):
    res = read_json_lines(exp_file_name)
    res = get_expriment_tests(res)
    return res
    # experiment_exists(all_exp_data_df, exp_data_dict)
def transform_time_V2(years, months, days):
    all_data = []
    for year, month, day in zip(years, months, days):
        tem_id_list = []
        for j in range(len(year)):
            token = year[j:j+1]+'y'
            tem_id_list.append(tem_dict[token])
        # print(tem_id_list)
        # exit()

        for j in range(1):
            # print(month[1])
            # exit()
            token1 = month[0]+'m'
            tem_id_list.append(tem_dict[token1])
            token2 = month[0]+'m'
            tem_id_list.append(tem_dict[token2])


        for j in range(len(day)):
            token = day[j:j+1]+'d'
            tem_id_list.append(tem_dict[token])
            
        all_data.append(torch.tensor(tem_id_list))
    return all_data

def transform_time(raw_time):
    year, month, day = raw_time.split("-")
    tem_id_list = []
    for j in range(len(year)):
        token = year[j:j+1]+'y'
        tem_id_list.append(tem_dict[token])
    # print(tem_id_list)
    # exit()

    for j in range(1):
        # print(month[1])
        # exit()
        token1 = month[0]+'m'
        tem_id_list.append(tem_dict[token1])
        token2 = month[0]+'m'
        tem_id_list.append(tem_dict[token2])


    for j in range(len(day)):
        token = day[j:j+1]+'d'
        tem_id_list.append(tem_dict[token])
    return tem_id_list

def transform_time_v3(raw_time):
    date = list(map(float, raw_time.split("-")))
    return date

def transform_time_v4(raw_time):
    year, month, day = raw_time.split("-")
    year, month, day = int(year), int(month), int(day)
    return month + day

def transform_time_v5(raw_time):
    year, month, day = raw_time.split("-")
    return [int(year), int(month), int(day)]

def get_temporal_dictionaries(df, mode='simple'):

    tmp = list(set(df['start_time'].unique()).union(set(df['end_time'].unique())))
    if mode == 'simple_time':
        # return {timee: i for i, timee in enumerate(sorted(tmp))}
        idx = {timee: i for i, timee in enumerate(sorted([datetime.strptime(dt, "%Y-%m-%d") for dt in tmp]))}
        time_trans = torch.tensor(list(idx.values()))
        return idx, time_trans
    elif mode == 'simple':
        # return {timee: i for i, timee in enumerate(sorted(tmp))}
        idx = {timee: i for i, timee in enumerate(sorted([dt for dt in tmp]))}
        time_trans = torch.tensor(list(idx.values()))
        return idx, time_trans
    elif mode == 'seq':
        idx = {timee: i for i, timee in enumerate(sorted(tmp))}
        time_trans = torch.vstack([torch.tensor(transform_time(timee)) for timee in sorted(tmp)])
        return idx, time_trans
        # return {timee: torch.tensor(transform_time(timee)) for i, timee in enumerate(sorted([datetime.strptime(dt, "%Y-%m-%d") for dt in tmp]))}
    elif mode == 'ymd':
        idx = {timee: transform_time_v4(timee) for i, timee in enumerate(sorted(tmp))}
        time_trans = torch.vstack([transform_time_v4(timee) for timee in sorted(tmp)])
        return idx, time_trans
    elif mode == 'ymd_':
        # idx = {timee: torch.tensor(transform_time_v5(timee)) for i, timee in enumerate(sorted(tmp))}
        # time_trans = torch.vstack([torch.tensor(transform_time_v5(timee)) for timee in sorted(tmp)])
        # return idx, time_trans
        idx = {timee: i for i, timee in enumerate(sorted(tmp))}
        time_trans = torch.vstack([torch.tensor(transform_time_v5(timee)) for timee in sorted(tmp)])
        return idx, time_trans

def cconv(a, b):
    return torch.fft.ifft(torch.fft.fft(a) * torch.fft.fft(b)).real


def ccorr(a, b):
    return torch.fft.ifft(torch.conj(torch.fft.fft(a)) * torch.fft.fft(b)).real
