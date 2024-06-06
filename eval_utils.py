
from torch import tensor


def get_true_targets(dictionary, key1, key2, key3, key4, true_idx, i):
    try:
        if key4 is not None:
            true_targets = dictionary[key1[i].item(), key2[i].item(), key3[i].item(), key4[i].item()].copy()
        else:
            true_targets = dictionary[key1[i].item(), key2[i].item(), key3[i].item()].copy()
            
        if true_idx is not None:
            true_targets.remove(true_idx[i].item())
            if len(true_targets) > 0:
                return tensor(list(true_targets)).long()
            else:
                return None
        else:
            return tensor(list(true_targets)).long()
    except KeyError:
        return None


# filter_scores(scores, self.kg.dict_of_tails, h_idx, r_idx, t_idx, start_time, end_time)
def filter_scores(scores, dictionary, key1, key2, key3, key4, true_idx):
    # filter out the true negative samples by assigning - inf score.
    b_size = scores.shape[0]
    filt_scores = scores.clone()

    for i in range(b_size):
        true_targets = get_true_targets(dictionary, key1, key2, key3, key4, true_idx, i)
        if true_targets is None:
            continue
        filt_scores[i][true_targets] = - float('Inf')

    return filt_scores


def get_rank(data, true, low_values=False):
    true_data = data.gather(1, true.long().view(-1, 1))

    if low_values:
        return (data <= true_data).sum(dim=1)
    else:
        return (data >= true_data).sum(dim=1)