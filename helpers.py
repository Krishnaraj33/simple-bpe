from typing import List

def get_stats(ids: List[int], count_dict: dict[tuple[int,int],int] | None = None) -> dict[tuple[int,int],int]:
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    if count_dict is None:
        count_dict = {}
    
    for pair in zip(ids,ids[1:]):
        count_dict[pair] = count_dict.get(pair,0) + 1
    return count_dict


def merge_pairs(ids: List[int], pair: tuple[int,int], idx: int) ->  List[int]:
    """
    In the list of integers (ids), replace all consecutive occurrences 
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    new_ids = []
    i = 0
    
    while i < len(ids):
        # For no bounds error 
        if i < (len(ids)-1) and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

if __name__ == "__main__":
    l = [1, 2, 3, 1, 2]
    res = get_stats(l)
    most_freq_pair = max(res,key=res.get)
    