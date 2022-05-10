import os

def create_exp_folder(path):
    _f = next(os.walk(path))[1]
    _last_idx = max( [int(f.split('_')[1]) for f in _f] ) + 1
    _folder_name = "exp_%i"%(_last_idx)
    return _folder_name

# implement mixup

# implement cutmix