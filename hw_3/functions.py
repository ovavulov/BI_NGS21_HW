import os

def get_prev_path(path, n_steps):
    res_path = path
    for _ in range(n_steps):
        res_path = os.path.split(res_path)[0]
    return res_path