import torch
import numpy as np
import os
from Config.options_1 import opt
from torch.autograd import Variable

def getLatestCheckpointName():
    if os.path.exists(opt.checkpoints_dir):
        file_names = os.listdir(opt.checkpoints_dir)
        names_ext = [os.path.splitext(x) for x in file_names]
        checkpoint_names_G = []

        for i in range(len(names_ext)):
            if names_ext[i][1] == '.pt':
                module = str(names_ext[i][0]).split('_')
                if module[0] == 'DICAM' and module[1].isdigit():
                    checkpoint_names_G.append(int(module[1]))

        if len(checkpoint_names_G) == 0:
            return None

        g_index = max(checkpoint_names_G)
        ckp_g = None

        for fname in file_names:
            parts = fname.replace('.pt', '').split('_')
            if parts[0] == 'DICAM' and int(parts[1]) == g_index:
                ckp_g = fname
                break

        return ckp_g
