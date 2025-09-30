import torch.nn as nn
import torch
import shutil
import torch.optim as optim
import math
from pytorch_msssim import ms_ssim
import torch.nn.functional as F
import numpy as np
import wandb



class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


def sec_to_hours(seconds):
    a=str(seconds//3600)
    b=str((seconds%3600)//60)
    c=str((seconds%3600)%60)
    d=["{} hours {} mins {} seconds".format(a, b, c)]
    print(d[0])




from datetime import datetime
from os.path import join         
def create_savepath(args,epoch,epoch_enc):
    now = datetime.now()
    date_time = now.strftime("%m%d")
    c = join(date_time,"_lambda_",str(args.lmbda),"_epoch_",str(epoch),"_epochenc_",str(epoch_enc)).replace("/","_")

    
    c_best = join(c,"best").replace("/","_")
    c = join(c,args.suffix).replace("/","_")
    c_best = join(c_best,args.suffix).replace("/","_")
    
    
    path =  "/data/"#args.filename
    savepath = join(path,c)
    savepath_best = join(path,c_best)
    
    print("savepath: ",savepath)
    print("savepath best: ",savepath_best)
    return savepath, savepath_best








def set_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)



def configure_optimizers(net, args, baseline = False):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {n for n, p in net.named_parameters() if not n.endswith(".quantiles") and (p.requires_grad or "sos" in n)} 



    params_dict = dict(net.named_parameters())

    if baseline:

        aux_parameters = {
            n
            for n, p in net.named_parameters()
            if n.endswith(".quantiles") and p.requires_grad
        }

        # Make sure we don't have an intersection of parameters
        params_dict = dict(net.named_parameters())
        inter_params = parameters & aux_parameters
        union_params = parameters | aux_parameters

        assert len(inter_params) == 0
        #assert len(union_params) - len(params_dict.keys()) == 0



        aux_optimizer = optim.Adam((params_dict[n] for n in sorted(aux_parameters)),lr=args.aux_learning_rate,)
    else:
        aux_optimizer = None

    
    if args.sgd == "adam":
        optimizer = optim.Adam((params_dict[n] for n in sorted(parameters)),lr=args.learning_rate,)
    else: 
        optimizer = optim.SGD((params_dict[n] for n in sorted(parameters)),lr=args.learning_rate)

    return optimizer, aux_optimizer

def save_checkpoint_our(state, is_best, filename,filename_best):
    torch.save(state, filename)
    wandb.save(filename)
    if is_best:
        shutil.copyfile(filename, filename_best)
        wandb.save(filename_best)



def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[:-8]+"_best"+filename[-8:])



##########################################################################################
#########################################################################################
##########################################################################################

