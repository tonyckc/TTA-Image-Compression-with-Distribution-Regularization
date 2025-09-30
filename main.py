import os
import torch
import torch.nn as nn
from compressai.zoo import load_state_dict
from dataset import KodakDataset
import torch.nn.functional as F
from net import ScaleHyperpriorSGA,Cheng2020AnchorSGA,Cheng2020Anchor
import argparse
import math
from tqdm import tqdm



parser = argparse.ArgumentParser(prog='main')

parser.add_argument('-q', '--quality', default=0, required=False, help='quality = {0,...,7}')
parser.add_argument('-mr', '--model_root', default='your_path/image_coding/',required=False, help='root of model tar')
parser.add_argument('-dr', '--data_root', default='your_path/image_coding/kodak_images',required=False, help='root of kodak_images dataset')
parser.add_argument('-mn', '--model_name', default='cheng',required=False, help='cheng or hyper')
parser.add_argument('-rn', '--run_name', default='-',required=False, )
parser.add_argument('-gn', '--gpu_id', required=False, default=1,)
parser.add_argument('-type', '--type', type=str,required=False, default='automatically be set later')
parser.add_argument('-latent', '--latent', type=str,required=False, default='ours',help='blr or hlr or ours')
parser.add_argument('-ti', '--total_iter', default=2000,type=int,required=False, )
parser.add_argument('-lr', '--lr', default=1e-3,type=float,required=False, )
parser.add_argument('-cop', '--cor_para', type=float, default=1.0,required=False)
parser.add_argument('--cor', action='store_true', default=False, help="if we introduce the correlation bewtween two networks (True for ours)")
parser.add_argument('--sga', action='store_true', default=False, help="Stochastic Gumbel Annealing = SGA, in Yang'20 paper of hlr")
parser.add_argument('--seed', default=0,type=int,required=False)
parser.add_argument('--dropN', default=1,type=int,required=False, help='number of dropout layers')
parser.add_argument('--tau', default=700,type=int,required=False,help="hyperparameter of SGA, fixed")
import wandb
import random
import numpy as np


def psnr(mse):
    return 10*torch.log10((255**2) / mse)

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    print(size)
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

def set_seed(seed):
    '''
    seed  setting
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    args = parser.parse_args()

    set_seed(args.seed)
    model_name = args.model_name
    if model_name not in ['cheng','hyper']:
        print('No model!!!')
        exit()
    dataset_name = args.data_root
    model_root = args.model_root + model_name
    ################################

    if args.latent == 'blr':
        args.type = 'y'
        args.sga = False
        args.cor = False

    elif args.latent == 'hlr':
        args.type = 'y_and_z'
        args.sga = True
        args.cor = False
    elif args.latent == 'ours':
        args.type = 'y_and_z'
        args.sga = True
        args.cor = True
    else:
        print('No model!!')
        exit()
    print(args)
    dataset_name = dataset_name.split('/')[-1]
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu_id)
    wandb.init(project="{}-train-model-{}-time-test".format(model_name,dataset_name), name='{}-{}-cor-{}-sga-{}-quality-{}-tau-{}-dropN-{}-beta-{}-step-{}'.format(args.latent,args.type,args.cor,args.sga,args.quality,args.tau,args.dropN,args.cor_para,args.total_iter))
    wandb.run.log_code("/home/ckc/pytorch-improving-inference-for-neural-image-compression-master/", include_fn=lambda path: path.endswith(".py"))
    if model_name == 'cheng':
        model_names = [ "cheng2020-anchor-1-dad2ebff.pth.tar",
                        "cheng2020-anchor-2-a29008eb.pth.tar",
            "cheng2020-anchor-3-e49be189.pth.tar",
            "cheng2020-anchor-4-98b0b468.pth.tar",
            "cheng2020-anchor-5-23852949.pth.tar",
            "cheng2020-anchor-6-4c052b1a.pth.tar"]
    else:
        model_names = ["bmshj2018-hyperprior-1-7eb97409.pth.tar",
                       "bmshj2018-hyperprior-2-93677231.pth.tar",
                       "bmshj2018-hyperprior-3-6d87be32.pth.tar",
                       "bmshj2018-hyperprior-4-de1b779c.pth.tar",
                       "bmshj2018-hyperprior-5-f8b614e1.pth.tar",
                       "bmshj2018-hyperprior-6-1ab9c41e.pth.tar",
                       "bmshj2018-hyperprior-7-3804dcbd.pth.tar",
                       "bmshj2018-hyperprior-8-a583f0cf.pth.tar"]
    if model_name == 'cheng':
        lams = [0.0018, 0.0035, 0.0067, 0.013, 0.025, 0.048]
    else:
        lams = [0.0018,0.0035,0.0067,0.0130,0.0250,0.0483,0.0932,0.1800]
    q = int(args.quality)
    if model_name == 'cheng':
        Ns, Ms = [128, 128, 128, 192, 192, 192], [128, 128, 128, 192, 192, 192]
    else:
        Ns, Ms = [128,128,128,128,128,192,192,192], [192,192,192,192,192,320,320,320]
    N, M = Ns[q], Ms[q]

    model_path = os.path.join(model_root, model_names[q])
    if model_name == 'cheng':
        model = Cheng2020AnchorSGA(N=N)
    else:
        model = ScaleHyperpriorSGA(N, M)


    model_dict = load_state_dict(torch.load(model_path))


    model.load_state_dict(model_dict)




    model = model.cuda()

    dataset = KodakDataset(kodak_root=args.data_root)
    dataloader = torch.utils.data.DataLoader(dataset)

    model.eval()
    bpp_init_avg, mse_init_avg, psnr_init_avg, rd_init_avg = 0, 0, 0, 0
    bpp_post_avg, mse_post_avg, psnr_post_avg, rd_post_avg = 0, 0, 0, 0





    num_psnr_neg = 0
    time_total = 0
    for idx, img in enumerate(dataloader):


        img = img.cuda()
        img_h, img_w = img.shape[2], img.shape[3]

        img = img.permute(0, 3, 1, 2) if img.shape[1] != 3 else img  # check channel is 3

        img_pixnum = img_h * img_w

        # first round
        with torch.no_grad():
            ret_dict = model.forward(img, mode='init')
        bpp_init = torch.sum(-torch.log2(ret_dict["likelihoods"]["y"])) / (img_pixnum) +\
                   torch.sum(-torch.log2(ret_dict["likelihoods"]["z"])) / (img_pixnum)
        bpp_mse = - torch.sum(ret_dict["likelihoods"]["bpp"])/(torch.log(torch.tensor(2.)).cuda()*img_pixnum)
        mse_true = torch.sum(ret_dict["mse"])/img_pixnum
        wandb.log({"mse-true":mse_true,"mse":bpp_mse, 'bpp_y_init': torch.sum(-torch.log2(ret_dict["likelihoods"]["y"])) / (img_pixnum),'bpp_z_init':torch.sum(-torch.log2(ret_dict["likelihoods"]["z"])) / (img_pixnum) })



        # To prevent the resolution change after the reconstruction
        if img.shape == ret_dict["x_hat"].shape:
            mse_init = F.mse_loss(img, ret_dict["x_hat"]) * (255 ** 2)
        else:
            ret_dict["x_hat"] = ret_dict["x_hat"][:,:,:min(ret_dict["x_hat"].shape[2],img_h),:min(ret_dict["x_hat"].shape[3],img_w)]
            mse_init = F.mse_loss(img, ret_dict["x_hat"]) * (255 ** 2)

        rd_init = bpp_init + lams[q] * mse_init
        psnr_init = psnr(mse_init)
        if psnr_init < 0:
            num_psnr_neg += 1
            continue

        bpp_init_avg += bpp_init
        mse_init_avg += mse_init
        psnr_init_avg += psnr_init
        rd_init_avg += rd_init


        # start to test-time adaptation
        # make latent code learnable parameters
        y, z = nn.parameter.Parameter(ret_dict["y"]), nn.parameter.Parameter(ret_dict["z"])
        # only optimize y
        lr = args.lr
        if args.type == 'y':
            opt = torch.optim.Adam([y], lr=lr)
        else:
            # sga optimize y and z
            opt = torch.optim.Adam([y] + [z], lr=lr)

        tau_decay_it = args.tau
        # keep fixed for SGA
        tau_decay_factor = 0.001

        tot_it = args.total_iter
        import time

        start_time = time.time()
        for it in tqdm(range(tot_it)):
            decaying_iter: int = it - tau_decay_it
            # if decaying_iter < 0, tau should be 0.5.
            tau: float = min(0.5, 0.5 * np.exp(-tau_decay_factor * decaying_iter))

            opt.zero_grad()
            if args.sga == True:
                ret_dict = model(img, "sga", y, z, it, tot_it,tau, dropN=args.dropN)
            else:
                ret_dict = model(img, "noise", y, z, it, tot_it)

            if args.type == 'y':
                bpp = torch.sum(-torch.log2(ret_dict["likelihoods"]["y"])) / (img_pixnum)+ \
                      torch.sum(-torch.log2(ret_dict["likelihoods"]["z"])) / (img_pixnum)
            else:
                bpp = torch.sum(-torch.log2(ret_dict["likelihoods"]["y"])) / (img_pixnum) + \
                      torch.sum(-torch.log2(ret_dict["likelihoods"]["z"])) / (img_pixnum)


            if img.shape == ret_dict["x_hat"].shape:
                mse = F.mse_loss(img, ret_dict["x_hat"]) * (255 ** 2)
            else:
                ret_dict["x_hat"] = ret_dict["x_hat"][:,:,:min(ret_dict["x_hat"].shape[2],img_h),:min(ret_dict["x_hat"].shape[3],img_w)]
                mse = F.mse_loss(img, ret_dict["x_hat"]) * (255 ** 2)

            bpp_mse = - torch.sum(ret_dict["likelihoods"]["bpp"])/(torch.log(torch.tensor(2.)).cuda()*img_pixnum)

            mse_true = 0.5 * torch.sum(ret_dict["mse"]) / img_pixnum


            if args.cor:
                # bpp_mse is the correlation loss
                rdcost = bpp + lams[q] * mse + bpp_mse*args.cor_para
                rdcost_validate = bpp + lams[q] * mse
            else:
                rdcost = bpp + lams[q] * mse
                rdcost_validate = bpp + lams[q] * mse


            wandb.log({"mse-true":mse_true,"loss": rdcost,"mse":bpp_mse,'rd_validate':rdcost_validate,"y_entropy_running":torch.sum(-torch.log2(ret_dict["likelihoods"]["y"])) / (img_pixnum),
                       "z_entropy_running": torch.sum(-torch.log2(ret_dict["likelihoods"]["z"])) / (img_pixnum)})
            rdcost.backward()
            opt.step()

        end_time = time.time()




        with torch.no_grad():
            ret_dict = model(img, "round", y, z)

        bpp_post = torch.sum(-torch.log2(ret_dict["likelihoods"]["y"])) / (img_pixnum) +\
                   torch.sum(-torch.log2(ret_dict["likelihoods"]["z"])) / (img_pixnum)
        wandb.log({'bpp_y_post': torch.sum(-torch.log2(ret_dict["likelihoods"]["y"])) / (img_pixnum),
                   'bpp_z_post': torch.sum(-torch.log2(ret_dict["likelihoods"]["z"])) / (img_pixnum)})

        if img.shape == ret_dict["x_hat"].shape:
            mse_post = F.mse_loss(img, ret_dict["x_hat"]) * (255 ** 2)
        else:
            ret_dict["x_hat"] = ret_dict["x_hat"][:,:,:min(ret_dict["x_hat"].shape[2],img_h),:min(ret_dict["x_hat"].shape[3],img_w)]
            mse_post = F.mse_loss(img, ret_dict["x_hat"]) * (255 ** 2)
        rd_post = bpp_post + lams[q] * mse_post
        psnr_post = psnr(mse_post)
        bpp_post_avg += bpp_post
        mse_post_avg += mse_post
        psnr_post_avg += psnr_post
        rd_post_avg += rd_post
        time_total += end_time - start_time




        print("img: {0}, psnr init: {1:.4f}, bpp init: {2:.4f}, rd init: {3:.4f}, psnr post: {4:.4f}, bpp post: {5:.4f}, rd post: {6:.4f}"\
              .format(idx, psnr_init, bpp_init, rd_init, psnr_post, bpp_post, rd_post))
        wandb.log({'psnr_init':psnr_init, 'bpp_init': bpp_init, 'rd_init': rd_init, 'psnr_post': psnr_post, 'bpp_post': bpp_post, 'rd_post': rd_post})



    bpp_init_avg /= (idx + 1- num_psnr_neg)
    mse_init_avg /= (idx + 1- num_psnr_neg)
    psnr_init_avg /= (idx + 1- num_psnr_neg)
    rd_init_avg /= (idx + 1- num_psnr_neg)

    bpp_post_avg /= (idx + 1 - num_psnr_neg)
    mse_post_avg /= (idx + 1- num_psnr_neg)
    psnr_post_avg /= (idx + 1- num_psnr_neg)
    rd_post_avg /= (idx + 1- num_psnr_neg)

    time_average = time_total/(idx + 1- num_psnr_neg)
    print("mean, psnr init: {0:.4f}, bpp init: {1:.4f}, rd init: {2:.4f}, psnr post: {3:.4f}, bpp post: {4:.4f}, rd post: {5:.4f}"\
          .format(psnr_init_avg, bpp_init_avg, rd_init_avg, psnr_post_avg, bpp_post_avg, rd_post_avg))
    wandb.log(
        {'psnr_init_avg': psnr_init_avg, 'bpp_init_avg': bpp_init_avg, 'rd_init_avg': rd_init_avg, 'psnr_post_avg': psnr_post_avg,
         'bpp_post_avg': bpp_post_avg,
         'rd_post_avg': rd_post_avg,'average_time':time_average})

if __name__ == "__main__":
    main()
