import torch 

import wandb
from compress.utils.help_function import compute_msssim, compute_psnr
import torch.nn.functional as F 
from PIL import Image
from torchvision import transforms
from pytorch_msssim import ms_ssim
import math
from compress.datasets import AdapterDataset

def read_image(filepath, adapt =False):
    #assert filepath.is_file()
    img = Image.open(filepath)
    
    if adapt:
        i =  img.size
        i = 256, 256#i[0]//2, i[1]//2
        img = img.resize(i)
    img = img.convert("RGB")
    return transforms.ToTensor()(img) 


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(model, criterion, train_dataloader, optimizer, epoch, clip_max_norm ,counter):
    model.train()
    device = next(model.parameters()).device



    
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    y_bpp = AverageMeter()
    z_bpp = AverageMeter()

    adapter_loss = AverageMeter()




    for i, d  in enumerate(train_dataloader):


        counter += 1
        d = d.to(device)
        optimizer.zero_grad()
        #if aux_optimizer is not None:
        #    aux_optimizer.zero_grad()


        out_net = model(d)
        out_criterion = criterion(out_net, d)

        out_criterion["loss"].backward()
        optimizer.step()

        loss.update(out_criterion["loss"].clone().detach())
        mse_loss.update(out_criterion["mse_loss"].clone().detach())
        bpp_loss.update(out_criterion["bpp_loss"].clone().detach())


        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)



        #if aux_optimizer is not None:
        #    aux_loss = model.aux_loss()
        #    aux_loss.backward()
        #    aux_optimizer.step()

        if i % 100 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                
                f'\tMSE loss: {out_criterion["mse_loss"].item() * 255 ** 2 / 3:.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'

            )


        wand_dict = {
            "train_batch": counter,
            #"train_batch/delta": model.gaussian_conditional.sos.delta.data.item(),
            "train_batch/losses_batch": out_criterion["loss"].clone().detach().item(),
            "train_batch/bpp_batch": out_criterion["bpp_loss"].clone().detach().item(),
            "train_batch/mse":out_criterion["mse_loss"].clone().detach().item(),
        }
        wandb.log(wand_dict)


        if "z_bpp" in list(out_criterion.keys()):

            wand_dict = {
                "train_batch": counter,
                "train_batch/factorized_bpp": out_criterion["z_bpp"].clone().detach().item(),
                "train_batch/gaussian_bpp": out_criterion["y_bpp"].clone().detach().item()

            }
            wandb.log(wand_dict)  
            y_bpp.update(out_criterion["y_bpp"].clone().detach())
            z_bpp.update(out_criterion["z_bpp"].clone().detach())    

    log_dict = {
        "train":epoch,
        "train/loss": loss.avg,
        "train/bpp": bpp_loss.avg,
        "train/mse": mse_loss.avg,
        "train/adapter_loss":adapter_loss.avg
        }
        
    wandb.log(log_dict)
    return counter




def test_epoch(epoch, test_dataloader, model, criterion,  valid):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()


    psnr = AverageMeter()
    ssim = AverageMeter()


    quantization_error = AverageMeter()

    adapter_error = AverageMeter()
    with torch.no_grad():
        for d in test_dataloader:

            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])




            psnr.update(compute_psnr(d, out_net["x_hat"]))
            ssim.update(compute_msssim(d, out_net["x_hat"]))


    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg * 255 ** 2 / 3:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
    )


    if valid is False:
        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMSE loss: {mse_loss.avg * 255 ** 2 / 3:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
        )
        log_dict = {
        "test":epoch,
        "test/loss": loss.avg,
        "test/bpp":bpp_loss.avg,
        "test/mse": mse_loss.avg,
        "test/psnr":psnr.avg,
        "test/ssim":ssim.avg,
        "test/quantization_error":quantization_error.avg,
        "test/adapter_error": adapter_error.avg
        }
    else:

        print(
            f"valid epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMSE loss: {mse_loss.avg * 255 ** 2 / 3:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
        )
        log_dict = {
        "valid":epoch,
        "valid/loss": loss.avg,
        "valid/bpp":bpp_loss.avg,
        "valid/mse": mse_loss.avg,
        "valid/psnr":psnr.avg,
        "valid/ssim":ssim.avg,
        "valid/quantization_error":quantization_error.avg,
        "valid/adapter_error": adapter_error.avg
        }       

    wandb.log(log_dict)
   

    return loss.avg



from compressai.ops import compute_padding
import os

def compress_with_ac(model,  filelist, device, epoch, name = "",loop = True,  writing = None, save_images = None ):
    #model.update(None, device)
    print("ho finito l'update")
    bpp_loss = AverageMeter()
    psnr_val = AverageMeter()
    mssim_val = AverageMeter()

    
    with torch.no_grad():
        for i,d in enumerate(filelist):
            print("------------------------------ IMAGE ----> ",d) 

            x = read_image(d, adapt = False).to(device)


            nome_immagine = d.split("/")[-1].split(".")[0]
            x = x.unsqueeze(0) 
            h, w = x.size(2), x.size(3)
            pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
            x_padded = F.pad(x, pad, mode="constant", value=0)


            data =  model.compress(x_padded)
            #print("end compress")
            out_dec = model.decompress(data["strings"], data["shape"])
            #print("end decompress")

            out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)

            out_dec["x_hat"].clamp_(0.,1.)     

            if save_images is not None:


                image = transforms.ToPILImage()(out_dec['x_hat'].squeeze())
                nome_salv = os.path.join(save_images, nome_immagine + "base2.png")
                image.save(nome_salv)



            # parte la prova, devo controllare che y_hat sia sempre uguale!!!!! !!!!
            #out = model.forward(x)
            #y_hat_t = out["y_hat"].ravel()
            #y_hat_comp = out_dec["y_hat"].ravel()
            
            #for i in range(10):
            #    print("----> ", y_hat_t[i]," ",y_hat_comp[i],"  ",out_dec["x_hat"].shape)





            psnr_im = compute_psnr(x, out_dec["x_hat"])
            ms_ssim_im = compute_msssim(x, out_dec["x_hat"])
            ms_ssim_im = -10*math.log10(1 - ms_ssim_im )
            psnr_val.update(psnr_im)
            mssim_val.update(ms_ssim_im)
            
            size = out_dec['x_hat'].size()
            num_pixels = size[0] * size[2] * size[3]
            bpp = sum(len(s[0]) for s in data["strings"]) * 8.0 / num_pixels#sum(len(s[0]) for s in data["strings"]) * 8.0 / num_pixels

            print("la lunghezza del bpp è -----> ",sum(len(s[0]) for s in data["strings"]) * 8.0)
            print("num_pixels-----> ",num_pixels)
            print("bpp totale ",bpp)
            print("bpp !augmented-----> ",(sum(len(s[0]) for s in data["strings"]) * 8.0 + 32*3) / num_pixels)

            
            bpp_loss.update(bpp)

            if writing is not None:
                fls = writing + name + ".txt"
                f=open(fls , "a+")
                f.write("SEQUENCE "  +   nome_immagine + " BITS " +  str(bpp) + " PSNR " +  str(psnr_im)  + " MSSIM " +  str(ms_ssim_im) + "\n")
                f.close()  
                
            #print("image: ",d,": ",bpp," ",compute_psnr(x, out_dec["x_hat"]))




                    

    if loop:
        log_dict = {
             name +   "compress":epoch,
             name +   "compress/bpp_with_ac": bpp_loss.avg,
             name +   "compress/psnr_with_ac": psnr_val.avg,
             name +   "compress/mssim_with_ac":mssim_val.avg
        }
        
        wandb.log(log_dict)
        print("RESULTS OF COMPRESSION IS : ",bpp_loss.avg," ",psnr_val.avg," ",mssim_val.avg)
    else:
        print("RESULTS OF COMPRESSION IS : ",bpp_loss.avg," ",psnr_val.avg," ",mssim_val.avg)
    
    if writing is not None:

        fls = writing + name + ".txt"
        f=open(fls , "a+")
        f.write("SEQUENCE "  +   "AVG " + "BITS " +  str(bpp_loss.avg) + " YPSNR " +  str(psnr_val.avg)  + " YMSSIM " +  str(mssim_val.avg) + "\n")



    return psnr_val.avg, bpp_loss.avg



def bpp_calculation(out_net, out_enc, fact = False):
        size = out_net['x_hat'].size() 
        num_pixels = size[0] * size[2] * size[3]
        if fact is False:
            bpp_1 = (len(out_enc[0]) * 8.0 ) / num_pixels
            bpp_2 =  (len(out_enc[1]) * 8.0 ) / num_pixels
            return bpp_1 + bpp_2, bpp_1, bpp_2
        else:
            bpp = (len(out_enc[0]) * 8.0 ) / num_pixels 
            return bpp, bpp, bpp


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:


    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())

def compute_metrics( org, rec, max_val: int = 255):
    metrics =  {}
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    metrics["psnr"] = psnr(org, rec).item()
    print("la metrica psnr è questa ", metrics["psnr"])
    metrics["ms-ssim"] = ms_ssim(org, rec, data_range=max_val).item()
    print("la metrica psnr è questa ", metrics["psnr"],"   ",metrics["ms-ssim"])
    return metrics


def evaluate_base_model(model, args,device, considered_classes = ["sketch","clipart","comic","kodak","infographics"]):
    """
    Valuto la bontà del modello base, di modo da poter vedere se miglioraiamo qualcosadddd
    """
    res = {}
    considered_classes = args.considered_classes if considered_classes is None else considered_classes

    model.to(device)
    model.update()
    test_transforms = transforms.Compose([transforms.ToTensor()])

    for i,cl in enumerate(considered_classes):
        print("**********************************   ",cl," *******************************************")
        cons_classes = args.considered_classes + [cl] if cl not in args.considered_classes else considered_classes
        txt_file = "_" + cl + "_.txt"
        cl_class = AdapterDataset(root = args.root + "/test", 
                              path  =  [txt_file],
                              classes = cons_classes, 
                              transform = test_transforms,
                               num_element = 30,
                              train = False) 
        cl_class_f = cl_class.samples 
        class_filelist = [] 
        for i in range(len(cl_class_f)):
            class_filelist.append(cl_class_f[i][0])
        psnr, bpp = compress_with_ac(model, 
                                    class_filelist,
                                    device, -1,
                                    loop=False, 
                                    writing= args.writing + args.quality + "/" + cl + "_")
        res[cl] = [bpp,psnr]
        print(bpp," ",psnr)
    return res