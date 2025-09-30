
import torch 
import math
import torch.nn as nn



class GateLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.dist_metric = nn.MSELoss()

    def forward(self,output, target):
        out = {}
        out["CrossEntropy"] = self.loss(output["logits"], target[1]) 
        out["loss"] = out["CrossEntropy"]
        out["mse_loss"] = self.dist_metric(output["x_hat"], target[0])

        return out
    


class GateDistorsionLoss(nn.Module):
    def __init__(self, lmbda = 0.1, policy = "e2e"):
        super().__init__()
        self.lmbda = lmbda 
        self.gate_metric = nn.CrossEntropyLoss()
        self.dist_metric = nn.MSELoss()
        self.policy = policy




    def forward(self, output, target):

        out = {}

        logits = output["logits"]
        classes = target[1]
        out["CrossEntropy"] = self.gate_metric(logits, classes)
        N, _, H, W = target[0].size()      

        
        num_pixels = N * H * W
    
           


        out["y_bpp"] = torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
        out["z_bpp"]  = torch.log(output["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels) 
        out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"].values())   
        
        
        out["mse_loss"] = self.dist_metric(output["x_hat"], target[0])  #
        #print("mse loss is: ",out["mse_loss"],"  ",out["CrossEntropy"]," ",self.lmbda *255**2 * out["mse_loss"] )
        if self.policy == "e2e":
            out["loss"] =   self.lmbda * 255**2 * out["mse_loss"]  + out["CrossEntropy"]
        else:
            out["loss"] =   self.lmbda * 255**2 * out["mse_loss"] 

        return out  
    





class RateDistortionLoss(nn.Module):

    def __init__(self, lmbda = 1e-2):
        super().__init__()

        self.dist_metric = nn.MSELoss()
        self.lmbda = lmbda 
        self.type = type


    def forward(self, output, target):
        N, _, H, W = target.size()      
        out = {}
        out["mse_loss"] = self.dist_metric(output["x_hat"], target)
        num_pixels = N * H * W
    
           


        out["y_bpp"] = torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
        out["z_bpp"]  = torch.log(output["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels) 
        out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"].values())   
        
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"] 

        return out  
    




class AdaptersLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.dist_metric = nn.MSELoss()
        self.lmbda = 0.0483
        self.loss = nn.CrossEntropyLoss()



    def forward(self, output, target):
    
        out = {}
        out["mse_loss"] = self.dist_metric(output["x_hat"], target[0])       
        out["loss"] =   self.lmbda * 255 ** 2 *out["mse_loss"]




        N, _, H, W = target[0].size()      
        num_pixels = N * H * W
    
           
        out["CrossEntropy"] =  self.loss(output["logits"], target[1]).to("cuda")

        out["y_bpp"] = torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
        out["z_bpp"]  = torch.log(output["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels) 
        out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"].values())   



        return out


from pytorch_msssim import ms_ssim

class MssimLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self ):
        super().__init__()

        self.metric = ms_ssim
        self.loss = nn.CrossEntropyLoss()
  

    def forward(self, output, target):
        N, _, H, W = target[0].size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))for likelihoods in output["likelihoods"].values())

        out["mse_loss"] = self.metric(output["x_hat"], target[0], data_range=1)
        distortion = 1 - out["mse_loss"]




        out["loss"] =  32*distortion 
        return out



class DistorsionLoss(nn.Module):

    def __init__(self, lmbda = 1):
        super().__init__()

        self.dist_metric = nn.MSELoss()
        self.lmbda = lmbda #1 #0.0483



    def forward(self, output, target):
    
        out = {}
        out["mse_loss"] = self.dist_metric(output["x_hat"], target)       
        out["loss"] =   self.lmbda * 255 ** 2 *out["mse_loss"]




        N, _, H, W = target.size()      
        num_pixels = N * H * W
    
           


        out["y_bpp"] = torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
        out["z_bpp"]  = torch.log(output["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels) 
        out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"].values())   



        return out
    



    



