import numpy as  np 
import os 
import matplotlib.pyplot as  plt 
from os.path import join

from matplotlib import rc

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Imposta la palette "tab10" di Seaborn
palette = sns.color_palette("tab10")
#rc('text', usetex=True)
rc('font', family='Times New Roman')

new_names = {


    "zou22-kodak-BS":"Reference",
    "zou22-kodak-OQ": "our-quant",
    "zou22-kodak-TQ":"their-quant",
    "zou22-kodak-MR":"our-meanremoval",
    "basezou22-kodak-lrp-gm00":"lrp-gm00",
    "basezou22-kodak-lrp-gm05":"lrp-gm05",
    "basezou22-kodak-nolrp-gm00":"nolrp-gm00",
    "basezou22-kodak-nolrp-gm05":"nolrp-gm05",
    
    




}


Colors = {
    "Reference":[palette[0],'-.'],
    "our-quant":[palette[5],"-"],
    "their-quant":[palette[3],"-"],
    "Zou22-ours-3A":[palette[3],"-"],
    "Zou22-ours-1A":[palette[3],"-"],
    "Zou22-ours-2A":[palette[3],"-"],
    "Proposed":[palette[3],"-"],
    "Zou22-ours-5A":[palette[3],"-"],
    "lrp-gm00": [palette[0],"-"],
    "lrp-gm05": [palette[1],"-"],
    "nolrp-gm00": [palette[2],"-"],
    "nolrp-gm05": [palette[3],"-"],
    "our-meanremoval":[palette[8],"-"]
}

def plot_rate_distorsion(metrics, savepath, dataset, name):

    plt.figure(figsize=(12,8)) # fig, axes = plt.subplots(1, 1, figsize=(8, 5))
    
    list_names =list(metrics.keys()) #["Minnen-ours","Cheng20"]
    name_model = name
    print("name model: ",name_model)
    for i,type_name in enumerate(list_names):



        #if "ours" in type_name:
        #name_model = type_name.split("-")[0]
        print(type_name)
        values = metrics[type_name] 
        psnr = np.sort(values["psnr"])
        bpp =np.sort(values["bpp"])
        cols = Colors[type_name]
        met = psnr


        print("----------------------> ",type_name,": ",met,"   ",bpp)
        if "Proposed" in type_name:  
            plt.plot(bpp,met,cols[1],color = cols[0], label = type_name,markersize=7)
            plt.plot(bpp,met,'o',color =  cols[0], markersize=11)
        else: 
            plt.plot(bpp,met,cols[1],color = cols[0],label = type_name, markersize=7)
            plt.plot(bpp,met,'o',color =  cols[0],marker="X", markersize=18)          
        if "Proposed" in type_name:
            plt.plot(bpp[-1],met[-1],marker="X", markersize=18,color =  cols[0] ) 




    plt.ylabel('PSNR', fontsize = 30)
    plt.yticks([ 37.6,37.65,37.7])
    plt.xticks([0.915,0.917,0.92])
    plt.xlabel('Bit-rate [bpp]', fontsize = 30)
    plt.yticks(fontsize=27)
    plt.xticks(fontsize=27)
    #plt.title('MS-SSIM comparison')
    plt.grid()
    plt.legend(loc='lower right', fontsize = 25)



    met = "psnr"
    nome = name_model + "_" + dataset +   met + ".png"

    cp =  join(savepath,nome)

    plt.grid(True)
    #plt.savefig(cp)
    plt.savefig(cp, dpi=200, bbox_inches='tight', pad_inches=0.01)
    plt.close()  



def define_dictionary(lista_file):

    res = {}
    for f in lista_file:
        


        bpp_q, psnr_q  = [], []
        file1 = open(f, 'r')
        Lines = file1.readlines()

        nome = f.split("\\")[-1].split(".")[0]
        print(nome,"  ",f)
        for line in Lines:

            #print(line.strip().split(" ")[5])
            bpp_q.append(float(line.strip().split(" ")[3]))
            #if "sos" in path:
                    
            psnrt = line.strip().split(" ")[-1]#[4:] # + "." +  line.strip().split(" ")[9].split(".")[1]
            psnr_q.append(float(psnrt))
            print("entro qua")
        
        res[new_names[nome]] = {"bpp": np.array(bpp_q),"psnr": np.array(psnr_q)}

    
    # open jpeg 
    #c = np.load("/scratch/inference/results/files/jpeg.npy",allow_pickle=True)
    #res["jpeg"] = {"bpp": c[0],"mssim":c[1],"psnr":c[2]}

    return res



def main(): 
    dataset = "kodak"
    vtm = False
    psnr_type = True
    model = "zou22"

    lista_file = [join("..","results","files",f) for f in os.listdir(join("..","results","files")) if model in f and  dataset in f]  # "2A" in f or "bas" in f or "4A" in f or 
    print("-----" ,lista_file)#
    res = define_dictionary(lista_file)
    print("done")
    plot_rate_distorsion(res, savepath = "../results/images/", dataset = dataset ,name = model )
    print("DONE")

if __name__ == "__main__":

 
    main()
