# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
from random import sample, seed, shuffle
import random






    


class ImageFolder(Dataset):


    def __init__(self, root, num_images = 24000, transform=None, split="train", num_ex_tr = 0):
        splitdir = Path(root) / split / "data"

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples =[]# [f for f in splitdir.iterdir() if f.is_file()]

        num_images = num_images
            

        for i,f in enumerate(splitdir.iterdir()):
            if i%10000==0:
                print(i)
            if i <= num_images: 
                if f.is_file() and i < num_images:
                    self.samples.append(f)
            else:
                break


        cont_sk, cont_clip = 0,0
        if num_ex_tr > 0:

            if split == "train":
                pth = "/scratch/dataset/DomainNet/splitting/mixed/train.txt"
            else:
                pth = "/scratch/dataset/DomainNet/splitting/mixed/valid.txt"

            file_d = open(pth,"r") 
            Lines = file_d.readlines()

            for i,lines in enumerate(Lines):
                if i%10000==0:
                    print(i)
                if i > 300000:
                    break
                if int(lines.split(" ")[1]) == 1 and cont_sk < num_ex_tr:
                    self.samples.append(lines.split(" ")[0]) 
                    cont_sk +=1          
                if int(lines.split(" ")[1]) == 2 and cont_clip < num_ex_tr:
                    self.samples.append(lines.split(" ")[0]) 
                    cont_clip += 1    
            

        print("ma esco da qua???????")
        random.shuffle(self.samples)
        print("ma esco da qua???????")

        print("lunghezza: ",len(self.samples))
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)
    

import os






    











class TestKodakDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = [os.path.join(self.data_dir,f) for f in os.listdir(self.data_dir)]

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        transform = transforms.Compose(
        [ transforms.ToTensor()]
    )
        return transform(image)

    def __len__(self):
        return len(self.image_path)


def  handle_dataset(args,device): 

    if args.dataset_type == "openimages":


        train_transforms = transforms.Compose([transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
        train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms, num_images=args.num_images_train, num_ex_tr = 8032)
        train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True, pin_memory=(device == "cuda"),)

        valid_transforms = transforms.Compose([transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
        valid_dataset = ImageFolder(args.dataset, split="test", transform=valid_transforms, num_images=args.num_images_val, num_ex_tr = 116)
        valid_dataloader = DataLoader(valid_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=False,pin_memory=(device == "cuda"),)
        #test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)sss
        test_dataset = TestKodakDataset(data_dir="/scratch/dataset/kodak")
        test_dataloader = DataLoader(test_dataset, batch_size=1,num_workers=args.num_workers,shuffle=False,pin_memory=(device == "cuda"),)
        
        filelist = [os.path.join("/scratch/dataset/kodak",f) for f in os.listdir("/scratch/dataset/kodak")]

        return train_dataloader, valid_dataloader, test_dataloader, filelist






class AdapterDataset(Dataset):



    def __init__(self, root, path=["train.txt"], transform = None, classes =  ["natural","sketch","clipart","watercolor","comic","infographics","quickdraw"], num_element = 2000, train = True):

        self.classes = classes
        self.samples =[] 
        for p in path:
            splitdir = root + "/" + p
            file_d = open(splitdir,"r") 
            Lines = file_d.readlines()
            self.class_label = {}
            if train is True:
                for i,cl in enumerate(classes):
                    self.class_label[cl] = i
            else: 
                self.class_label["kodak"] = 0
                self.class_label["clic"] = 0

                for i,cl in enumerate(classes):
                    if cl != "natural" or cl != "openimages":
                        self.class_label[cl] = i

            print("This are the labels ",self.class_label)
            for cl in list(self.class_label.keys()):
                counter = 0
                for i,lines in enumerate(Lines):

                    spec_num_element = num_element if cl == "openimages" else num_element
                    if cl in splitdir and counter < spec_num_element: #splitdir
                        
                        if  lines.split(" ")[0][-1] == '\n': #"quickdraw" in cl : #"kodak" not in cl and "clic" not in cl:
                            ln = lines.split(" ")[0][:-1]
                        else: 
                            ln = lines.split(" ")[0]
                        if "quickdraw" in splitdir: 
                            self.samples.append(("/scratch/dataset/DomainNet/data/" + ln, str(self.class_label[cl])))
                        else:
                            self.samples.append((ln, str(self.class_label[cl])))
                        counter +=1


        print("lunghezza: ",len(self.samples))
        self.transform = transform

    def __getitem__(self, index):

        img = Image.open(self.samples[index][0]).convert("RGB")
        classes = int(self.samples[index][1])
        if self.transform:
            return self.transform(img),classes
        return img, classes

    def __len__(self):
        return len(self.samples)
    


from sklearn.model_selection import train_test_split

def create_data_file_for_multiple_adapters(classes = ["documents"], #"natural""natural","comic",
                                            split = "test", 
                                           savepath = "/scratch/dataset/domain_adapter/MixedImageSets",
                                           num_im_per_class = 200,
                                    random_seed = 42):
    seed(random_seed)
    for (j,cl) in enumerate(classes):
        print("______________________ ",cl,"___________________________________________")
        fls = savepath + "/" + split + "/_" + cl +  "_.txt"
        f=open(fls , "a+")

        if cl == "natural":
            if split == "train":
                pth = "/scratch/dataset/openimages/train/data"
            else:
                pth = "/scratch/dataset/openimages/test/data"
            lista_immagini = [os.path.join(pth,f) for f in os.listdir(pth)]
            immagini_train = sample(lista_immagini,num_im_per_class)

            for single_images in immagini_train:
                f=open(fls , "a+")
                f.write( single_images + "\n")
                f.close()  
        elif cl  in ("quickdraw"):


            pth = "/scratch/dataset/DomainNet/splitting/" + cl  +  "/total_test.txt"
            file_d = open(pth,"r") 
            Lines = file_d.readlines()
            immagini_pox = []
            for i,line in enumerate(Lines):
                if i%10000==0:
                    print(i)
                f_temp =  "/scratch/dataset/DomainNet/data/"  + line.split(" ")[0]
                #path = os.path.join(line)

                img_t = Image.open(f_temp)

                l,w = img_t.size

                if l > 256 and w > 256:
                    immagini_pox.append(f_temp) # + " " + str(classes[cl]) + "\n")
                        #f=open(fls , "a+")
                        #f.write( f_temp + " " + str(classes[cl]) + "\n")
                        #f.close() 
                else:
                    print("queste dimensioni non vanno bene: ",l,w)
            numero_fin = min(num_im_per_class, len(immagini_pox))
            print("-------> ",numero_fin)
            immagini_train = sample(immagini_pox,numero_fin)

            for files in immagini_train:
                f=open(fls , "a+")
                f.write(files  + "\n")
                f.close()    
        else: # infographics e sketch qua devo fare tutto insieme perché altrimenti non è chiaro 
            print("-----------------> ",cl)
            
            # prendo lista tutti i fil 
            pth = "/scratch/dataset/domain_adapter/" + cl  + "/JPEGImages"
            print("inizio listato")
            tutti_i_file = [os.path.join(pth,f) for f in os.listdir(pth) if rispetto_dimensioni(pth,f) is True]
            print(len(tutti_i_file))
            numero_fin = min(num_im_per_class, len(tutti_i_file))
            immagini_train = sample(tutti_i_file,numero_fin)

            print("Inizio random")
            # Dividi la lista in tre parti
            train, rimanente = train_test_split(tutti_i_file, train_size=0.01, shuffle=True)
            valid, test = train_test_split(rimanente, train_size=0.01, shuffle=True)

            # Stampa le tre parti
            print("Prima parte:", len(train))
            print("Seconda parte:", len(valid))
            print("Terza parte:", len(test))


            fls = savepath + "/" + "train" + "/_" + cl +  "_.txt"
            for files in train:
                f=open(fls , "a+")
                f.write(files  + "\n")
                f.close()   
        
            fls = savepath + "/" + "valid" + "/_" + cl +  "_.txt"
            for files in valid:
                f=open(fls , "a+")
                f.write(files  + "\n")
                f.close()   
            
            fls = savepath + "/" + "test" + "/_" + cl +  "_.txt"
            for files in valid:
                f=open(fls , "a+")
                f.write(files  + "\n")
                f.close()   


            








                        

def rispetto_dimensioni(pth,f):
    with Image.open(os.path.join(pth,f)) as img_t:

        l,w = img_t.size
        if l > 256 and w > 256:   
            return True 
        else:
            return False

def build_test_datafile(type = "infograph", savepath = "/scratch/dataset/DomainNet/splitting/mixed/"):
    if type in ("kodak","clic"):
        classe = "0"
        fls = savepath + "/" + "test_" + type + ".txt"
        f=open(fls , "a+")  
        path = os.path.join("/scratch","dataset",type)
        lista_kodak_immagini =  [os.path.join(path, f) for f in os.listdir(path)]

        for files in lista_kodak_immagini:
            f=open(fls , "a+")
            f.write(files  + " " + classe + "\n")
            f.close()        

    elif type in ("clipart","sketch","painting", "infograph"):
        path = savepath + "valid_infograph_temp.txt"
        file_d = open(path,"r") 
        Lines = file_d.readlines()
        lista_immagini = []

        for line in Lines:
            if type in line:
                lista_immagini.append(line)
        seed(42)
        sub_test_list = sample(lista_immagini,100) # 100 immagini di test

        fls = savepath + "/" + "test_" + type + ".txt"
        f=open(fls , "a+")    

        for s in sub_test_list:
            f=open(fls , "a+")
            f.write(s)
            f.close()  

        fls = savepath + "/" + "valid_" + type + ".txt"
        f=open(fls , "a+")    

        for s in lista_immagini:
            if s not in sub_test_list:
                f=open(fls , "a+")
                f.write(s)
                f.close()  
        print("gol")

               












