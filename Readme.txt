## Prerequisites
* packages: pytorch + torchvision, compressai, numpy
* pre-trained models:
    * We use [Balle 2018] hyperprior as base model, so the following compressAI pretrain model should be downloaded:
    ```bash
    wget https://compressai.s3.amazonaws.com/models/v1/bmshj2018-hyperprior-1-7eb97409.pth.tar
    wget https://compressai.s3.amazonaws.com/models/v1/bmshj2018-hyperprior-2-93677231.pth.tar
    wget https://compressai.s3.amazonaws.com/models/v1/bmshj2018-hyperprior-3-6d87be32.pth.tar
    wget https://compressai.s3.amazonaws.com/models/v1/bmshj2018-hyperprior-4-de1b779c.pth.tar
    wget https://compressai.s3.amazonaws.com/models/v1/bmshj2018-hyperprior-5-f8b614e1.pth.tar
    wget https://compressai.s3.amazonaws.com/models/v1/bmshj2018-hyperprior-6-1ab9c41e.pth.tar
    wget https://compressai.s3.amazonaws.com/models/v1/bmshj2018-hyperprior-7-3804dcbd.pth.tar
    wget https://compressai.s3.amazonaws.com/models/v1/bmshj2018-hyperprior-8-a583f0cf.pth.tar
    ```
* dataset:
    * We use Kodak dataset with 24 images: https://r0k.us/graphics/kodak/, so the dataset should be downloaded