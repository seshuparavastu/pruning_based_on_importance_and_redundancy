# Pruning Based on Importance and Redundancy

## Abstract
Convolutional neural networks (CNNs) have become the architecture of choice for state-of-the-art computer vision algorithms. However, CNNs have high computational cost, and this can limit their application, particularly in mobile devices. In order to address this, several pruning techniques have been suggested to remove as many filters as possible with minimal impact on accuracy. However, most of the current approaches remove filters based on a proposed metric for how much information a filter contains independently (filter importance). We believe that effective pruning strategies should be based on eliminating filters based on importance and compressing layers based on redundancy. In this work, we propose a filter redundancy metric based on correlation. We also propose a pruning methodology that first eliminates filters based on importance and subsequently eliminates filters based on redundancy. We demonstrate our methodology on VGG16 with CIFAR-10 dataset. 

## CODE SETUP AND RUNNING

### Setup
 * git clone this repo by doing the following : 


```
       git clone https://github.com/seshuparavastu/pruning_based_on_importance_and_redundancy.git
```
* cd to the directory "pruning_based_on_importance_and_redundancy"

### Running the code
**There are two ways of running this code. In the first one, we start by training the base VGG-16 model from scratch and then run the algorithm (training takes about 1-2 hours to run). In the second way, we use a pre-trained VGG-16 model to save training time (the weights need to be downloaded from google drive) and then run the algorithm on this model. Both ways should yield similar results.**

### Running without the pre-trained model  
* Run the following code, note that the first argument is the threshold for entropy and the second argument is the threshold for Redundancy.

```
        python vgg16_pruning_based_on_importance_and_redundancy_no_pretrained.py 0.3 0.8 
```

This should generate a file with the name "data_stats0.80.3" which contains both the accuracy and the pruning ratio information. The file name is of the format "data_stats\<S\>\<E\>", where "\<S\>" is the redundancy threshold and "\<E\>" is the entropy threshold.

### Running with the pre-trained model 

* Download the "training_4" directory from this google drive https://drive.google.com/drive/folders/1Yd-DQnG-Borb_N87-pcw2Vfrss17gRBx?usp=share_link 
  * One way to do this is using gdown 
      ```
       gdown --folder https://drive.google.com/drive/folders/1Yd-DQnG-Borb_N87-pcw2Vfrss17gRBx?usp=share_link  
      ```
   * **Make sure the "training_4" directory is in the same level as the python script.**
   * This directory should have 3 files - "checkpoint", "cp.ckpt.data-00000-of-00001" and "cp.ckpt.index".

* Now run the following, note that the first argument is the threshold for entropy and the second argument is the threshold for Redundancy

```
        python vgg16_pruning_based_on_importance_and_redundancy.py 0.3 0.8 
```
This should generate a file with the name "data_stats0.80.3" which contains both the accuracy and the pruning ratio information. The file name is of the format "data_stats\<S\>\<E\>", where "\<S\>" is the redundancy threshold and "\<E\>" is the entropy threshold.

NOTE : In this code we simulate pruning by zero-ing out the weights and biases of the filter


 
