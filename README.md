# STCL-DRNet

This repo is the implementation of "Self-Training and Curriculum Learning Guided Dynamic Refined Network for Remote Sensing Class-Incremental Semantic Segmentation" 
<table>
    <tr>
      <td><img src="PaperFigs\Fig1.png" width = "100%" alt=" paradigm"/></td>
      <td><img src="PaperFigs\Fig2.png" width = "100%" alt="STCL-DRNet"/></td>
    </tr>
</table>


## Dataset Preparation

We select DeepGlobe, iSAID as benchmark datasets and create train, val, test list for researchers to follow. 

**In the following, we provide the detailed commands for dataset preparation.**

### DeepGlobe

      download the DeepGlobe dataset and unzip and move to the data/DeepGlobe2018 folder
      python data/rgb2label.py to generate the one-channel label and the data folder
      
### iSAID

      we use the mmsegmentation/tools/dataset_converters/iSAID.py to crop the picture to 512*512 overlaping 384 
      and generate the one channel labels.
### Datasets structures
      
```
data/
    --- DeepGlobe2018
        --- land_train/
        --- onechannel_label/
    --- iSAID
        --- img_dir
            --- train
            --- val
            --- test
        --- ann_dir
            --- train
            --- val
            --- test
```

## Install

     ```
     pip install -r requirements.txt
     ``` 

## Training


1. Class-Incremental Segmentation on DeepGlobe

     ```
     cd STCL-DRNet/scripts/train
     
     sh DeepGlobe_3-3.sh # 3-3 incremental learning
     sh DeepGlobe_2-2.sh # 2-2 incremental learning
     sh DeepGlobe_1-1.sh # 1-1 incremental learning
     ```

2. Class-Incremental Segmentation on iSAID:

     ```
     cd STCL-DRNet/scripts/train
     
     sh iSAID_14-1.sh # 14-1 incremental learning
     sh iSAID_10-5.sh # 10-5 incremental learning
     sh iSAID_10-1.sh # 10-1 incremental learning
     ```



### Testing
  
Trained with the above commands, you can get a trained model to test the performance of your model.   

1. test on DeepGlobe

    ```
     cd STCL-DRNet/scripts/test
     
     sh test_DeepGlobe_3-3.sh # 3-3 incremental learning
     sh test_DeepGlobe_2-2.sh # 2-2 incremental learning
     sh test_DeepGlobe_1-1.sh # 1-1 incremental learning
     ```

2. test on iSAID
    ```
    cd STCL-DRNet/scripts/train
     
     sh test_iSAID_14-1.sh # 14-1 incremental learning
     sh test_iSAID_10-5.sh # 10-5 incremental learning
     sh test_iSAID_10-1.sh # 10-1 incremental learning
   ```
   
### Results
<table>
    <tr>
      <td><img src="PaperFigs\table 1.png" width = "100%" alt=" table 1"/></td>>
      <td><img src="PaperFigs\table 2.png" width = "100%" alt=" table 2"/></td>>
    </tr>
</table>

If you have any question, please discuss with me by sending email to lyushuchang@buaa.edu.cn.

# References
Many thanks to their excellent works [SSUL](https://github.com/clovaai/SSUL)
