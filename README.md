# TBINet
[ACCV2022] [Three-Stage Bidirectional Interaction Network for Efficient RGB-D Salient Object Detection](https://openaccess.thecvf.com/content/ACCV2022/papers/Wang_Three-Stage_Bidirectional_Interaction_Network_for_Efficient_RGB-D_Salient_Object_Detection_ACCV_2022_paper.pdf)

## Abstract
The addition of depth maps improves the performance of salient object detection (SOD). However, most existing RGB-D SOD methods are inefficient. We observe that existing models take into account the respective advantages of the two modalities but do not fully explore the roles of cross-modality features of various levels. To this end, we remodel the relationship between RGB features and depth features from a new perspective of the feature encoding stage and propose a three-stage bidirectional interaction network (TBINet). Specifically, to obtain robust feature representations, we propose three interaction strategies: bidirectional attention guidance (BAG), bidirectional feature supplement (BFS), and shared network, and use them for the three stages of feature encoder, respectively. In addition, we propose a cross-modality feature aggregation (CFA) module for feature aggregation and refinement. Our model is lightweight (3.7 M parameters) and fast (329 ms on CPU). Experiments on six benchmark datasets show that TBINet outperforms other SOTA methods. Our model achieves the best performance and efficiency trade-off.

![image](https://user-images.githubusercontent.com/86772240/222141709-62ef796f-dca1-480c-a7b4-30213af8d198.png)

## Downloads
* Predicted saliency maps: [Google Drive](https://drive.google.com/file/d/112MHD1op2iIsFtgdNldz4WvcfNCupLsr/view?usp=sharing) [NJU2K, STERE, NLPR, SIP, DES, SSD, LFSD, ReDWeb-S, COME-E, COME-H] 
* Training dataset: [Google Drive](https://drive.google.com/file/d/1Orss85k3wEUgDhItwT1goEN6WQFA1SOw/view?usp=sharing)
* Testing dataset: [Google Drive](https://drive.google.com/file/d/1sWJqCg2dAKSSkfrvB7zkwwsW6Ybd4Gd1/view?usp=sharing)
* Pretrained weights: [link](https://github.com/AWORKERINKIKIMORE/TBINet/raw/main/TBINet.pth)

