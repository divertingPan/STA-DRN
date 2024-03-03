# STA-DRN
Code for the paper 'Spatial-Temporal Attention Network for Depression Recognition from Facial Videos' (doi: https://doi.org/10.1016/j.eswa.2023.121410)

![main figure](https://ars.els-cdn.com/content/image/1-s2.0-S0957417423019127-gr2.jpg)

## Pre-process
[face_detect.py](https://github.com/divertingPan/utility_room/blob/master/face_detect.py): Crop faces from a series of video frame images. The images should be frames already exported from the video. To run the program directly, the path format for storing images should be `dataset_path/video_001/00001.jpg、dataset_path/video_001/00002.jpg`, and another model file [shape_predictor_68_face_landmarks.dat](https://github.com/divertingPan/utility_room/blob/master/shape_predictor_68_face_landmarks.dat) is also required.

## How to Run
You can directly execute the `train.py` or `test.py` scripts with your own dataset.

To proceed:
1. Create a CSV file containing the `path`, which is the folder containing the images extracted from each video, and the corresponding `label`, representing the ground truth of that video.
2. Make sure to modify line 54 in both `train.py` and `test.py` to suit your dataset.

## Notes
Before running, ensure the videos are preprocessed to extract the required images.

Kindly note that due to authorization constraints, we are unable to share the AVEC datasets here. Therefore, it is necessary for you to independently extract, crop, and align the facial data.

It's worth highlighting that this model can be applied to various video-based tasks. We encourage you to give it a try!

## Citation
```
@article{PAN2023121410,
title = {Spatial-Temporal Attention Network for Depression Recognition from facial videos},
journal = {Expert Systems with Applications},
pages = {121410},
year = {2023},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.121410},
url = {https://www.sciencedirect.com/science/article/pii/S0957417423019127},
author = {Yuchen Pan and Yuanyuan Shang and Tie Liu and Zhuhong Shao and Guodong Guo and Hui Ding and Qiang Hu},
keywords = {Depression recognition, Attention mechanism, Video recognition, Deep learning, Visualization, Convolutional neural network},
abstract = {Recent studies focus on the utilization of deep learning approaches to recognize depression from facial videos. However, these approaches have been hindered by their limited performance, which can be attributed to the inadequate consideration of global spatial–temporal relationships in significant local regions within faces. In this paper, we propose Spatial-Temporal Attention Depression Recognition Network (STA-DRN) for depression recognition to enhance feature extraction and increase the relevance of depression recognition by capturing the global and local spatial–temporal information. Our proposed approach includes a novel Spatial-Temporal Attention (STA) mechanism, which generates spatial and temporal attention vectors to capture the global and local spatial–temporal relationships of features. To the best of our knowledge, this is the first attempt to incorporate pixel-wise STA mechanisms for depression recognition based on 3D video analysis. Additionally, we propose an attention vector-wise fusion strategy in the STA module, which combines information from both spatial and temporal domains. We then design the STA-DRN by stacking STA modules ResNet-style. The experimental results on AVEC 2013 and AVEC 2014 show that our method achieves competitive performance, with mean absolute error/root mean square error (MAE/RMSE) scores of 6.15/7.98 and 6.00/7.75, respectively. Moreover, visualization analysis demonstrates that the STA-DRN responds significantly in specific locations related to depression. The code is available at: https://github.com/divertingPan/STA-DRN.}
}
```
