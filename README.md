# STA-DRN
Code for the paper 'Spatial-Temporal Attention Network for Depression Recognition from Facial Videos'

## How to Run
You can directly execute the `train.py` or `test.py` scripts with your own dataset.

To proceed:
1. Create a CSV file containing the `path`, which is the folder containing the images extracted from each video, and the corresponding `label`, representing the ground truth of that video.
2. Make sure to modify line 54 in both `train.py` and `test.py` to suit your dataset.

## Notes
Before running, ensure the videos are preprocessed to extract the required images.

Kindly note that due to authorization constraints, we are unable to share the AVEC datasets here. Therefore, it is necessary for you to independently extract, crop, and align the facial data.

It's worth highlighting that this model can be applied to various video-based tasks. We encourage you to give it a try!
