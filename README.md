# Seq-Unet
Seq-UNet (Sequential U-Net) is an innovative semantic segmentation approach that enhances the segmentation of image sequences by leveraging previous predictions. This project specifically focuses on the real-time semantic segmentation of the median nerve in ultrasound videos.

## Overview
- **Objective:** Achieve real-time semantic segmentation of the median nerve in ultrasound video sequences.
- **Key Features:**
  - Utilizes sequential information for improved segmentation
  - Incorporates human-in-the-loop refinement
  - Designed for real-time performance
- **Application:** Medical imaging, specifically ultrasound video analysis

## Implementation
### Data
- A private dataset consisiting of ultrasound videos of the median nerve of 100 patients
- Each video consisted of 300 frames, spanning from the wrist to the elbow - 24,000 images for training, 6000 images split for validation and testing
- Data augmentation - horizontal flipping, random affine, rotation and crops

### Model 
A standard U-Net architecture was used with 2 input channels
- one containing the grascale image of the current frame
- second containing the prediction of the previous frame, acting as soft attention

### Training 
- Initial training without any prior mask, slowly introduced to prevent overfitting
- Loss function: Combination of Log Cosh Dice loss + Boundary Loss
- Training duration: Trained for 30 epochs 
- Hardware used: Nvidia A6000 GPU

### Human-in-the-loop
- Real time expert correction on suboptimal predictions
- 2% of frames corrected by experts
- Resulted in 15% overall performance increase


## Results
### Performance 
The model was tested on the videos of 10 patients
| Model                           | Precision | Recall | Dicescore |
| ------------------------------- | --------- | ------ | --------- |
| U-Net                           | 0.684     | 0.754  | 0.709     |
| U-Net++                         | 0.781     | 0.766  | 0.755     |
| Seq-Unet                        | 0.739     | 0.798  | 0.748     |
| Seq-Unet with expert annotation | 0.837     | 0.873  | 0.862     |

![unet_with_prior](https://github.com/user-attachments/assets/541cb811-6f16-4d3f-abcc-8db6f3362e13)

### Discussion
- Prior models showed minimal improvement over regular counterparts, possible overfitting to the prior mask
- Human-in-the-loop approach significantly boosted performance
- Repeated exposure to the same frame improved predictions, indicating self-correction capability

## Key Learnings
- Learning how to prevent the model from overfitting to the prior was a big part of the process.
- Using the human in the loop approach and integrating it into the workflow

## Future Work
1. Explore applications in other sequential segmentation tasks
2. Investigate methods to further reduce the need for human intervention
3. Optimize model for faster inference on mobile devices

