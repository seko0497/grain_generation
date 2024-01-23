# Grain Generation

PyTorch based framework for a diffusion model-based method to generate synthetic images of grain tools and their matching segmentation masks.
These synthetic images can be used to augment the corresponding real-world dataset to [improve
the segmentation accuracy](https://github.com/seko0497/grain_detection)

*The models were also trained on another dataset containing RGB-images of worn 
drilling tools. For legal reasons the data cannot be published*

## Methodology
### Sampling in one step
This approach concatenates an additional color channel for the segmentation masks and generates the segmentation mask simultaneously with the feature image.

&nbsp;
<img src='figures\one_stage_grain.png' align="left">  
&nbsp;
### Sampling in two steps
In this approach a segmentation mask $y_0$ is first sampled using an unconditional diffusion model. The feature images $x_0$ are then generated using a [Semantic Diffusion Model](https://arxiv.org/abs/2207.00050) that is conditioned on the 
previously generated segmentation mask $y_0$.

&nbsp;
<img src='figures\two_stage_grain.png' align="left">  
&nbsp;

### Super-Resolution
It has been found that image quality improves when images and masks are generated at 64 x 64 px.
Therefore, another diffusion model is used to increase the resolution to 256 x 256 px. The lower resolution image is interpolated to the higher resolution and concatenated channel-wise.

&nbsp;
<img src='figures\upsampling_grain.png' align="left">  
&nbsp;

## Example Results

&nbsp;
<p align="center">
<img src='figures\examples.png' align="ceter" width=400> 
</p> 
&nbsp;

## Scores
| Method          | FID Feature 1 | FID Feature 2 | FID Mask | FID Average | 
| --------------- | ------------- | --------------| ---------| ----------- |
|1 Step | 125.54 | 48.24 | 74.85 | 82.98 |
|2 Steps | 58.15 | 42.75 | 137.57 | 78.24 |

*As InceptionV3 is pre-trained on the ImageNet database for FID calculation, the scores are comparatively high, but suitable for comparing the methods. The segmentation results show that a lower FID corresponds to a higher increase in segmentation accuracy*.
