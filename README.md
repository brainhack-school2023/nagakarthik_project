## Super-resolution Reconstruction of Spinal Cord MRI with Implicit Neural Representations

<p align="center">
   <img width="900" alt="spinal-cord-inrs" src="https://github.com/brainhack-school2023/nagakarthik_project/assets/53445351/fd185c4a-c886-45d7-831a-928daa6728b9">
</p>


### About Me

Hello, I am Naga Karthik, a second-year PhD student at Polytechnique Montreal. My research broadly focuses on developing deep learning-based methods for medical image analysis. My current research project aims at designing novel methods for automatically segmenting lesions in traumatic spinal cord (SC) injury patients. During the first year of my PhD, I have worked on continual/lifelong learning methods for segmenting multiple sclerosis lesions in the brain. 

<a href="https://github.com/naga-karthik">
<img width="100" alt="Screen Shot 2023-05-08 at 3 30 47 PM" src="https://user-images.githubusercontent.com/53445351/236916753-8d15c32b-07c7-45a2-8227-039f36a44d03.png">
   <br /><sub><b>Naga Karthik</b></sub>
</a>


## Project Summary

This project explores the idea of using implicit neural representations (INRs) for reconstruction super-resolution (or, high-resolution) spinal cord MR images. 


### Introduction
Acquisition of MR images strikes a delicate balance between the scan time, spatial resolution of the image, and the signal-to-noise (SNR) ratio [1]. While images with isotropic resolution are ideal, they cannot always be acquired due to constraints such as patients' condition, motion artifacts, possibly limited MRI resources, etc. [2]. Moreover, improving spatial resolution comes at the high cost of lowering SNR and/or increasing the acquisition time. As a result, it is a common practice to require multiple low-resolution anisotropic images with high in-plane resolution. 

**Super-resolution** &ensp; Given multiple low-resolution images as inputs, can we combine them in a way that generates a high-resolution output? Does this method improve the resolution and the acquistion time over directly acquiring a high-resolution image? While several methods, for instance, iterative back projection, regularized least squares, etc., exist in the literature, this project explores INRs for reconstructing high-resolution images. 

**INRs**  &ensp; Implicit neural representations present a novel way of parameterizing images. Instead of considering images as a discrete grid-based representations (of an object/scene), INRs provide a powerful alternative for parameterizing images as continuous functions that map a 3D coordinate to its intensity value at that coordinate. As such continuous functions are analytically intractable, INRs use neural networks for approximating the functions. 

Intuitively, going from discrete to continuous representations is not uncommon. For instance, 1D discrete time-signals are approximations of a continuous function sampled at discrete points in time. For images, this implies transitioning from pixels with discrete boundaries to (continuous) RGB values where the boundaries of pixels are no longer visible as shown in the figure below. 

<p align="center">
<img width="450" alt="d2c_pixels" src="https://github.com/brainhack-school2023/nagakarthik_project/assets/53445351/531f1b26-4290-4ca3-9b8b-2931f72ced12">
</p>


### Objectives

Given the background on super-resolution and INRs, the objectives of my project were as follows: 

1. Understand the concept of implicit representations of images and their potential applications. ✅
2. Train a neural network (particuarly, a multi-layer perceptron) to reconstruct a high-resolution MRI of a spinal cord given two different low-resolution views as inputs. ✅
3. Perform an ablation study on the neural network parameters to analyse its effect on the reconstruction accuracy. ✅


### Data

The Spine Generic Public Database [3,4] was used in this project. This is a BIDS-standarized multi-site dataset consisting of 6 contrasts from a single healthy subject acquired using the spine-generic protocol. The dataset is open-source and can be downloaded via git-annex. The installation procedure is described in [README](https://github.com/brainhack-school2023/nagakarthik_project/blob/main/data/spine-generic/README.md) inside the `data/` folder along with the specific subjects used. 

In the context of this project, only the T2w contrast was used as it is an isotropic image. Typically, super-resolution methods tend to use isotropic images as they could be used as the ground-truth, which helps in quantifying the reconstruction accuracy. 


### Tools 

The following tools were used in the project: 

* `bash` for all terminal-related commands and running python scripts from the command-line
* `git` and `GitHub` for version control
* `git-annex` for downloading the spine-generic dataset
* `python` for all the code. Notable packages:
   * `scikit-learn` for computing reconstruction accuracy metric
   * `nibabel` and `nilearn` for data I/O
   * `pytorch` for training INRs (i.e. deep learning)
* `jupyter` (notebook) for plotting and analysing the results


### Deliverables

The deliverables for this project are:

* Introduction to INRs and references to existing literature --> can be found [here](https://github.com/brainhack-school2023/nagakarthik_project/blob/main/inrs/intro_and_resources.md#inrs-for-super-resolution)
* Data and preprocessing scripts used --> can be found [here](https://github.com/brainhack-school2023/nagakarthik_project/tree/main/data) and [here](https://github.com/brainhack-school2023/nagakarthik_project/tree/main/code/preprocessing)
* Code and the related documentation for training an INR --> can be found [here](https://github.com/brainhack-school2023/nagakarthik_project/tree/main/code)
* Jupyter notebook containing the ablation study analysis --> can be found 


### Results 

This section presents the super-resolution results and the ablation study analysis. A brief explanation of the method including the inputs and the model used, could be found [here](https://github.com/brainhack-school2023/nagakarthik_project/blob/main/inrs/intro_and_resources.md#inrs-for-super-resolution).


#### High-resolution images of the Spinal Cord

The figure below shows a comparison between the original (ground-truth, GT) image (left) and the reconstructed image (right), both at 0.8 mm isotropic resolution. Focusing on the zoomed patch, we can observe that the prediction is smoother compared to the original image. 

<p align="center">
<img width="800" alt="Screen Shot 2023-06-07 at 10 50 38 PM" src="https://github.com/brainhack-school2023/nagakarthik_project/assets/53445351/cbbef089-756e-4562-ae76-ed02ecec729b">
</p>

One of the advantages of INRs is that images can be reconstructed at an arbitrary resolution. This means that one can generate a high resolution image that is not bounded by the resolution of the GT image. The figure below shows the model's prediction of 0.5 mm^3 isotropic image. For such outputs, it is important to note that one can only perform a visual assessment of the reconstruction quality as a quantitative assessment is infeasible due to the lack of GT image.

<p align="center">
<img width="500" alt="iso0 5" src="https://github.com/brainhack-school2023/nagakarthik_project/assets/53445351/f784b057-1d4c-4007-933b-df82f5a58efe">

</p>




#### Ablation Study: Effect of the Fourier feature mapping dimension on the reconstruction accuracy

A crucial step when training a NN to reconstruct a high-resolution image is the projection of the 3D coordinates to a higher dimensional space using Fourier feature mapping. As explained in the [methods section](https://github.com/brainhack-school2023/nagakarthik_project/blob/main/inrs/intro_and_resources.md#inrs-for-super-resolution), such a projection helps in learning (and reconstructing) the high-frequency content in the image. Naturally, the dimensionality of the Fourier feature space plays an importat role in the quality of the reconstruction. Therefore, the purpose of this ablation study is to tweak this parameter and observe its downstream effect on the reconstruction accuracy. Two metrics, namely, structural similarity index (SSIM) and peak signal-to-noise ratio (PSNR), are used as quantitative metrics. 

**SSIM** &ensp; It is a metric used to measure the similarity between two images. It is designed to evaluate the perceptual quality of an image by comparing its structural information with a reference image. SSIM takes into account the luminance, contrast, and structural similarities between corresponding patches of the images.

**PSNR** &ensp; It is a commonly used metric to measure the quality or fidelity of a reconstructed or compressed image or video compared to the original, reference signal. It quantifies the ratio between the maximum possible power of a signal (usually taken as the maximum possible pixel value) and the distortion introduced by the compression or reconstruction process. It must be noted that PSNR is based solely on pixel-wise differences and does not always correlate perfectly with human perception. Typically, metrics such as SSIM, are often used in conjunction with PSNR to provide a more comprehensive assessment of visual quality.

The plots below show the SSIM (left) and PSNR (right) values as function of the dimensionality of the Fourier features (higher the metrics, the better). We observe that the reconstruction accuracy increases when increasing the Fourier feature dimensionality in both cases. 


<p align="center">
<img width="800" alt="ff_map_dim_vs_ssim_psnr" src="https://github.com/brainhack-school2023/nagakarthik_project/assets/53445351/17893555-0ad9-4fe3-9893-7fb83165afa4">
</p>


### Conclusions

Neural implicit representations present a powerful alternative for parameterizing discrete voxel-grid based 3D images. The parameterization is done by learning (approximating) a continuous function that maps the input coordinate to its intensity value at that image. Based on two downsampled views of the same contrast, the model was trained to reconstruct a high-resolution image using voxel-wise mean-squared error as the loss function. Lastly, an ablation study on the effect of the dimensionality of Fourier features showed that projecting the input coordinates to higher dimensions improves the reconstruction accuracy and results in better high-resolution images.  


### References

1. Plenge, Esben et al. “Super-resolution methods in MRI: can they improve the trade-off between resolution, signal-to-noise ratio, and acquisition time?.” Magnetic resonance in medicine vol. 68,6 (2012): 1983-93. 
2. McGinnis, Julian et al. “Multi-contrast MRI Super-resolution via Implicit Neural Representations.” ArXiv abs/2303.15065 (2023): n. Pag.
3. Cohen-Adad et al. Generic acquisition protocol for quantitative MRI of the spinal cord. Nature Protocols 2021 (doi: 10.1038/s41596-021-00588-0).
4. [Spine Generic Public Dataset (Single Subject)](https://github.com/spine-generic/data-single-subject#spine-generic-public-database-single-subject)
