## Introduction to Implicit Neural Representations for Super-resolution

This document gives a general introduction to implicit neural representations (INRs), briefly describes the method for using INRs for super-resolution, and lists some resources for further reading. 

### INR101

In traditional deep learning (DL), data are often represented as arrays [1]. For instance, images as 2D/3D grids of pixels/voxels, 3D shapes as occupancy values noted by 0 or 1, etc. Such discrete representations grow linearly in space and time complexity with increasing spatial resolution of images. As a result, most DL methods with 3D medical images, work in the realm of image patches. Recently, a novel representation that parameterizes images (or, more generally, discrete signals) as continous functions have gained popularity [2,3]. Images are represented by continuous functions that maps a 2D/3D coordinate to an intensity value (i.e. the RGB/Grayscale color). When such functions are parameterized by neural networks, they are referred to as implicit neural representations. Such a parameterization involving functional representations offers several advantages: 
1. **Agnosticity to spatial resolution**: Since INRs deal directly with the pixel coordinates, the learned representations are not tied to the spatial resolution anymore. As a result, inference can be done at arbitrary resolutions 
2. **Near constant memory**: Unlike discrete grids where memory grows cubically with increasing resolution of (3D) images, the memory required for continuous parameterization of the image is independent of the spatial resolution and only grows with the complexity of the underlying image. 
3. **Multi-modality**: The learned implicit representations are multi-modal, in that they can used for various downstream tasks such as image classification, synthesis, and segmentation. 

The basic building block of INRs is a multi-layer perceptron (or, a fully connected neural network) that takes as input a 3D _(x,y,z)_ coordinate and outputs and intensity value at that coordinate. Typically, some transformations are applied to the input coordinates such as sinusoidal activations [3], Fourier feature mapping [4], and positional encodings [5], such that the high frequency content in the images can be recovered. 

<p align="center">
  <img width="393" alt="Screen Shot 2023-06-07 at 4 44 02 PM" src="https://github.com/brainhack-school2023/nagakarthik_project/assets/53445351/e302a1ba-24e3-4e21-ae34-ee7f1e6328e7">
  </p> 

A major issue when dealing with fully connected, ReLU-based neural networks (NNs) is their inability to model high-frequency content in an image/object, hence resulting in blurred outputs. This phenomenon is referred to as the "spectral bias" [6] of NNs - a consequence of having over-parameterized networks that prioritize learning simple patterns from the inputs in the hopes of improving generalization on unseen data. A relatively well-known solution lies in using Fourier feature (FF) mapping [4], which projects the input coordinates to a higher dimensional Fourier space, before passing them through the NN. This simple transformation enables learning high-frequency functions and reconstructs sharp outputs. 
<!-- The figure below shows the resulting reconstructions with and without FF mapping.  -->


### INRs for Super-resolution

One of the earliest works using INRs for MRI super-resolution reconstruction used 3 low-resolution images (one each of sagittal, coronal, and axial) for reconstructing a single high-resolution image [7]. Another recent work [8] used two rigidly-registered contrasts for a given subject for super-resolution. The main idea here is to use the various images (typically with different spatial resolutions and fields-of-view) as a means of mutual information transfer that is ultimately useful for super-resolution reconstruction. Follwoing these approaches, the figure below shows the method used for spinal cord MRI super-resolution in this project. 

<p align="center">
<img width="900" alt="Screen Shot 2023-06-07 at 4 51 41 PM" src="https://github.com/brainhack-school2023/nagakarthik_project/assets/53445351/90a3e4cc-f979-4fc4-adb1-d316c3be05e2">
</p>

Given an 0.8mm isotropic T2w image as input, two downsampled views are created. The first view is downsampled with a factor of 5 along the _axial_ plane resulting in a `0.8 x 0.8 x 4.0 mm^3` image and the second view is downsampled with a factor 5 along the sagittal plane resulting in a `4.0 x 0.8 x 0.8 mm^3` image. Then, the sets of 3D coordinates corresponding to each view are concatenated. Following the literature, the coordinates are projected to a Fourier space using the mapping dimension `L=256`, which are then passed through the NN. The model is supervised by the Mean-squared error loss between the predicted intensity and the ground-truth intensity of the 0.8 mm isotropic image.


### Further Reading on Medical Applications

Here are some more references for various categories: 

* Segmentation
  * [Implicit Neural Representations for Medical Imaging Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-16443-9_42) (Khan et al. 2022)
* Registration
  * [Implicit Neural Representations for Deformable Image Registration](https://proceedings.mlr.press/v172/wolterink22a/wolterink22a.pdf) (Wolterink et al. 2022)
  * [Diffeomorphic Image Registration with Neural Velocity Field](https://openaccess.thecvf.com/content/WACV2023/papers/Han_Diffeomorphic_Image_Registration_With_Neural_Velocity_Field_WACV_2023_paper.pdf) (Han et al. 2023)


### References
1. [From data to functa: Your data point is a function and you can treat it like one](https://arxiv.org/abs/2201.12204) (Dupont et. al. 2022)
2. [DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](https://arxiv.org/abs/1901.05103) (Park et al. 2019)
3. [Implicit Neural Representations with Periodic Activation Functions](https://vsitzmann.github.io/siren/) (Sitzmann et al. 2020)
4. [Fourier features let networks learn high frequency functions in low dimensional domains](https://people.eecs.berkeley.edu/~bmild/fourfeat/) (Tancik et al. 2020) 
5. [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://www.matthewtancik.com/nerf) (Mildenhall et al. 2020)
6. [On the Spectral Bias of Neural Networks](https://arxiv.org/pdf/1806.08734.pdf) (Rahaman et al. 2018)
7. [IREM: High-Resolution Magnetic Resonance Image Reconstruction via Implicit Neural Representation](https://arxiv.org/pdf/2106.15097.pdf) (Wu et al. 2021)
8. [Multi-contrast MRI Super-resolution via Implicit Neural Representations](https://arxiv.org/pdf/2303.15065.pdf) (McGinnis et al. 2023)
