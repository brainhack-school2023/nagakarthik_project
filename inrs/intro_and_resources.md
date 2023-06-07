## Introduction to Implicit Neural Representations for Super-resolution

This document gives a general introduction to implicit neural representations (INRs), briefly describes the method for using INRs for super-resolution, and lists some resources for further reading. 

### INR101

In traditional deep learning (DL), data are often represented as arrays [1]. For instance, images as 2D/3D grids of pixels/voxels, 3D shapes as occupancy values noted by 0 or 1, etc. Such discrete representations grow linearly in space and time complexity with increasing spatial resolution of images. As a result, most DL methods with 3D medical images, work in the realm of image patches. Recently, a novel representation that parameterizes images (or, more generally, discrete signals) as continous functions have gained popularity [2,3]. Images are represented by continuous functions that maps a 2D/3D coordinate to an intensity value (i.e. the RGB/Grayscale color). When such functions are parameterized by neural networks, they are referred to as implicit neural representations. Such a parameterization involving functional representations offers several advantages: 
1. **Agnosticity to spatial resolution**: Since INRs deal directly with the pixel coordinates, the learned representations are not tied to the spatial resolution anymore. As a result, inference can be done at arbitrary resolutions 
2. **Near constant memory**: Unlike discrete grids where memory grows cubically with increasing resolution of images, the memory required for continuous parameterization of the image is independent of the the spatial resolution and only grows with the complexity of the underlying image. 
3. **Multi-modality**: The learned implicit representations are multi-modal, in that they can used for various downstream tasks such as image classification, synthesis, and segmentation. 

The basic building block of INRs is a multi-layer perceptron (or, a fully connected neural network) that takes as input a 3D _(x,y,z)_ coordinate and outputs and intensity value at that coordinate. Typically, some transformations are applied to the input coordinates such as sinusoidal activations [3], Fourier feature mapping [4], and positional encodings [5] so the high frequency content in the images can be recovered. 

TODO: add figure of relu mlp taken from youtube

A major issue when dealing with fully connected, ReLU-based neural networks (NNs) is their inability to model high-frequency content in an image/object, hence resulting in blurred outputs. This phenomenon is referred to as the "spectral bias" [6] of NNs - a consequence of having over-parameterized networks that prioritize learning simple patterns from the inputs in the hopes of improving generalization on unseen data. A relatively well-known solution lies in using Fourier feature (FF) mapping [4], which projects the input coordinates to a higher dimensional Fourier space, before passing them through the NN. This simple transformation enables learning high-frequency functions and reconstructs sharp outputs. The figure below shows the resulting reconstructions with and without FF mapping. 

TODO: Add Figure 1 from the paper. 


### INRs for Super-resolution

One of the earliest works using INRs for MRI super-resolution reconstruction used 3 low-resolution images (one each of sagittal, coronal, and axial) for reconstructing a single high-resolution image [7]. Another recent work [8] used two rigidly-registered contrasts for a given subject for super-resolution. The main idea here is to use the various images (typically with different spatial resolutions and fields-of-view) as a means of mutual information transfer that is ultimately useful for super-resolution reconstruction. Follwoing these approaches, the figure below shows the method used for spinal cord MRI super-resolution in this project. 

TODO: Add figure from slides


### Further Reading on Medical Applications

### References
1. [From data to functa: Your data point is a function and you can treat it like one](https://arxiv.org/abs/2201.12204) (Dupont et. al. 2022)
2. [DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](https://arxiv.org/abs/1901.05103) (Park et al. 2019)
3. [Implicit Neural Representations with Periodic Activation Functions](https://vsitzmann.github.io/siren/) (Sitzmann et al. 2020)
4. [Fourier features let networks learn high frequency functions in low dimensional domains](https://people.eecs.berkeley.edu/~bmild/fourfeat/) (Tancik et al. 2020) 
5. [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://www.matthewtancik.com/nerf) (Mildenhall et al. 2020)
6. [On the Spectral Bias of Neural Networks](https://arxiv.org/pdf/1806.08734.pdf) (Rahaman et al. 2018)
7. [IREM: High-Resolution Magnetic Resonance Image Reconstruction via Implicit Neural Representation](https://arxiv.org/pdf/2106.15097.pdf) (Wu et al. 2021)
8. [Multi-contrast MRI Super-resolution via Implicit Neural Representations](https://arxiv.org/pdf/2303.15065.pdf) (McGinnis et al. 2023)
