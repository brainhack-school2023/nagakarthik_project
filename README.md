## Super-resolution Reconstruction of Spinal Cord MRI with Implicit Neural Representations

TODO: Add a picture

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

**INRs**  &ensp; Implicit neural representations present a novel way of parameterizing images. Instead of considering images as a discrete grid-based representations (of an object/scene), INRs provide a powerful alternative for parameterizing images as continuous functions that map a 3D coordinate to its intensity value at that coordinate. As such continuous functions are analytically intractable, INRs use neural networks as an approximation. 

Intuitively, going from discrete to continuous representations is not uncommon - 1D discrete time-signals are approximations of a continuous function sampled at discrete points in time. For images, this implies transitioning from pixels with discrete boundaries to (continuous) RGB values where the boundaries of pixels are no longer visible as shown in the figure below. 

TODO: Add pixels figure



### Objectives

Given the background on super-resolution and INRs, the objectives of my project were as follows: 

1. Understand in-depth the concept of implicit representations of images and their potential applications 
2. Train a neural network (particuarly, a multi-layer perceptron) to reconstruct a high-resolution MRI of a spinal cord given two different low-resolution views as inputs.
3. Perform an ablation study on the neural network parameters to analyse its effect on the reconstruction accuracy. 


### Data


### Tools 


### Deliverables


### Results 


### Conclusions


### Limitations and Open Questions 


### References

1. Plenge, Esben et al. “Super-resolution methods in MRI: can they improve the trade-off between resolution, signal-to-noise ratio, and acquisition time?.” Magnetic resonance in medicine vol. 68,6 (2012): 1983-93. 
2. 
