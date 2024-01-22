# Enhancing Geometric Modeling in Convolutional Neural Networks: Limit Deformable Convolution


## Abstract

Traditional convolutional neural networks are limited in their ability to model geometric transformations due to their fixed geometric structure. To overcome this problem, researchers have introduced deformable convolution, which allows the convolution kernel to be deformable on the feature map. However, deformable convolution may introduce irrelevant contextual information during the learning process and thus affect the model performance.DCNv2 introduces a modulation mechanism to control the diffusion of the sampling points to control the degree of contribution of offsets through weights, but we find that such problems still exist in practical use. Therefore, we propose a new constrained deformable convolution to address this problem, which enhances the modeling ability by adding adaptive limiting units to constrain the offsets, and adjusts the weight constraints on the offsets to enhance the image focusing ability. In the subsequent work, we performed a lightweight work on the restriction deformable convolution and designed three kinds of LDBottleneck to adapt to different scenarios. Our restricted deformable network equipped with the optimal LDBottleneck improves mAP75 by 1.4\% compared to DCNv1 and 1.1\% compared to DCNv2 on the VOC2012+2007 dataset. On the CoCo2017 dataset, different Backbones equipped with our restricted deformable module all achieve good results.

## Abstract

## 安装

提供安装所需的步骤，包括依赖项的安装。

```bash
npm install
