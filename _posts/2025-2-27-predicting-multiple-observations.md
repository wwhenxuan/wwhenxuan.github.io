---
title: 'Predicting multiple observations in complex systems through low-dimensional embeddings'
date: 2025-02-27
category: nature
permalink: /posts/2025-2-27-predicting-multiple-observations
excerpt: "This article reduces the dimensionality of high-dimensional complex systems through manifold learning and delayed embedding methods, and realizes the modeling and prediction of the complex system in low-dimensional space.<br/><img src='../images/blogs/2025-2-27-predicting-multiple-observations/cover.png'>"
tags:
  - Nature Communications
  - Complex Systems Modeling
  - Machine Learning
paperurl: 'https://www.nature.com/articles/s41467-024-46598-w'
---

This article reduces the dimensionality of high-dimensional complex systems through manifold learning and delayed embedding methods, and realizes the modeling and prediction of the complex system in low-dimensional space.<br/><img src='../images/blogs/2025-2-27-predicting-multiple-observations/cover.png'>

## Abstract

> Received: 11 November 2023
>
> Accepted: 4 March 2024

Forecasting all components in complex systems is an open and challenging task, possibly due to high dimensionality and undesirable predictors. 
We bridge this gap by proposing a data-driven and model-free framework, namely, feature-and-reconstructed manifold mapping (FRMM), which is a combination of feature embedding and delay embedding. For a high-dimensional dynamical system, FRMM finds its topologically equivalent manifolds with low dimensions from feature embedding and delay embedding and then sets the low-dimensional feature manifold as a generalized predictor to achieve predictions of all components. The substantial potential of FRMM is shown for both representative models and real-world data involving Indian monsoon, electroencephalogram (EEG) signals, foreign exchange market, and traffic speed in Los Angeles Country. 
FRMM overcomes the curse of dimensionality and finds a generalized predictor, and thus has potential for applications in many other real-world systems.

> 核心观点：高维复杂系统中通常包含大量的冗余信息，其基本动力学规则或结构可以通过低维系统来表征。

## Introduction

1. 复杂系统的观测数据通常为高维时间序列，其中的变量数目过多，并且变量之间的相互作用过于复杂；
2. 在处理高维复杂系统时，为了应对变量之间复杂关系，降维是一种最为常用的方法；
3. 但在建模复杂系统时人们应该尽可能小心的去忽略其中的一些不重要的变量，因为可能



在研究这些复杂系统时人们应该尽可能小心的去忽略其中的一些不重要的变量。

可能这些变量发生一点小小的扰动就会导致复杂系统的表征发生很大的改变。

