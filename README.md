<h2 align="center">
  Active View Selection with Perturbed Gaussian Ensemble for Tomographic Reconstruction
</h2>

<p align="center">
  <a href="https://arxiv.org/abs/2603.06852" target='_blank'><img src="http://img.shields.io/badge/cs.CV-arXiv%3A2603.06852-b31b1b"></a>
</p>

## Overview

<div align="center"><img src="./assets/overview.jpg" width=60%></div>

We present *Perturbed Gaussian Ensemble*, a novel active view
selection framework for progressive reconstruction tailored specifically to X-ray
Gaussian Splatting. By applying stochastic density perturbations to
low-density primitives that are highly susceptible to geometric degradation
and measuring the structural disagreement in projection space, our method
accurately localizes epistemic uncertainty and predicts the next best view.

## Installation

```
# Download code
git clone https://github.com/yulunwu0108/Perturbed-Gaussian-Ensemble.git --recursive

# Install environment
cd Perturbed-Gaussian-Ensemble
conda env create --file environment.yml
conda activate perturbed-gaussian-ensemble
```

## Dataset

The data used in our experiments will be released soon. Please stay tuned!

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{wu2026active,
  title={Active View Selection with Perturbed Gaussian Ensemble for Tomographic Reconstruction},
  author={Wu, Yulun and Zha, Ruyi and Cao, Wei and Li, Yingying and Cai, Yuanhao and Liu, Yaoyao},
  journal={arXiv preprint arXiv:2603.06852},
  year={2026}
}
```

## Acknowledgement

This project has benefited from [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [R2-Gaussian](https://github.com/Ruyi-Zha/r2_gaussian), [FisherRF](https://github.com/JiangWenPL/FisherRF), and [DiffDRR](https://github.com/eigenvivek/DiffDRR). We sincerely thank the authors for their open-source contributions.