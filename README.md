# Out-of-Domain Robustness via Targeted Augmentations
Code for the paper [Out-of-Domain Robustness via Targeted Augmentations]() by Irena Gao*, Shiori Sagawa*, Pang Wei Koh, Tatsunori Hashimoto, and Percy Liang.

<small>Repository originally forked from [WILDS](https://github.com/p-lambda/wilds).</small>

## Abstract

> Models trained on one set of domains often suffer performance drops on unseen domains, e.g., when wildlife monitoring models are deployed on new camera locations. In this work, we study principles for designing data augmentations for out-of-domain (OOD) generalization. In particular, we focus on real-world scenarios in which some domain-dependent features are robust, i.e., some features that vary across domains are predictive OOD. For example, in the wildlife monitoring application above, image backgrounds vary across  camera locations but indicate habitat type, which helps predict the species of photographed animals. Motivated by theoretical analysis on a linear setting, we propose targeted augmentations, which selectively randomize spurious domain-dependent features while preserving robust ones. We prove that targeted augmentations improve OOD performance, allowing models to generalize better with fewer domains. In contrast, existing approaches such as generic augmentations, which fail to randomize domain-dependent features, and domain-invariant augmentations, which randomize all domain-dependent features, both perform poorly OOD. In experiments on three realworld datasets, we show that targeted augmentations set new states-of-the-art for OOD performance by 3.2–15.2%.

## Code
To install dependencies, run 
```
pip install -r requirements.txt
```

The repository supports running

* Five algorithms: Empirical Risk Minimization, [DeepCORAL](https://arxiv.org/abs/1607.01719), [IRM](https://arxiv.org/abs/1907.02893), [DANN](https://arxiv.org/abs/1505.07818), [CDAN](https://arxiv.org/abs/1705.10667)
* Three datasets: [iWildCam2020-WILDS](https://wilds.stanford.edu/datasets/#iwildcam), [Camelyon17-WILDS](https://wilds.stanford.edu/datasets/#camelyon17), and BirdCalls, a dataset we curate from ornithology data
* Several data augmentations, including [RandAugment](https://arxiv.org/abs/1909.13719), [CutMix](https://arxiv.org/abs/1905.04899), [MixUp](https://arxiv.org/abs/1710.09412), [Cutout](https://arxiv.org/abs/1708.04552), [LISA](https://arxiv.org/abs/2201.00299), [SpecAugment](https://arxiv.org/abs/1904.08779), random low / high pass filters, and [noise reduction via spectral gating](https://github.com/timsainb/noisereduce).

### Training with targeted augmentations

We also provide implementations for the three targeted data augmentations introduced in the paper.

1. **Copy-Paste (Same Y) for iWildCam2020-WILDS.** In iWildCam, image backgrounds are domain-dependent features with both spurious and robust components. While low-level background features are spurious, habitat features are robust. Copy-Paste (Same Y) transforms input $(x, y)$ by pasting the animal foreground onto a random training set background---but only onto backgrounds from training cameras that also observe $y$. This randomizes low-level background features while roughly preserving habitat.

```bash
python examples/run_expt.py --root_dir path/to/data --lr 3.490455181206744e-05 --transform_p 0.5682688104816859 --train_additional_transforms copypaste_same_y --algorithm ERM --dataset iwildcam --download
```

2. **Stain Color Jitter for Camelyon17-WILDS.** In Camelyon17, stain color is a spurious domain-dependent feature, while stage-related features are robust domain-dependent features. Stain Color Jitter ([Tellez et al., 2018](https://pubmed.ncbi.nlm.nih.gov/29994086/)) transforms $x$ by jittering its color in the hematoxylin and eosin staining color space.

```bash
python examples/run_expt.py --root_dir path/to/data --lr 0.0030693212138627936 --transform_p 0.5682688104816859 --train_additional_transforms camelyon_color --transform_kwargs sigma=0.1 --algorithm ERM --dataset camelyon17 --download
```

3. **Copy-Paste + Jitter (Region) for BirdCalls.** In BirdCalls, low-level noise and gain levels are spurious domain-dependent features, while habitat-specific noise is a robust domain-dependent feature. Copy-Paste + Jitter (Region) leverages time-frequency bounding boxes to paste bird calls onto other training set recordings from the same geographic region (Southwestern Amazon Basin, Hawaii, or Northeastern United States). After pasting the bird call, we also jitter hue levels of the spectrogram to simulate randomizing microphone gain settings.

```bash
python examples/run_expt.py --root_dir path/to/data --lr 0.00044964663762800047 --transform_p 0.5983713912982213 --train_additional_transforms copypaste_same_region --algorithm ERM --dataset birdcalls --download
```

## Citation
