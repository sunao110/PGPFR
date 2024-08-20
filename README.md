# PGPFR
This repository is the official implementation of the ACM MM 2024 submission "Data-Free Class-Incremental Gesture Recognition with Prototype-Guided Pseudo Feature Replay".

# Abstract
Gesture recognition is an important research area in the field of computer vision. Most gesture recognition efforts focus on close-set scenarios, thereby limiting the capacity to effectively handle unseen or novel gestures. We aim to address class-incremental gesture recognition, which entails the ability to accommodate new and previously unseen gestures over time. Specifically, we introduce a Prototype-Guided Pseudo Feature Replay (PGPFR) framework for data-free class-incremental gesture recognition. This framework comprises four components: Prototype-Guided Pseudo Feature Generation (PGPFG), Variational Prototype Replay (VPR) for old classes, Truncated Cross-Entropy (TCE) for new classes, and Continual Classifier Re-Training (CCRT). To tackle the issue of catastrophic forgetting, the PFGBP dynamically generates a diversity of pseudo features in an online manner, leveraging class prototypes of old classes along with batch class prototypes of new classes. Furthermore, the VPR enforces consistency between the classifier’s weights and the prototypes of old classes, leveraging class prototypes and covariance matrices to enhance robustness and generalization capabilities. The TCE mitigates the impact of domain differences of the classifier caused by pseudo features. Finally, the CCRT training strategy is designed to prevent overfitting to new classes and ensure the stability of features extracted from old classes. Extensive experiments conducted on two widely used gesture recognition datasets, namely SHREC 2017 3D and EgoGesture 3D, demonstrate that our approach outperforms existing state-of-the-art methods by 11.8% and 12.8% in terms of mean global accuracy, respectively.



# Datasets

* [EgoGesture3D](https://drive.google.com/file/d/1pHE0Q9MtVS5BLaV2CBN1rLP_Ed7nvfac/view?usp=drive_link): The license for the EgoGesture3D skeletons are the same as this repository (license.txt). Please refer to the [EgoGesture](https://ieeexplore.ieee.org/document/8299578) paper and the website for the original video dataset and corresponding license.
* [SHREC-2017 train/val/test splits](https://drive.google.com/file/d/1o5T1b_jUG-czGp-xsGOFaVgzEJNEMnmh/view?usp=drive_link): This zip file only contains the split files comprising the list of files. Please refer to the [SHREC 2017 website](http://www-rech.telecom-lille.fr/shrec2017-hand/) to download the dataset.
  
# Training
