## Right for the Wrong Reason
Official implementation for MICCAI 2023 paper: **Right for the Wrong Reason: Can Interpretable ML Techniques Detect Spurious Correlations?**
[Arxiv Paper](https://arxiv.org/abs/2307.12344)

## Overview of Experiment settings
Deep neural networks tend to learn spurious correlations instead of using class-relevant features. When deploying such models on the test set that has a different distribution, the classification performance drops significantly as shown in the figure below.

<div style="displaystyle=block;align=center;"><p align="center" >
  <img src="figs/classification.png"/ width="60%" height="60%">
  </p>
</div>


We performed a rigorous evaluation of post-hoc explanations and inherently interpretable techniques for the detection of spurious correlations in a medical imaging task.
* We designed three kinds of spurious signals named "Tag", "Hyperintensities" and "Obstruction" and contaminated the positive samples in the train set with ratios of 20%, 50%, 80% and 100%.
* We designed two novel metrics named "Confounder Sensitivity" and "Explanation NCC" to evaluate five post-hoc explanation methods and one inherently interpretable model in their ability to detect the spurious signal.

<div style="displaystyle=block;align=center;"><p align="center" >
  <img src="figs/overview.png"/ width="75%" height="75%">
  </p>
</div>



## Results
We evaluated the explanation methods and model both qualitatively and quantitatively.

<div style="displaystyle=block;align=center;"><p align="center" >
  <img src="figs/different_scale.png"/ width="60%" height="60%">
  </p>
</div>

<div style="displaystyle=block;align=center;"><p align="center" >
  <img src="figs/exp_results.png"/ width="60%" height="60%">
  </p>
</div>

## Installation

```
conda env create -f right_for_wrong.yml
conda activate right_for_wrong
```


## Datasets
We perform evaluations with the following three Chest X-ray datasets.
**CheXpert** (https://stanfordmlgroup.github.io/competitions/chexpert/)

# References
If you use any of the code in this repository for your research, please cite as:
```
 @misc{sun2023right,
      title={Right for the Wrong Reason: Can Interpretable ML Techniques Detect Spurious Correlations?}, 
      author={Susu Sun and Lisa M. Koch and Christian F. Baumgartner},
      year={2023},
      eprint={2307.12344},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
}
