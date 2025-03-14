# BatteryLife
This is the official repository for [BatteryLife: A Comprehensive Dataset and Benchmark for Battery Life Prediction](https://arxiv.org/abs/2502.18807)

## Highlights

- **The largest battery life dataset:** BatteryLife is created by integrating 16 datasets, providing more than ninety thousand samples from 998 batteries with life labels. This is 2.4 times the size of BatteryML, which is the previous largest battery life resource.
- **The most diverse battery life dataset:** BatteryLife contains 8 battery formats, 80 chemical systems, 12 operation temperatures, and 646 charge/discharge protocols. Compared with the previous largest battery life resource (BatteryML), BatteryLife furnishes 4 times formats, 16 times chemical systems, 2.4 times operating temperature, and 3.4 times charge/discharge protocols.
- **A comprehensive benchmark for battery life prediction:** BatteryLife provides 18 benchmark methods with open-source codes in this repository. The 18 benchmark methods include popular methods for battery life prediction, popular baselines in time series analysis, and a series of baselines proposed by this work.

## Data availability

The processed datasets can be accessed via multiple ways:
1. You can download the datasets from [Huggingface](https://huggingface.co/datasets/Hongwxx/BatteryLife_processed/tree/main) [[tutorial]](./assets/Data_download.md).
2. You can download the datasets from [Zenodo](https://zenodo.org/records/14969822).
   

Note that brief introductions to each dataset are available under the directory of each dataset.

All the raw datasets are publicly available, interested users can download them from the following links:
- Zn-ion, Na-ion, and CALB datasets: [Zenodo link](https://zenodo.org/records/15013636) [Huggingface link](https://huggingface.co/datasets/Hongwxx/BatteryLife_Raw/tree/main) [[tutorial]](./assets/Data_download.md#how-to-download-the-raw-data-from-huggingface)
- CALCE: [link](https://calce.umd.edu/battery-data)
- MATR: [Three batches](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204) and [Batch 9](https://data.matr.io/1/projects/5d80e633f405260001c0b60a/batches/5dcef1fe110002c7215b2c94)
- HUST: [link](https://data.mendeley.com/datasets/nsc7hnsg4s/2)
- RWTH: [link](https://publications.rwth-aachen.de/record/818642/files/Rawdata.zip)
- ISU\_ILCC: [link](https://iastate.figshare.com/articles/dataset/_b_ISU-ILCC_Battery_Aging_Dataset_b_/22582234)
- XJTU: [link](https://zenodo.org/records/10963339)
- Tongji: [link](https://zenodo.org/records/6405084)
- Stanford: [link](https://data.matr.io/8/)
- HNEI, SNL, MICH, MICH_EXP and UL_PUR datasets: [BatteryArchive](https://www.batteryarchive.org/index.html).

## Quick start

### Install

```
pip install -r requirements.txt
# You should also install BatteryML (https://github.com/microsoft/BatteryML)
```

### Preprocessing [[tutorial](./assets/Preprocess.md)]

After downloading all raw datasets provided in "Data availability" section, you can run the following script to obtain the processed datasets:

```
python preprocess_scripts.py
```
If you download the processed datasets, you can skip this step.

### Train the model [[tutorial](./assets/Model_training.md)]

Before you start training, please move all **processed datasets (such as, HUST, MATR, et al.)**, **Life labels folder** (downloaded from Hugginface or Zenodo websites), and **seen_unseen_labels** into `./dataset` path under the root folder.

After that, just feel free to run any benchmark method. For example:

```sh
sh ./train_eval_scripts/CPTransformer.sh
```

### Evaluate the model

If you want to evaluate a model in detail. We have provided the evaluation script. You can use it as follows:

```sh
sh ./train_eval_scripts/evaluate.sh
```

## Data Structure

The data structure of the standardized data is described in [Data_structure_description.md](./assets/Data_structure_description).

## Welcome contributions

To facilitate advances on battery life prediction, the community needs standardized datasets. However, the available battery life datasets are typically stored in different places and different formats. We have put great efforts in integrating 13 previously available datasets and 3 our datasets. We warmly welcome contributions from the community to further enhance this collection by submitting datasets standardized according to the BatteryLife standards. 

If you are interested in contributing, please either submit a pull request or contact us via email at rtan474@connect.hkust-gz.edu.cn and whong719@connect.hkust-gz.edu.cn. Kindly include a list of contributors in your pull request or email. We will acknowledge all contributors in the acknowledgement section of this repository.

## Citation
If you find this work useful, we would appreciate citations to the BatteryLife paper:

```
@misc{tan2025batterylifecomprehensivedatasetbenchmark,
      title={BatteryLife: A Comprehensive Dataset and Benchmark for Battery Life Prediction}, 
      author={Ruifeng Tan and Weixiang Hong and Jiayue Tang and Xibin Lu and Ruijun Ma and Xiang Zheng and Jia Li and Jiaqiang Huang and Tong-Yi Zhang},
      year={2025},
      eprint={2502.18807},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.18807}, 
}
```

Additionally, please cite the original papers that conducted experiments. Please cite [BatteryArchive](https://www.batteryarchive.org/index.html) as the data source for the HNEI, SNL, MICH, MICH_EXP, and UL_PUR datasets.

## Acknowledgement
This repo is constructed based on the following repos:
- https://github.com/thuml/Time-Series-Library
- https://github.com/microsoft/BatteryML

