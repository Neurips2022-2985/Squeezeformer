# Squeezeformer:  An Efficient Transformer for Automatic Speech Recognition
![Screenshot from 2022-05-22 15-23-07](https://user-images.githubusercontent.com/50283958/169718508-fa7fd22f-9038-44f8-9e8e-bc9c6afff124.png)

We provide testing codes for Squeezeformer, along with the pre-trained checkpoints.

## Install Squeezeformer

We recommend using Python version 3.8.  

### 1. Install dependancies

We support Tensorflow version of 2.5. Run the following commands depending on your target device type.

* Running on CPUs: `pip install -e '.[tf2.5]'`
* Running on GPUs: `pip install -e '.[tf2.5-gpu]'`

### 2. Install CTC decoder 
```bash
cd scripts
bash install_ctc_decoders.sh
```

## Prepare Dataset

### 1. Download Librispeech

[Librispeech](https://ieeexplore.ieee.org/document/7178964) is a widely-used ASR benchmark that consists of 960hr speech corpus with text transcriptions.
The dataset consists of 3 training sets (`train-clean-100`, `train-clean-360`, `train-other-500`), 
2 development sets (`dev-clean`, `dev-other`), and 2 test sets (`test-clean`, `test-other`).

Download the datasets from this [link](http://www.openslr.org/12) and untar them.
If this is for testing purposes only, you can skip the training datasets to save disk space.
You should have flac files under `{dataset_path}/LibriSpeech`.

### 2. Create Manifest Files

Once you download the datasets, you should create a manifest file that links the file path to the audio input and its transcription.
We use a script from [TensorFlowASR](https://github.com/TensorSpeech/TensorFlowASR).

```bash
cd scripts
python create_librispeech_trans_all.py --data {dataset_path}/LibriSpeech --output {tsv_dir}
```
* The `dataset_path` is the directory that you untarred the datasets in the previous step.
* This script creates tsv files under `tsv_dir` that list the audio file path, duration, and the transcription.
* To skip processing the training datasets, use an additional argument `--mode test-only`.

If you have followed the instruction correctly, you should have the following files under `tsv_dir`.
* `dev_clean.tsv`, `dev_other.tsv`, `test_clean.tsv`, `test_other.tsv`
* `train_clean_100.tsv`, `train_clean_360.tsv`, `train_other_500.tsv` (if not `--mode test-only`)
* `train_other.tsv` that merges all training tsv files into one (if not `--mode test-only`)


## Testing Squeezeformer

### 1. Download Pre-trained Checkpoints

We provide pre-trained checkpoints for all variants of Squeezeformer.

|      **Model**      |                                                  **Checkpoint**                            | **test-clean** | **test-other** |
| :-----------------: | :---------------------------------------------------------------------------------------:  | :------------: | :------------: |
|  Squeezeformer-XS   | [link](https://drive.google.com/file/d/1qSukKHz2ltBiWU-xHGmI-P9ziPJcLcSu/view?usp=sharing) |    3.74        |      9.09      |
|  Squeezeformer-S    | [link](https://drive.google.com/file/d/1PGao0AOe5aQXc-9eh2RDQZnZ4UcefcHB/view?usp=sharing) |    3.08        |      7.47      |
|  Squeezeformer-SM   | [link](https://drive.google.com/file/d/17cL1p0KJgT-EBu_-bg3bF7-Uh-pnf-8k/view?usp=sharing) |    2.79        |      6.89      |
|  Squeezeformer-M    | [link](https://drive.google.com/file/d/1fbaby-nOxHAGH0GqLoA0DIjFDPaOBl1d/view?usp=sharing) |    2.56        |      6.50      |
|  Squeezeformer-ML   | [link](https://drive.google.com/file/d/1-ZPtJjJUHrcbhPp03KioadenBtKpp-km/view?usp=sharing) |    2.61        |      6.05      |
|  Squeezeformer-L    | [link](https://drive.google.com/file/d/1LJua7A4ZMoZFi2cirf9AnYEl51pmC-m5/view?usp=sharing) |    2.47        |      5.97      |


### 2. Run Inference!

Run the following commands:
```bash
cd examples/squeezeformer
python test.py --bs {batch_size} --config configs/squeezeformer-S.yml --saved squeezeformer-S.h5 \
    --dataset_path {tsv_dir} --dataset {dev_clean|dev_other|test_clean|test_other}
```

* `tsv_dir` is the directory path to the tsv manifest files that you created in the previous step.
* You can test on other Squeezeformer models by changing `--config` and `--saved`, e.g., Squeezeformer-L or Squeezeformer-M.
