# Sound Event Classification

This repository contains code for various sound event classification systems implemented on two datasets - (a) DCASE 2019 Task 5 and (b) a subset of Audioset.

## Getting Started
This repository uses the code in the [External-Attention-pytorch repository](https://github.com/xmu-xiaoma666/External-Attention-pytorch) as a submodule. 

Hence to only clone this repository:
```
git clone xx
```

To clone this repository and the sub-modules at the same time: 
```
git clone --recurse-submodules -j8 xx
```
`-j8` is an optional performance optimization that became available in version 2.8, and fetches up to 8 submodules at a time in parallel

Look at this [StackOverflow answer](https://stackoverflow.com/a/4438292) for more information.

## Organization
Each dataset has its own folder and contains the following subfolders:
1. augmentation: package which contains code for various data augmentations
2. data: contains raw audio files and generated features
3. split: contains metadata required to parse the dataset
4. models: contains saved weights
Other than these subfolders, each dataset folder contains a script `statistics.py` to evaluate the channel wise mean and standard deviation for the various features. 

## Datasets
### A. DCASE
This is the Urban Sound Tagging dataset from DCASE 2019 Task 5. It contains one training split and one validation split. 
### B. Audioset
This is a subset of Audioset containing 10 classes. It is split into 5 different folds for 5-fold cross validation. 

## Reproducing the results
To reproduce the results, first clone this repository. Then, follow the steps below. 
### 1. Generating the features
Generate the required type of feature using the following <br/>
`python compute_<feature_type>.py <input_path> <output_path>`<br/>
Replace `<feature_type>` with one of `logmel`, `cqt`, `gammatone`. The output path has to be `./<dataset>/data/<feature_type>` where `<dataset>` is one of `dcase` or `audioset`. 

Command line example for generating logmelspec for an example dataset
`python compute_logmel.py "/dataset/train_labels" "./sound-event-classification/audioset/data/logmelspec/audio_44.1k/train"` <br/>
### 2. Evaluating channel wise mean and standard deviation
Evaluate the channel wise mean and standard deviation for the features using `statistics.py`. 
#### DCASE
Only two arguments are required: feature type and number of frequency bins. <br/>
`python statistics.py -w $WORKSPACE -f logmelspec -n 128` <br/>
#### Audioset
An additional parameter is required to specify the training, validation and testing folds. For training on folds 0, 1 and 2, validating on 3 and testing on 4, run <br/>
`python statistics.py -w $WORKSPACE -f logmelspec -n 128 -p 0 1 2 3 4` <br/>
### 3. Parse the data
To parse the DCASE data, run `parse_dcase.py`. No such step is required for Audioset.
### 4. Training
#### DCASE
The `train.py` file for DCASE takes in 3 arguments: feature type, number of time frames and random seed. To train logmel, run <br/>
`python train.py -w $WORKSPACE -f logmelspec -n 636 --seed 42 <br/>
#### Audioset
Other than the three arguments above, the `train.py` file for Audioset takes in an additional argument to specify the training, validation and testing folds. For training on folds 0, 1 and 2, validating on 3 and testing on 4, run <br/>
An experiment name is also required
`python train.py -w $WORKSPACE -f logmelspec -n 636 -p 0 1 2 3 4 --expt_name mobilenetv2` <br/>
Different flags you can use with `python train.py` are:
- `-en`, `--expt_name`: Specifies an experiment name (This is a REQUIRED parameter)
- `-w`, `--workspace`: To provide the path of the active workspace.
- `-f`, `--feature_type`: Default = `logmelspec`. To provide which model feature to use for training.
- `-ma`, `--model_arch`: Which model architecture to use when training. Default = `mobilenetv2`. Options: `[mobilenetv2, pann_cnn10, pann_cnn14]`
- `-cp10`, `--pann_cnn10_encoder_ckpt_path`: Path of the pretrained model weights for PANN architecture. Only required when `model_arch=pann_cnn10`
- `-cp14`, `--pann_cnn14_encoder_ckpt_path`: Path of the pretrained model weights for PANN architecture. Only required when `model_arch=pann_cnn14`
- `-n`, `--num_frames`:Number of frames in generated logmelspecs.
- `-p`, `--permutation`: Specifies the data splits used for training and validation. Default `p = 0 1 2 3 4`
- `-s`, `--seed`: Int seed value to set for torch and numpy. Helpful in replicating results.
- `-rt`, `--resume_training`: Whether to resume training from the last epoch it stopped at. Will automatically load the previous model weights. Default `'yes'`
- `-bs`, `--balanced_sampler`: Whether to use a balanced sampler when loading data. Default = `False`
- `-ga`, `--grad_acc_steps`: Number of epochs to perform gradient accumulation on. Default = `2`
- `-abs`, `--auto_batch_size`: Determines whether batch size will be automatically determined. Default = `2`
### 5. Validating
For validation, run `evaluate.py` with the same arguments as above but without the random seed argument.
### 6. Feature Fusion
In order to perform feature fusion, refer to section 1 to generate  `logmelspec`, `cqt` and  `gammatone` features and then train their respective models. Next, to generate the weights of each feature, run <br/>
`python generate_weights.py -w $WORKSPACE -p 0 1 2 3 4` <br/>

Finally, run <br/>
`python feature_fusion.py -w $WORKSPACE -p 0 1 2 3 4` <br/>

### 7. Audioset subset results
![image](https://user-images.githubusercontent.com/25906470/145518047-e6762918-b56c-4ba2-8ed6-56dae87b0cf8.png)


### 8. Dataset Utilities
Example of how to generate distribution pie charts based on a folder of .wav files <br/>
`python dataset_util.py "path_to_dataset/train_labels/" -a 'chart' -t 'Train set distribution (70%)'`<br/>
`python dataset_util.py "path_to_dataset/test_labels/" -a 'chart' -t 'Test set distribution (15%)'`<br/>
`python dataset_util.py "path_to_dataset/val_labels/" -a 'chart' -t 'Valid set distribution (15%)'`<br/>


Data augmentation using audiomentations can be obtained by utilizing the augment action<br/>
`python dataset_util.py "path_to_dataset/train_labels/" -a 'augment' --output_path='/notebooks/feb/augmented/''`