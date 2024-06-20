# HalfedgeCNN

Version 1.0

Note: This code is based on a modification of the code from https://ranahanocka.github.io/MeshCNN/.


# Getting Started

### Install dependencies
1. Create a new Anaconda environment: 
```bash
conda create --name halfedgecnn
```
2. Switch to the new environment:
```bash
conda activate halfedgecnn    
```
3. Install Pytorch:
```bash
conda install pytorch torchvision torchaudio -c pytorch
```
4. Install tensorboardX for viewing the training plots (optionally):
```bash
conda install -c conda-forge tensorboardx
```

Depending on the system, the installation of additional packages might be necessary.

### 3D Shape Classification on SHREC
For starting the SHREC classification training, first download and unzip the dataset using the get_shrec_data.sh script:
```bash
bash scripts/get_shrec_data.sh
```
Now start the training, with the following command:
```bash
python scripts/train_with_settings.py shrec_16  
```
To view the training loss and accuracy plots, run ```tensorboard --logdir runs``` in another terminal and click [http://localhost:6006](http://localhost:6006).

After training the latest model can be tested using the following command:
```bash
python scripts/test_with_settings.py shrec_16  
```
The best found model can be tested using the following command:
```bash
python scripts/test_best_with_settings.py shrec_16
```

The resulting poolings can be exported with:
```bash
python scripts/export_results_as_obj.py shrec_16  
```
Examples of the resulting poolings can be viewed (after exporting) for example with the following command:
```bash
python util/mesh_viewer.py --files checkpoints/shrec_16/result_objs/T74_0.obj checkpoints/shrec_16/result_objs/T74_1.obj checkpoints/shrec_16/result_objs/T74_2.obj checkpoints/shrec_16/result_objs/T74_3.obj checkpoints/shrec_16/result_objs/T74_4.obj
```


### 3D Shape Segmentation on Humans
For starting the human segmentation training, first download and unzip the dataset using the get_human_data.sh script:
```bash
bash scripts/get_human_data.sh
```

Make sure that in the general settings (scripts/settings/general_settings.txt) the segmentation base is set to the right value, in our example edge based (--segmentation_base edge_based)

Then start the training with the following command:
```bash
python scripts/train_with_settings.py human_seg  
```
Again, to view the training loss and accuracy plots, run ```tensorboard --logdir runs``` in another terminal and click [http://localhost:6006](http://localhost:6006).

After training the latest model can be tested using the following command:
```bash
python scripts/test_with_settings.py human_seg  
```
The best found model can be tested using the following command:
```bash
python scripts/test_best_with_settings.py human_seg
```

The resulting poolings and segmentations can be exported with:
```bash
python scripts/export_results_as_obj.py human_seg
```

The resulting segmentation for one mesh can be viewed (after exporting) for example with the following command:
```bash
python util/mesh_viewer.py --files checkpoints/human_seg/result_objs/shrec__14_0.obj
```

Similar to the shrec classification and human segmentation task, other classification and segmentation tasks can be performed. 
For every dataset a settings file needs to be created in the scripts/settings directory.
In it it needs to be defined whether it is a classification or segmentation dataset, what the name of the dataset is, what the name of the results folder should be and what the highest number or faces occurring in the dataset is. 
In the case of a classification task, also the pooling resolutions needs to be given, because the settings differ between the classification datasets.
The dataset setting files can also be used to override the default settings for specific datasets. If a setting is given both in the default settings and in the dataset settings, the dataset setting is used.
For example, the settings might be as follows:

Classification:
```bash
--dataset_mode classification
--dataroot datasets/shrec_16
--name shrec_16
--pool_res 1200 900 600 360
--niter_decay 100
```
Segmentation:
```bash
--dataset_mode segmentation
--dataroot datasets/human_seg
--name human_seg
--number_input_faces 1520
```

### Evaluate Runs
To simplify the evaluation of the runs, one can use the evaluate_runs.py, remove_unfinished_runs.py and show_chart.py script.

The evaluate_runs scripts creates a list of the end-values and the best values for all recorded runs in the runs directory and computes the mean, the 
standard deviation and other interesting statistics for all runs.

The remove_unfinished_runs.py scripts removes all runs that have crashed or that did not reach the desired number of epochs.

The show_chart.py script shows the development of the all recorded runs, marking the first occurrence of the best value of each run. 

All scripts need the desired number of epochs as a parameter, for example 200 for the shrec classification task and 300 for the human segmentation tasks.
```bash
python scripts/evaluate_runs.py 200
python scripts/remove_unfinished_runs.py 200
python scripts/show_chart.py 200
```


### Change Neighborhood Size
The used convolution neighborhood can be selected by changing the number behind the --nbh_size parameter in scripts/settings/general_settings.txt.
The nomenclature used for the nbh_size is as described in figure 2 and the table in section 7 of the paper. 
Available are neighborhoods sizes 2, 3, 4, 5, 7, and 9.

### Change Feature Selection 
By changing the number behind the --feat_selection parameter, a different feature combination can be selected.
Following the nomenclature used in the paper, "0" means symmetrized features, "1" oriented features, and "2" fundamental features.

### Change Pooling
Two different pooling methods are available, Edge-Pooling and Half-Edge-Pooling. 
For Half-Edge-Pooling the --pooling parameter has to be set to "half_edge_pooling". 
For Edge-Pooling the parameter has to be set to "edge_pooling".

### Change Segmentation Base
Three different segmentation bases are available: edge based, half-edge based, and face based.
For edge based segmentation the --segmentation_base parameter has to be set to "edge_based", for halfedge based segmentation to "half_edge_based", and for face based segmentation to "face_based".
