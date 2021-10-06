# [MEDIRL: Predicting the Visual Attention of Drivers via Deep Inverse Reinforcement Learning](https://arxiv.org/pdf/1912.07773.pdf)
[Sonia Baee](http://soniabaee.com), [Erfan Pakdamanian](http://www.cs.virginia.edu/~ep2ca/),  [Inki Kim](http://www.sys.virginia.edu/inki-kim.html), [Lu Feng](https://www.cs.virginia.edu/~lufeng/), [Vicente Ordonez](http://vicenteordonez.com/), [Laura Barnes](https://faculty.virginia.edu/S2HeLab/index.php)


![ICCV Poster](https://github.com/soniabaee/MEDIRL-EyeCar/blob/master/iccv21_poster_7007_medirl.png)


**This repo is under construction and we will add the clean and most up to dated code shortly. Stay tuned!**
## To - Do
- [ ] Revise the code and add the missed functions
- [ ] Add the final dataset
- [ ] Change the format of the data to the .json file (EyeCar)
- [ ] Add the trained models



<!-- ![medirl](https://github.com/soniabaee/eyeCar/blob/master/figures/paper8.png) -->




## Overview
Inspired by human visual attention, we propose a novel inverse reinforcement learning formulation using Maximum Entropy Deep Inverse Reinforcement Learning (MEDIRL) for predicting the visual attention of drivers in accident-prone situations. MEDIRL predicts fixation locations that lead to maximal rewards by learning a task-sensitive reward function from eye fixation patterns recorded from attentive drivers. Additionally, we introduce EyeCar, a new driver attention dataset in accident-prone situations. We conduct comprehensive experiments to evaluate our proposed model on three common benchmarks: (DR(eye)VE, BDD-A, DADA-2000), and our EyeCar dataset. Results indicate that MEDIRL outperforms existing models for predicting attention and achieves state-of-the-art performance. We present extensive ablation studies to provide more insights into different features of our proposed model. The code and dataset are provided for reproducibility.

## Dataset - EyeCar 
We select 21 front-view videos that were captured in various traffic, weather, and day light conditions. Each video is 30sec in length and contains typical driving tasks (e.g., lane-keeping, merging-in, and braking) ending to rear-end collisions. Note that all the conditions were counterbalanced among all the participants. Moreover, EyeCar provides information about the speed and GPS of the ego-vehicle. In addition, each video frame comprises 4.6 vehicles on average, making EyeCar driving scenes more complex than other visual attention datasets. The [EyeCar dataset](https://github.com/soniabaee/eyeCar/tree/master/EyeCar) contains 3.5h of gaze behavior (aggregated and raw) from the 20 participants, as well as more than 315,000 rear-end collisions video frames. In EyeCar dataset, we account for the sequence of eye fixations, and thus we emphasize on attention shift to the salient regions in a complex driving scene. EyeCar also provides a rich set of annotations (e.g., scene tagging, object bounding, lane marking, etc.). You can accesss to the data in this [EyeCar](https://github.com/soniabaee/eyeCar/tree/master/EyeCar). Compared  to  prior  datasets,  EyeCar  is  the  only dataset  captured  from  a  point-of-view (POV)  perspective, involving collisions, and including metadata for both speedand  GPS.  EyeCar  also  has  the  largest  average  number  of vehicles per scene, and gaze data for 20 participants. 

![EyeCar-Dataset](https://github.com/soniabaee/eyeCar/blob/master/figures/eyeCar.png)

<!-- ## Modules of MEDIRL
![medirl-visual-module](https://github.com/soniabaee/eyeCar/blob/master/figures/visual.png)
1. **The visual module**: The visual module extracts low and mid-level visual cues that are useful for a variety of visual attention tasks. We rely on pre-existing models for semantic and instance segmentation, as well as depth estimation. In addition, we propose an approach to detect brake lights and traffic lights. 


![medirl-driving-module](https://github.com/soniabaee/eyeCar/blob/master/figures/driving.png)
2. **The driving module**: The driving module extracts driving-specific visual features for driving tasks. 


![medirl-attention-module](https://github.com/soniabaee/eyeCar/blob/master/figures/attention.png)
3. **The attention module**: Drivers pay attention to the task-related regions of the scene to filter out irrelevant information and ultimately make optimal decisions. Drivers do this with a sequence of eye fixations. To learn this process in various driving tasks ending in rear-end collisions, we cast it as a maximum inverse reinforcement learning approach.


## Some Results:
![medirl-result](https://github.com/soniabaee/eyeCar/blob/master/figures/results.png) -->

## Requirements

The simplest way to clone the development environment is with the 
`anaconda-project` and `conda-env` systems; these can be installed with [Anaconda](https://www.anaconda.com/) with the command

```bash
conda install anaconda anaconda-project
```

All necessary packages are found in `anaconda-project.yml`. To create an Anaconda virtual environment, run

```bash
rm environment.yml #if `environment.yml` already exists
anaconda-project export-env-spec -n medirl ./environment.yml
conda-env create -n medirl -f environment.yml
conda activate medirl
```

To deactivate the virtual environment, run

```bash
conda deactivate
```


To run a jupyter notebook inside this virtual environment, activate the
virtual env `medirl` and run

```bash
(medirl)$ python -m ipykernel install --user --name=medirl
```

where the `(medirl)` prefix indicates that you are currently in the 
`medirl` virtual environment. Then, select the "medirl" kernel 
after creating or opening a jupyter notebook.


## Model
You can run the main script to get the result of each component:

```bash
python main.py
```
We saved the best/last of models of each module in [Models](https://github.com/soniabaee/eyeCar/tree/master/Models) folder.

## Data
You can accesss to the data in this [EyeCar](https://github.com/soniabaee/eyeCar/tree/master/EyeCar)

## Citing
If you find this paper/code useful, please consider citing our paper:
```bash
  @InProceedings{Baee_2021_ICCV,
    author    = {Baee, Sonia and Pakdamanian, Erfan and Kim, Inki and Feng, Lu and Ordonez, Vicente and Barnes, Laura},
    title     = {MEDIRL: Predicting the Visual Attention of Drivers via Maximum Entropy Deep Inverse Reinforcement Learning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {13178-13188}
}
```



