# MEDIRL: Predicting the Visual Attention of Drivers via Deep Inverse Reinforcement Learning
![medirl](https://github.com/soniabaee/eyeCar/blob/master/figures/figure1.png)

## Overview
Inspired by human visual attention, we introduce a Maximum Entropy Deep Inverse Reinforcement Learning~(MEDIRL) framework for modeling the visual attention allocation of drivers in imminent rear-end collisions. MEDIRL is composed of visual, driving, and attention modules. Given a front-view driving video and corresponding eye fixations from humans, the visual and driving modules extract generic and driving-specific visual features, respectively. Finally, the attention module learns the intrinsic task-sensitive reward functions induced by eye fixation policies recorded from attentive drivers. MEDIRL uses the learned policies to predict visual attention allocation of drivers. We also introduce EyeCar, a new driver visual attention dataset during accident-prone situations. We conduct comprehensive experiments and show that MEDIRL outperforms previous state-of-the-art methods on driving task-related visual attention allocation on the following large-scale driving attention benchmark datasets: DR(eye)VE, BDD-A, and DADA-2000. The code and dataset are provided for reproducibility.

## Modules of MEDIRL
![medirl-visual-module](https://github.com/soniabaee/eyeCar/blob/master/figures/visual.png)
1. The visual module: The visual module extracts low and mid-level visual cues that are useful for a variety of visual attention tasks. We rely on pre-existing models for semantic and instance segmentation, as well as depth estimation. In addition, we propose an approach to detect brake lights and traffic lights. Figure~\ref{medirl-visual-module} displays an overview of these individual components.


![medirl-driving-module](https://github.com/soniabaee/eyeCar/blob/master/figures/driving.png)
2. The driving module: The driving module extracts driving-specific visual features for driving tasks. Overview of the driving module is shown in Figure~\ref{medirl-driving-module}.


![medirl-attention-module](https://github.com/soniabaee/eyeCar/blob/master/figures/attention.png)
3. Drivers pay attention to the task-related regions of the scene to filter out irrelevant information and ultimately make optimal decisions. Drivers do this with a sequence of eye fixations. To learn this process in various driving tasks ending in rear-end collisions, we cast it as a maximum inverse reinforcement learning approach. Figure~\ref{medirl-attention-module} depicts an illustration of the approach.



## Requirements

The simplest way to clone the development environmen is with the 
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

## License

This project is licensed under the [MIT license](https://opensource.org/licenses/MIT):


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
