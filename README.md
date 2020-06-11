# Learning the Visual Attention of Drivers via Deep Inverse Reinforcement Learning

## Overview
In this work, we introduce a Maximum Entropy Deep Inverse Reinforcement Learning (MEDIRL) framework for modeling the visual attention allocation of drivers in imminent rear-end collisions. Our goal is to learn the policies that _attentive drivers_ use for allocating visual attention to salient regions within their field of view, and use these learned policies to potentially flag _inattentive drivers_. We also introduce EyeCar, a new dataset comprising more than 315,000 video frames of rear-end collisions in distinct environments along with eye-tracking data from human subjects. We show that MEDIRL successfully learns task sensitive reward functions from multimodal features including front-view videos, vehicle motion patterns, and semantic and instance segmentations. Additionally, MEDIRL establishes a new state-of-the-art accuracy for visual attention prediction on the following large-scale driving attention benchmark datasets: BDD-A, DADA-2000, and DR(eye)VE.


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


## Data
You can accesss to the data in this [folder](https://drive.google.com/drive/folders/1G-3t3T8QLLeO6fdwetbF7AnjliQA6uDm?usp=sharing)




### the eventl list of this dataset:
- event ID  event type
- 2934487   crash
- 5592471   crash
- 5996103   crash
- 9886399   crash
- 9886402   crash
- 10528128  crash
- 10528254  crash
- 10814075  crash
- 10814077  crash
- 15396983  crash
- 15396984  crash
- 16992777  crash
- 17726433  crash
- 22484772  crash
- 22485631  crash
- 23340980  crash
- 23362586  crash
- 23671177  crash
- 23675224  crash
- 24523230  crash
- 128888417 crash
- 26508566  Baseline
- 116154578 Baseline
- 128905745 Baseline
- 132361827 Baseline
- 132361987 Baseline
- 151089859 Baseline
- 151089962 Baseline
- 151090080 Baseline


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
