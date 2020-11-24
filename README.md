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


## Results
\begin{table*}[htb]
%\setlength\extrarowheight{-0.5pt}
\centering
\resizebox{0.80\textwidth}{!}{%
%\renewcommand{\arraystretch}{0.92}
% \setlength{\tabcolsep}{5.5pt}
\begin{tabular}{l|l|c|c|c|c|c|c|c|c|c}
%\toprule
\parbox[t]{2mm}{\multirow{2}{*}{\rotatebox[origin=c]{90}{Data}}}&\multirow{2}{*}{\diagbox{Method}{Task}} &\multicolumn{3}{c|}{Merging-in} & \multicolumn{3}{c|}{Lane-keeping}& \multicolumn{3}{c}{Braking}\\
%\cline{3-11}
           &    &   CC$\uparrow$  &   s-AUC$\uparrow$ & KLD$\downarrow$  & CC$\uparrow$    &   s-AUC$\uparrow$ & KLD$\downarrow$  &   CC$\uparrow$  &s-AUC$\uparrow$ & KLD$\downarrow$\\\hline
\parbox[t]{2mm}{\multirow{7}{*}{\rotatebox[origin=c]{90}{DR(eye)VE~\cite{palazzi2018predicting}}}}
% &\textit{Ground truth}              &  \textit{1.00}   &\textit{ 0.95}  & \textit{0.01} & \textit{1.00}   &\textit{ 0.98} & \textit{0.00}   & \textit{1.00}  & \textit{0.99} & \textit{0.00}   \\

&Multi-branch~\cite{palazzi2018predicting} & 0.48 & 0.41  & 2.80 & 0.55 & 0.51 & 1.87 & 0.71 & 0.53 & 2.20 \\
&HWS~\cite{xia2018predicting}          & 0.51 & 0.48  & 2.12 & 0.75 & 0.49 & 1.72 & 0.74 & 0.57 & 1.99 \\

&SAM-ResNet~\cite{cornia2018predicting}    & \textbf{0.78} & 0.54  & 2.01 & 0.80 & 0.59 & 1.80 & 0.79 & 0.69 & 1.89 \\
&SAM-VGG~\cite{cornia2018predicting}      & \textbf{0.78} & 0.55  & 2.05 & 0.82 & 0.56 & 1.84 & 0.80 & 0.66 & 1.81 \\
&TASED-NET~\cite{min2019tased}      & 0.68 & 0.59  & 1.89 & 0.73 & 0.61  & 1.71 & 0.70 & 0.62  & 1.89 \\
&MEDIRL (ours) & \textbf{0.78} & \textbf{0.69}  &\textbf{0.88} & \textbf{0.89} & \textbf{0.67} & \textbf{0.75} & \textbf{0.85} &\textbf{ 0.63} & \textbf{0.82} \\ \hline %\cline{2-11}
%------------------
\parbox[t]{2mm}{\multirow{7}{*}{\rotatebox[origin=c]{90}{BDD-A~\cite{xia2018predicting}}}}
% &\textit{Ground truth}              &  \textit{1.00}   & \textit{0.97}  & \textit{0.01} & \textit{1.00}   &\textit{ 0.99} & \textit{0.00}   & \textit{1.00}  & \textit{0.98} & \textit{0.00}   \\

&Multi-branch~\cite{palazzi2018predicting} & 0.58 & 0.51  & 2.08 & 0.75 & 0.72 & 2.0  & 0.69 & 0.77 & 2.04 \\
&HWS~\cite{xia2018predicting}          & 0.53 & 0.59  & 1.95 & 0.67 & 0.89 & 1.52 & 0.69 & 0.81 & 1.59 \\
&SAM-ResNet~\cite{cornia2018predicting}   & 0.74 & 0.61  & 2.00 & 0.89 & 0.79 & 1.83 & 0.85 & 0.88 & 1.89 \\
&SAM-VGG~\cite{cornia2018predicting}      & 0.76 & 0.62  & 1.79 & 0.89 & 0.82 & 1.64 & 0.86 & 0.87 & 1.85 \\
&TASED-NET~\cite{min2019tased}      & 0.73 & 0.68  & 1.83 & 0.81 & 0.66  & 1.17 & 0.87 & 0.88  & 1.12 \\
%\hline
&MEDIRL (ours) & \textbf{0.82} &\textbf{ 0.79}  & \textbf{0.91} & \textbf{0.94} & \textbf{0.91} & \textbf{0.85} & \textbf{0.93} & \textbf{0.92} & \textbf{0.89 }\\ \hline %\cline{2-11}
%-------------------
\parbox[t]{2mm}{\multirow{7}{*}{\rotatebox[origin=c]{90}{DADA-2000~\cite{fang2019dada}}}}
% &\textit{Ground truth}             & \textit{1.00}   & \textit{0.94}  & \textit{0.02} & \textit{1.00}   & \textit{0.98} & \textit{0.00}   & \textit{1.00}   & \textit{0.99} &\textit{0.00}   \\

&Multi-branch~\cite{palazzi2018predicting} & 0.44 & 0.53  & 3.65 & 0.69 & 0.54 & 2.85 & 0.67 & 0.64 & 2.91 \\
&HWS~\cite{xia2018predicting}        & 0.49 & 0.59  & 3.02 & 0.72 & 0.53 & 2.65 & 0.69 & 0.77 & 2.80 \\
&SAM-ResNet~\cite{cornia2018predicting}   & 0.65 & 0.61  & 2.39 & 0.78 & 0.64 & 2.32 & 0.75 & 0.81 & 2.34 \\
&SAM-VGG~\cite{cornia2018predicting}    & 0.68 & 0.60  & 2.41 & 0.76 & 0.62 & 2.24 & 0.75 & 0.80 & 2.35 \\
&TASED-NET~\cite{min2019tased}      & 0.69 & 0.66  & 1.98 & 0.78 & 0.69  & 1.87 & 0.80 & 0.81  & 1.45 \\
&MEDIRL (ours) &\textbf{ 0.70} & \textbf{0.68}  & \textbf{1.31} & \textbf{0.89} & \textbf{0.71} & \textbf{0.92} & \textbf{0.81} & \textbf{0.88} &\textbf{ 0.99} \\ 
%\bottomrule
\end{tabular}
}
% \begin{tablenotes}
%       \small
%       \item *methods tested on EyeCar dataset
%     \end{tablenotes}
%\vspace{.3em}
\caption{Comparison of different visual attention allocation models trained on the BDD-A~\cite{xia2018predicting} train set using multiple evaluation metrics. We evaluate them with respect to Dr(eye)VE~\cite{palazzi2018predicting}, BDD-A~\cite{xia2018predicting}, and DADA-2000~\cite{fang2019dada} test sets.\vspace{-0.1in}\label{tbl:benchmarks}}
\end{table*}

%on the test set of something
%\textcolor{red}{Comparison with other visual attention allocation prediction models~(row) trained on the BDD-A~\cite{xia2018predicting} train dataset using multiple evaluation metrics~(columns) for different driving tasks of rear-end collisions and tested on EyeCar dataset.}

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
