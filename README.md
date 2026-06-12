# Physically Guided Symbolic Regression for Nuclear Binding Energy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center"><img src="results/figures/f2_BE_vs_SR_Z37AB.pdf" alt="SR on binding energy" width="80%" height="auto"/></p>

---

## What is this repository?

This repository contains the code and analysis for the project _Physically Guided Symbolic Regression for Nuclear Binding Energy_. This study aims to discover new models for the binding energy per nucleon using Symbolic Machine Learning (SML). We suggest two methods to the task, of which the reidual method in particular surpasses the accuracy of the Semiempirical Mass Formula (SEMF).

<!-- **You can read the full report [here](https://hdl.handle.net/2445/222442)** -->

Repository structure:

```bash
.
├── assets/                                   # Images, GIF
├── .gitignore
├── trainA.py
├── trainA__2.py
├── trainC__2.py      
├── scr                                       
    f1_BE.ipynb
    f1_BE_SR.ipynb
    f2_BE_SR.ipynb
├── results                                   
├── data                                      
├── CITATION.cff                              # Citation file
├── LICENSE                                   # License
├── README.md
├── env.yml                                   # Conda environment
└── requirements.txt                          # Requirements
```

---

## Installation
To install this project in your computer, choose one of the following options:

### With conda        
1. Clone the repository:

`git clone https://github.com/javier-rozalen/ml-tools-for-qm.git && cd ml-tools-for-qm`

2. If ```conda``` is not installed in your system, you can download it from https://docs.conda.io/en/latest/miniconda.html. 
3. Create a conda environment from the ```.yml``` file in the repository: 

`conda env create -f env.yml`

4. Activate the environment: 

`conda activate pysr-tools`

5. Install further requirements:

`pip install -r requirements.txt`

## Usage
There are three five files: 
- For code:
* trainA.ipynb
* trainA__2.ipynb
* trainC__2.ipynb
- For analysis:
* f1_BE.ipynb
* f1_BE_SR.ipynb
* f2_BE_SR.ipynb

f2_BE_SR.ipynb must output, among other figures, the folowing:

<p align="center"><img src="results/f2_Isotope_Map_HoFB.gif" width="100%" height="auto" /></p>

## Uninstall
To remove the virtual environment created with Option 1 follow the steps below:

1. Make sure your current environment is not `pysr-tools`, or if it is, type:

`conda deactivate`

2. Remove the environment.

`conda remove -n pysr-tools --all`

3. Remove the local repository.

Windows: `rmdir /S pysr-tools`

Linux/MacOS: `rm -r pysr-tools`

## Support

For questions or suggestions, feel free to reach out:
**nmlaau@gmail.com**  
---
