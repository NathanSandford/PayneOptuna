# PayneOptuna
The Payne wrapped in Pytorch Lightning w/ Optuna integration

## Installation
### 1. Clone this GitHub Repository <br>
```bash
git clone git@github.com:NathanSandford/PayneOptuna.git
```
   
### 2. Create and activate Conda environment <br>
```bash
cd PayneOptuna
conda env create -f environment.yml 
conda activate payne_optuna
```

**NOTE:**
Some of the packages in environment.yml are not strictly required
but are anticipated to be useful (i.e., scipy, matplotlib, etc.)

###3. Install payne_optuna <br>
```bash
python setup.py develop
```