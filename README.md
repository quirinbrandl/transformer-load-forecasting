# Training Size Matters: Impact of Training Data Size on Electrical Load Forecasting


This repository contains the source code used to run the experiments described in [the research paper](https://dl.acm.org/doi/10.1145/3679240.3734635), 
presented at [the 6th International Workshop on Energy Data and Analytics (EDA)](https://nfdi4energy.uol.de/sites/eda_workshop/)
as part of the 16th ACM International Conference on Future and Sustainable Energy Systems ACM e-Energy 2025 conference
on June 17th, 2025 in Rotterdam, Netherlands.

### BibTeX Entry
```
@inproceedings{gagin2025a,
    author = {Gagin, Stepan and Tekles, Alexander and de Meer, Hermann},
    title = {Training Size Matters: Impact of Training Data Size on Electrical Load Forecasting},
    year = {2025},
    doi = {10.1145/3679240.3734635},
    isbn = {9798400711251},
    booktitle = {Proceedings of the 16th ACM International Conference on Future and Sustainable Energy Systems},
    publisher = {Association for Computing Machinery},
    location = {Rotterdam, Netherlands},
    pages = {677–686},
    numpages = {10},
    series = {e-Energy '25}
}
```

### Paper Abstract
Electrical load forecasting is an essential component of smart grid
systems. The availability of historical data about the energy load
of buildings can be restricted, for instance, because of the recently
installed metering system. On the other hand, training the models
with large datasets requires a lot of computation operations, thus
a long execution time. The limited size of data and the desire for
fast execution pose challenges for the training of forecasting models.
In this paper, the six commonly used model architectures for
short-term electrical load forecasting are compared based on their
performance and execution time. In the experiments, different sizes
of training datasets are considered. The results demonstrate that
there is no universal model that achieves both the fastest execution
time and the lowest error level. With a limited training dataset,
autoregressive integrated moving average demonstrates the lowest
errors. Random forest and support vector regression demonstrate
higher error metrics with small training data sizes, however, faster
execution times than autoregressive integrated moving average.
If a large training dataset is available, the random forest demonstrates
a good balance between performance and execution time in
comparison to other models considered in the experiments.

---
## Getting Started

1. The project includes an environment file `environment.yml` that can be used to create [conda](https://github.com/conda/conda) environment:
```commandline
conda env create -f environment.yml
```
2. By default, the data from [The Building Data Genome 2 Data-Set](https://github.com/buds-lab/building-data-genome-project-2/tree/master/data) should be placed under `data/building-data-genome-project-2-master`.
The path to the dataset files can be changed in `src/config.yml`.
3. The parameters of the experiments can be set up in the config file `src/config.yml`.
By default, the file with the name `config.yml` located in the directory of a Python script is used.
The custom path to the config path can be passed as argument to a Python script. 

---
## Project Structure

- `data` - directory for data used for the experiments (and also preprocessed data);
- `output` - directory where the results of the experiments and hyperparameter tuning are stored;
- `src` - code for the experiments:
  - `config.yml` - configuration file with experiment parameters
  - `preprocess_data.py` -  preprocesses and transforms Building Data Genome 2 dataset files to the format used in the project files;
  - `tune_hyperparameters.py` - performs hyperparameter tuning phase;
  - `train_predict.py` - trains the models with hyperparameters provided in the `config.yml`.
The models are then used to make the predictions of energy load.
  - `evaluation.ipynb` - notebook for the analysis of the experimental results.
