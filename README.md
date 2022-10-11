# Tsetlin Machine for solving contextual bandit problems.


This repository contains the code for the experiment presented in the paper 
> Raihan Seraj, Jivitesh Sharma, Ole-Christoffer Granmo, " [Tsetlin Machine for Solving Contextual Bandit Problems] ", 2020

### Installation 
To install the dependencies of the code in a virtual environment run the setup script.

    bash setup.sh
### Usage

Activate the virtual environment installed using the following command:

    source python-vms/cb_tm/bin/activate


To run the experiment using default configuration for `Iris` dataset use the following command

    python src/main.py --dataset_name Iris --default_configuration True

The program accepts the following command line arguments:

| Option                    | Description                                                                                                                                                            |
|---------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--dataset_name`          | The name of the dataset. Possible choices are `Iris`, `Breast_Cancer`, `Noisy_XOR`, `Simulated Article`, `Movielens`, `Adult`, `Statlog_Shuttle`, `MNist`, `Covertype` |
| `--default_configuration` | Whether to use the default configuration, the value should be `True` or `False`                                                                                        |
| `--max_bits`              | The maximum number of bits per feature to be used during binarization                                                                                                  |
| `--num_clauses`           | The number of clauses to be used for the Tsetlin Machine learner.                                                                                                      |
| `--T`                     | The `T` parameter for the Tsetlin Machine learner.                                                                                                                     |
| `--s`                     | The `s` parameter for the Tsetlin Machine learner.                                                                                                                     |
| `num_state_bits`          | The number of state bits to be used for the Tsetlin Machine learner.                                                                                                   |
| `--interpretability`      | Whether to perform interpretability for the Tsetlin Machine learner. If the value is set to `True`, boolean expressions for the decisions are saved in a `.txt`file.   |
| `--drop_clause_p`         | The probability of dropping a clause at random for the Tsetlin Machine learner.                                                                                        |
| `--num_runs`              | The number of independent runs of the experiment to be performed.                                                                                                      |
| `--num_rounds`            | The number of rounds or simulation steps for each run.                                                                                                                 |
| `--exploration_probs`     | The probability that a random action will be chosen for epsilon greedy algorithms.                                                                                     |
| `--use_cuda`              | Whether to use cuda for the TM learner.                                                                                                                                |
| `--results_dir`           | The directory name where the results should be stored. The default value is `Results/`                                                                                 |


#### Abalation for Binarization

To perform an abalation study for different levels of binarization use the following code

    python src/abalation_binarization.py

The script `abalation_binarization.py` can be edited to run the abalation study for different datasets and different binarization values. 





