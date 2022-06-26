# Project Title:
### COD-rNA Prediction
___
### Overview
This project was carried out with the aim to predict the state of sensorless machines based on information as regards their present state of operation.

### Stack
All stack elements are open-sourced:
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/downloads/release/python-360/)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)](https://www.jupyter.org)
[![PyCharm](https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green)](https://www.jetbrains.com/pycharm/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)](http://git-scm.com/)
[![saythanks](https://img.shields.io/badge/say-thanks-ff69b4.svg?&style=for-the-badge)](https://saythanks.io/to/kennethreitz)

### Features and Data Information
There are eight (8) features contained in the data, and they are as follows:
- Divide by 10 to get deltaG_total value computed by the Dynalign algorithm
- The length of shorter sequence
- 'A' frequencies of sequence 1
- 'U' frequencies of sequence 1
- 'C' frequencies of sequence 1
- 'A' frequencies of sequence 2
- 'U' frequencies of sequence 2
- 'C' frequencies of sequence 2

### Motivation


### Quick Start
The dataset for this project was of a file size beyond GitHub accommodation levels. As such, an abstraction is provided to read the dataset into memory and compress it for storage on the fly. The compressed data file is to be found in the `data/archive` directory. For use in the project, the compressed file is to be decompressed into the `data/dataset` directory.

To run the scripts, type as below in the Terminal:

1. Navigate to the `scripts` directory.
```
./ $ cd scripts
```
2. Next, run the `main.py` file with the following syntax:

    `py main --argument argument_value`

Example:

```
./scripts/ $ py main.py --n_jobs -1
```
Acceptable arguments include:
- n_jobs (default = -1)
- visualize (default = False)
- r_state (default = 42; random state)
- data_dir (data directory)
- arch_dir (compressed file directory)
- thresh (minimum limit for feature importance)
- train (create train split?)
- valid (create valid split?)
- test (create test split?)

Others may be found in the `main.py` script.

3. Generated diagnostics, text and images, will populate the `reports/text` and `reports/images` directories respectively.
4. Find trained model artefact in the `artefacts` directory.

### Training Procedure
Training was done via composite models, i.e. estimators and transformers chained via a Pipeline API in Scikit Learn. The dataset was plagued by class imbalance, and an oversampling technique was applied during modelling.

The Pipeline steps are:
- MinMax scaler
- Quantile transformer
- SMOTE Oversampler
- Learning algorithm

Two learning algorithms were attempted:
- Logistic Regression
- Support Vector Machines (SVMs)


### Performance Report
A considerable test performance improvement was observed, from `~ 89 %` for a `LogisticRegression` Pipeline to `~ 95 %` for an `SVM`. These performance metrics were obtained for the major classification metrics (accuracy, f1-score, recall, precision, and AUC score), via `macro` averaging.



### To-Dos
1. Flesh out this README.
2. Tighten up the `main.py` file.

### Appendix

Data Source:

Source: [Train](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna) [Test](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna.t)

### Authors and Citation

1. Andrew V Uzilov, Joshua M Keegan, and David H Mathews. Detection of non-coding RNAs on the basis of predicted secondary structure formation free energy change. BMC Bioinformatics, 7(173), 2006.


