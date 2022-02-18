# Darknet traffic classification and characterization

Darknet is a set of networks and technologies, having as fundamentalprinciples anonymity and security. In many cases, they are associated with illicit activities, opening space for malware traffic and attacks to legitimate services.To prevent Darknet misuse is necessary to classify and characterize its existing traffic. In this paper, we characterize and classify the real Darknet traffic available from the _CIC-Darknet2020_ dataset. Therefore, we performed the feature extraction and grouped the possible subnets with an n-gram approach. Furthermore, we evaluated the relevance of the best features selected by the Recursive Feature Elimination method for the problem.  Our results indicate that simple models, like Decision Trees and Random Forests, reach an accuracy above 99% on traffic classification, representing a gain up to 13% in comparison with the state-of-the-art.

The CIC-Darknet2020 dataset can be found at [this](https://www.unb.ca/cic/datasets/darknet2020.html) link.

## Notebooks

All the notebooks have the preffix Darknet, so it'll be ommited on the descriptions.

- [preprocessing](https://github.com/mateus558/Darknet-traffic-classification/blob/main/Darknet%20-%20preprocessing.ipynb): contains all the preprocessing made on the dataset;
- [analysis](https://github.com/mateus558/Darknet-traffic-classification/blob/main/Darknet%20-%20analysis.ipynb): plots and analysis of traffic data;
- [detection models](https://github.com/mateus558/Darknet-traffic-classification/blob/main/Darknet%20-%20detection%20models.ipynb): training and validation of tree based models for darknet traffic detection;
- [characterization models](https://github.com/mateus558/Darknet-traffic-classification/blob/main/Darknet%20-%20characterization%20models.ipynb): training and validation of tree based models for darknet traffic characterization;
- [feature selection](https://github.com/mateus558/Darknet-traffic-classification/blob/main/Darknet%20-%20feature%20selection.ipynb): selection and analysis of the best features for darknet traffic detection and characterization. 

## Running the notebooks

All the notebooks can be executed locally in a conda environment by [anaconda](https://www.anaconda.com/products/individual) installation and the execution of the following commands inside the notebooks folder.

```
conda env create -f darknet.yml
conda activate darknet
jupyter notebook
```
