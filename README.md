# PCG Classification
## Introduction
This is a project aimed at classifying normal and abnormal heart sounds (PCG) using deep learning techniques. The project makes use of the PhysioNet 2016 dataset and PASCAL dataset to train a deep neural network.

## Requirements
- PyTorch
- TorchAudio
- TorchMetrics
- Numpy
- Pandas
- Matplotlib

## Data
The data used in this project can be found at the following link:
- [PhysioNet 2016 dataset](https://archive.physionet.org/physiobank/database/challenge/2016/)
- [PASCAL dataset](http://www.peterjbentley.com/heartchallenge/)

## Code
The code for this project is organized as follows:
- **Initialization:** This step involves loading the required packages and setting up the environment.
- **Preprocessing:** This step involves loading the data, processing it, and preparing it for training.
- **Training:** This step involves training the deep neural network on the processed data.
- **Evaluating:** This step involves evaluating the performance of the trained model on the test data.
- **Custom Models:** The custom models used in this project can be found in the packages directory.
- **Trained Models:** The `state_dict` of already trained models can be found in the models directory. Note that the file names ending with `_TEMP` are saved as a backup during training in case of any program failure.

The entire project is implemented in PyTorch and can be found in the main.ipynb Jupyter Notebook.

## Usage
1. Clone the repository:
```bash
git clone https://github.com/iTzAymen/PCG_Classification.git
```
2. Install the required packages:
```bash
pip install -r requirements.txt
```
3. Open the `main.ipynb` and follow the instructions given in the notebook.

## Results
The results of the project will be reported here.

## References
- PhysioNet 2016 dataset: Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215-e220 [Circulation Electronic Pages; http://circ.ahajournals.org/cgi/content/full/101/23/e215]; 2000 (June 13).

- PASCAL dataset: Clifford GD, Malik M, O'Muircheartaigh J, Li W, Mark RG, Camm J, Stankowski J, Goldberger AL, Moody GB. The impact of using different paradigms for heart sound and pulse analysis in atrial fibrillation detection. Physiol Meas 34(12):1497-1509; 2013 Dec. (PMID:24194764)
