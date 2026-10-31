# Health Classifier: Diabetes Prediction with PyTorch

A feed-forward neural network that predicts diabetes from 8 medical measurements, built for the Systems Engineering module (WiSe 2023/24, group project).

## Dataset

`dataset/diabetes.csv`: Pima Indians Diabetes Database, 768 patients (female, 21+, Pima Indian heritage), 8 predictor variables and a binary `Outcome` target. Originally from the National Institute of Diabetes and Digestive and Kidney Diseases; see `dataset/readme.txt` for the citation.

## Method

- 80/20 train/test split with a fixed seed (`state = 974`) for reproducibility
- Features standardised with `StandardScaler`
- Fully connected network with ReLU activations and dropout (p = 0.5), single-logit output trained with `BCEWithLogitsLoss`
- SGD optimiser; learning rate and epoch count were tuned across trials
- Metrics: accuracy, precision, recall, F1-score, AUC-ROC

Every run is appended to `Trials_list.json` and exported to `Trials.csv`.

## Results

Best trial: test accuracy 81.17 %, precision 76.92 %, recall 70.18 %, F1 73.39 %, AUC-ROC 84.97 % (lr = 0.01, 7000 epochs). All runs are in `Trials.csv`.

## Running

```bash
pip install torch pandas scikit-learn matplotlib seaborn
python Code.py
```
