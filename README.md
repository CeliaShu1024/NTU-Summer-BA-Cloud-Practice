# Model-Training
* Content of dataset:
  We applied `DBS_SingDollar.csv` as the dataset. It is a small dataset containing 369 data. There are 3 attributes in this dataset and only 2 of them were used in this model training stage, which are DBS and SGD.
  
* Regressive algorithm used:
  The algorithms of regression are linear regression and classical decision tree. These two algorithm are embedded in data scikit-learn Python API. All parameters of these algorithms are set as default. The progress of this model training stage can be found in `model.py`.
  
* Exporting:
  To export models, we applied joblib API of Python.
  
* Verification:
  We tested the validity of these two models by the dataset. This progress can be found in `check_model.py`.
