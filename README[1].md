# MineScope Data Mining Lab

## Complete workflow

### Preprocess
Upload your own `.arff` or `.csv` dataset.

The Current Relation panel displays:
- Relation
- Attributes
- Instances
- Sum of Weights

It also displays attribute types, missing values, unique values, duplicate count and a dataset preview.

### Classify
Choose:
- ID3
- J48
- Naive Bayes
- KNN

Choose the class/target attribute.

### Test
Choose:
- Cross Validation: fixed at 10 folds
- Percentage Split: fixed at 66% training / 34% testing

### Classification Output
When you click **Start Classification**, the classifier output is displayed immediately on the Classify page. A separate Classification Output page is also included for reviewing completed runs. The output contains:
- classifier
- class attribute
- evaluation method
- training/testing counts
- Accuracy
- Precision
- Recall
- F1
- ROC AUC
- Decision Time
- Confusion Matrix
- Classification Report

The output can be downloaded as a `.txt` file.

### ROC Curves
Compare all classifiers that have been run.

### Decision Tree
Visual tree for ID3 and J48.

## Run on Windows

Recommended Python: 3.11 or 3.12.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open:
http://localhost:8501

## Note
ID3 and J48 are implemented using entropy-based scikit-learn decision trees. J48 is an educational C4.5/J48-family approximation and may differ from exact Weka J48 output.

## Automatic Classification
There is no **Start Classification** button. Classification runs automatically after selecting the algorithm, class attribute, and test option on the Classify page.


## Automatic Classification
The **Start Classification** button has been removed. Classification and classifier output are generated automatically after selecting the algorithm, class attribute and test option.


## Automatic Class Attribute
The Class/Target Attribute selection has been removed from the UI.
The application automatically uses the **last column of the uploaded dataset** as the class attribute.


## Workspace Update
The separate **Classification Output** workspace has been removed.
Classifier output remains available directly on the **Classify** page after automatic classification.
