# mental-health-self-analysis

## Data preprocessing-

->Data collection

->data cleaning 

->handling missing values

->feautre engineering

## Model Selection Rationale
Random Forest was selected as the final model due to its:

1)High accuracy and precision on the test set.

2)Ability to handle imbalanced data using class weights.

3)Interpretability through feature importance scores.

## Installation

### Prerequisites
Ensure Python 3.8+ is installed. Then, install required libraries:
```
pip install -r requirements.txt

```


### To run the inference script:
```

python main.py --gad 10 --phq 12 --epworth 8
```


(replace with actual values)

### To run the streamlit ui
```
streamlit run app.py

```




