# Medical Ensurance

## Hybrid Quantum Models for Health Insurance & Provider Fraud

### This repo contains two Jupyter notebooks exploring hybrid quantum–classical models on healthcare tabular data:
- 1. health_insurance_hybrid.ipynb
High-cost claims detection with a hybrid QNN vs. RandomForest, including a rare-event experiment (top-1% tail).
- 2. fraud_insurance.ipynb
Provider-level fraud detection built from raw claim tables using robust cleaners + provider aggregations, then a hybrid QNN baseline.

### Both notebooks include:
- Robust data cleaning & preprocessing
- Class imbalance handling
- A PennyLane circuit with optional colored diagram styling


### Notebook health_insurance_hybrid.ipynb: Insurance — Rare Event Experiment

Data: leandrenash/enhanced-health-insurance-claims-dataset

How the “risk” label is created (high vs not_high)

Creating the “risk” Label (high vs not_high)

Step 1. Identify the Claim Cost Column
We first look for standard claim-related fields:
	•	ClaimAmount (preferred; “The amount claimed in USD”)
	•	Alternatives: ClaimTotal, BilledAmount, ClaimValue, Charges
We explicitly exclude non-claim financial fields such as:
	•	PatientIncome, Salary, Wage, Payroll
Even if PatientIncome is larger, it is not the claim cost — the label must be based only on the claim amount, not patient earnings.

Step 2. Compute the 90th Percentile (q90)
We calculate the 90th percentile of the claim-cost values.
Example: if q90 = 5000, only the top 10% of claims cost more than 5000.

Step 3. Create the Labels
	•	If a claim’s cost is ≥ q90 → label high
	•	Otherwise → not_high

This ensures that “high” = the most expensive 10% of claims, and avoids mistakenly using fields like PatientIncome.

Rare-event variant (top-1%)

To probe quantum benefits in the extreme tail, the notebook also defines rare_high as the top 1% of claim costs and compares Hybrid QNN vs. RandomForest with PR-AUC, recall, precision@k, etc.

Model architecture (two-layer hybrid quantum)

A two-layer hybrid quantum model is used to handle complex, high-dimensional, imbalanced claim data:
	•	Classical front-end: MLP → 6-dimensional embedding
	•	Quantum core (run twice sequentially):
	•	6 wires; data embedding with RY on each wire
	•	BasicEntanglerLayers with 4 entangling blocks (RY per wire + ring CNOTs)
	•	Readout: [⟨Z₀⟩, ⟨Z₁⟩]
	•	Classical head: small linear layer → logits for 2 classes

The quantum blocks can capture multi-way categorical interactions and rare patterns more compactly than classical-only models, which may help in rare-event detection.

#### Results
```bash
ROC-AUC=0.986 | PR-AP=0.999 | Brier=0.398
Threshold: 0.3425525525525525
Accuracy: 94.07%

Classification report:
               precision    recall  f1-score   support

    nonfraud       0.61      0.98      0.75        61
       fraud       1.00      0.94      0.97       614

    accuracy                           0.94       675
   macro avg       0.80      0.96      0.86       675
weighted avg       0.96      0.94      0.95       675

Confusion matrix:
 [[ 60   1]
 [ 39 575]]
```

### Notebook fraud_insurance.ipynb: Provider Fraud — Aggregations + Hybrid QNN

Data: Automatically tries these public slugs until one succeeds:
	•	rohitrox/healthcare-provider-fraud-detection-analysis
	•	shivamb/healthcare-provider-fraud-detection
	•	luisfredgs/healthcare-provider-fraud-detection-analysis
	•	govindkrishnareddy/healthcare-provider-fraud-detection-analysis

Pipeline summary
	•	Robust cleaners (clean_dataframe) and type normalization (yes/no→0/1, date parsing, LOS derivation)
	•	Provider-level aggregation (build_provider_agg) — counts, LOS stats, cost stats (sum/mean/std/min/max), and cardinalities
	•	Merge labels → provider-level feature matrix → standardize
	•	Hybrid QNN classifier
	•	Imbalance-aware training, ROC/PR/Brier, confusion matrices, calibration curves

#### Results
```bash
ROC-AUC=1.000 | PR-AP=0.998 | Brier=0.164
Threshold: 0.37786786786786786
Accuracy: 99.72%

Classification report:
               precision    recall  f1-score   support

    nonfraud       1.00      1.00      1.00       981
       fraud       0.98      0.99      0.99       101

    accuracy                           1.00      1082
   macro avg       0.99      0.99      0.99      1082
weighted avg       1.00      1.00      1.00      1082

Confusion matrix:
 [[979   2]
 [  1 100]]
 ```
