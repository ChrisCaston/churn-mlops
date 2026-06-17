# Customer Churn Prediction — End-to-End MLOps on Google Vertex AI

An end-to-end machine learning project that trains, deploys, and serves a customer churn prediction model on Google Cloud's Vertex AI, then translates the model's predictions into a concrete business outcome: a simulated retention campaign with a measurable return on investment.

The goal of this project was not just to train a model, but to build the full MLOps lifecycle around it — from raw data in cloud storage through a live prediction endpoint — and to demonstrate how the model drives real business value.

## Problem

A telecom company wants to reduce customer churn. If they can predict which customers are likely to leave, they can intervene with retention offers before those customers cancel. This project builds that predictive capability and quantifies what it's worth.

## Dataset

The [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — a public benchmark dataset published by IBM. It contains ~7,000 customer records with 21 features (tenure, monthly charges, contract type, services subscribed, etc.) and a binary churn label. The data is synthetic but realistic, with no privacy or licensing concerns, which makes it well-suited to a public portfolio project.

## Architecture

The project runs entirely on Google Cloud Platform:

```
Raw CSV → Cloud Storage (GCS)
            ↓
      Data prep & EDA (Vertex AI Workbench)
            ↓
      Processed train/test splits → GCS
            ↓
      Vertex AI Custom Training Job (scikit-learn Random Forest)
            ↓
      Model artifact → GCS
            ↓
      Vertex AI Model Registry (versioned)
            ↓
      Vertex AI Endpoint (live online predictions)
            ↓
      Business value analysis (retention campaign ROI)
```

## Tech stack

- **Platform:** Google Cloud Platform — Vertex AI (Workbench, Custom Training, Model Registry, Endpoints), Cloud Storage
- **ML:** scikit-learn (Random Forest classifier)
- **Language:** Python (pandas, NumPy, scikit-learn)
- **Tooling:** Vertex AI Python SDK, Git/GitHub

## Repository contents

| File | Description |
|------|-------------|
| `churn_eda_preprocessing.ipynb` | Phase 1 — loads raw data from GCS, exploratory data analysis, preprocessing, and train/test split saved back to GCS |
| `train.py` | The training script submitted to the Vertex AI Custom Training Job |
| `churn_training.ipynb` | Phase 2 — packages and submits the custom training job to Vertex AI |
| `churn_deployment.ipynb` | Phase 3 — registers the model in the Model Registry and deploys it to a live endpoint |
| `churn_business_demo.ipynb` | Business value demo — calls the live endpoint and simulates a retention campaign with ROI analysis |

## Model performance

The Random Forest classifier was trained with balanced class weights to account for the ~73/27 class imbalance.

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.81 |
| Accuracy | 0.77 |
| Churn recall | 0.65 |
| Churn precision | 0.55 |

The model is intentionally tuned toward higher recall on the churn class — in a retention context, missing a customer who is about to leave is far more costly than sending a retention offer to someone who would have stayed.

## Business impact

The technical metrics only matter if they translate to value. The business demo simulates how a retention team would use the model on a customer base of 1,407 (the held-out test set):

- The model flagged **290 customers** as churn risks
- A retention campaign sends each a one-time $50 offer → **$14,500 campaign cost**
- Of the flagged customers, 178 were genuine churners; at a 30% offer-acceptance rate, an estimated **53 customers are retained**
- Retained customers preserve an estimated **$49,123** in 12-month revenue
- **Net value: $34,623 — a 239% return on the campaign**

The key insight: even though the model is imperfect (112 flagged customers would have stayed anyway), the campaign is still strongly profitable, because a wasted $50 offer is trivial next to the ~$930 lifetime value of a retained customer. This is the business justification for optimizing the model toward recall.

## Cost management

Vertex AI prediction endpoints bill per hour while deployed, regardless of traffic. To stay within the GCP free-tier credits, the endpoint was undeployed immediately after testing — the registered model and endpoint shell remain (these are free) and the model can be redeployed on demand. Training used small machine types in `us-central1`.

## Future improvements

- **Return churn probabilities** rather than binary labels, to let the business prioritize the highest-risk customers
- **Tune the decision threshold** to the actual cost/value tradeoff rather than a fixed 0.5 cutoff
- **Model retention offers as ongoing discounts** to capture lost margin on false positives
- **Add Vertex AI Model Monitoring** for prediction drift and feature skew
- **Automate retraining** via a Vertex AI Pipeline that retrains and conditionally redeploys
- **One-hot encode** unordered categorical features instead of label encoding

## Notes

This project was built as a hands-on learning exercise covering the Google Cloud Professional Machine Learning Engineer certification domains: data preparation, model development, deployment and serving, and MLOps practices.
