# Diabetic Meal Recommendation System

A machine learning-based system for recommending suitable meal categories to diabetic patients using BM25 and cosine similarity algorithms. The system analyzes patient characteristics and historical meal data to provide personalized food category recommendations.

## Features

- **Patient Similarity Analysis**: Uses cosine similarity to find patients with similar characteristics (age, gender, BMI, body weight, height)
- **BM25 Weighting**: Implements BM25 algorithm for meal recommendation scoring
- **Food Category Classification**: Recommends from 7 Korean food categories:
  - 과일군 (Fruit group)
  - 곡류군 (Grain group)
  - 혼합식품 (Mixed foods)
  - 어육류군 (Fish and meat group)
  - 우유군 (Dairy group)
  - 채소군 (Vegetable group)
  - 지방군 (Fat group)
- **Multiple Model Support**: Includes BM25+CosSim and Logistic Matrix Factorization (LMF) models
- **Evaluation Metrics**: Uses Recall@K for model performance evaluation

## Project Structure

```
Diabetic-Meal-Recommendation-System/
├── dataset/
│   ├── evaluated_meals.csv           # Meal data with nutritional information
│   ├── total_metrics.csv             # Patient characteristics data
│   ├── x.clean.*.csv                 # Preprocessed feature datasets
│   └── delta_g.clean.*.csv           # Cleaned target datasets
├── saved_models/
│   ├── BM25CosSim_model.pkl          # Trained recommendation model
│   ├── XGB_delta_g.pkl               # XGBoost model for delta_g prediction
│   └── XGB_g_max.pkl                 # XGBoost model for g_max prediction
├── DataLoader.py                     # Data loading and preprocessing utilities
├── DataCleansing.py                  # Feature engineering (basic)
├── DataMetaFeature.py                # Time-based feature engineering
├── model.py                          # Model implementations (BM25CosSim, LMF, XGBoost)
├── preprocess.py                     # Data preprocessing and normalization utilities
├── main.py                           # Main execution script
├── main_BM25CosSim.py                # BM25+CosSim evaluation script
├── main_delta_g.py                   # XGBoost delta_g training and evaluation
├── main_g_max.py                     # XGBoost g_max training and evaluation
├── train.py                          # BM25 hyperparameter grid search
├── train_LMF.py                      # LMF model training script
├── train_delta_g.py                  # XGBoost delta_g model training
└── train_g_max.py                    # XGBoost g_max model training
```

## Data Schema

### Patient Features
- `patient_id`: Unique patient identifier
- `Age`, `Gender`, `BMI`, `Body weight`, `Height`: Patient characteristics

### Meal Features
- `meal_time`, `meal_type`: Temporal meal information
- `carbs`, `protein`, `fat`, `fiber`: Nutritional content
- `delta_g`, `g_max`: Glucose response metrics
- `gl`: Glycemic load
- `cho_ratio`, `protein_ratio`, `fat_ratio`: Macronutrient ratios
- `식품군분류`: Food category classification
- `meal_score`: Quality score of the meal

## Models

### 1. BM25CosSim
Combines BM25 weighting with cosine similarity for patient-based collaborative filtering:
- **BM25 Parameters**: K1 (optimal: ~3.02), B (optimal: ~1.99)
- **Similarity Features**: Age, Gender, BMI, Body weight, Height
- **Output**: Ranked food category recommendations

### 2. Logistic Matrix Factorization (LMF)
Implements collaborative filtering using matrix factorization:
- **Factors**: 50 (optimal)
- **Learning Rate**: 0.001
- **Regularization**: 100.0
- **Iterations**: 250

### 3. XGBoost Regression Models
Predicts glucose response metrics from meal characteristics:
- **delta_g**: Change in glucose levels after meal
- **g_max**: Maximum glucose level after meal
- **Features**: Nutritional content, time features, patient characteristics
- **Evaluation**: K-fold cross-validation with MAE and R² metrics

## Installation

```bash
# Clone the repository
git clone git@github.com:Jaehyun-Jeong/Diabetic-Meal-Recommendation-System.git
cd Diabetic-Meal-Recommendation-System

# Install required packages
pip install pandas numpy scipy scikit-learn implicit tqdm joblib catboost xgboost
```

## Usage

### Basic Recommendation

```python
from DataLoader import load_data, select_similar_features, split_train_val
from model import BM25CosSim

# Load and prepare data
df = load_data()
df = df.loc[df['meal_score'] >= 50.0]  # Filter good meals

# Split data
patient_ids = df['patient_id'].unique()
val_ids = np.random.choice(patient_ids, size=10, replace=False)
train_df, val_df = split_train_val(df, val_ids)

# Initialize and train model
model = BM25CosSim(K1=3.02, B=1.99)
model.fit(train_df)

# Make recommendations
patient_features = select_similar_features(val_df, ['Age', 'Gender', 'BMI', 'Body weight ', 'Height '])
recommendations = model.predict(patient_features)
```

### Training with Hyperparameter Tuning

```bash
# Run grid search for optimal BM25 parameters
python train.py

# Train LMF model
python train_LMF.py
```

### Training Glucose Prediction Models

```bash
# Step 1: Preprocess data (choose feature set)
python DataCleansing.py          # Basic features
python DataMetaFeature.py        # Add time-based features (sin/cos transformations)

# Step 2: Train XGBoost models
python main_delta_g.py --xpath dataset/x.clean.pruned.v3.csv --ypath dataset/delta_g.clean.pruned.v3.csv
python main_g_max.py --xpath dataset/x.clean.pruned.v2.csv --ypath dataset/g_max.clean.pruned.v2.csv

# Or use dedicated training scripts
python train_delta_g.py
python train_g_max.py
```

### Model Evaluation

The system uses Recall@K metric to evaluate recommendation quality:

```python
from model import BM25CosSim

# Evaluate model performance
score = BM25CosSim.recall_at_K(predictions, ground_truth, K=3)
print(f"Recall@3: {score:.4f}")
```

## Configuration

### Key Parameters
- `GOOD_MEAL_SCORE`: 50.0 (minimum meal quality threshold)
- `FOOD_CATEGORIES`: 7 Korean food categories
- `Validation Size`: 10 patients
- `Recall@K`: K=3 for evaluation

### Optimal Hyperparameters (from grid search)
- **BM25 K1**: 3.02
- **BM25 B**: 1.99
- **LMF Factors**: 50
- **LMF Learning Rate**: 0.001
- **LMF Regularization**: 100.0

## Model Persistence

```python
from model import save_model_normalizer, load_model_normalizer

# Save trained model
save_model_normalizer('saved_models/model.pkl', model, normalizer)

# Load model for inference
model, normalizer = load_model_normalizer('saved_models/model.pkl')
```

## Workflow Overview

### Recommendation System Workflow
1. Load data from `evaluated_meals.csv` and `total_metrics.csv`
2. Filter meals with `meal_score >= 50.0` (good meals only)
3. Split data into train/validation sets by patient_id
4. Extract and normalize similarity features (Age, Gender, BMI, Weight, Height)
5. Train BM25CosSim or LMF model on training data
6. Generate food category rankings for validation patients
7. Evaluate using Recall@K metric

### Glucose Prediction Workflow
1. Preprocess data with `DataCleansing.py` or `DataMetaFeature.py`
2. Create features including:
   - Nutritional content (carbs, protein, fat, fiber)
   - Time features (meal_time transformed to sin/cos)
   - Patient characteristics (Age, BMI, diabetes status)
3. Train XGBoost regression models
4. Evaluate using K-fold cross-validation (MAE, R²)
5. Save models to `saved_models/`

## Key Notes

- **Column Names**: Note trailing spaces in `'Body weight '` and `'Height '`
- **Random Seed**: Set to 42 for reproducibility
- **Food Categories**: Must be set as categorical with specific order
- **Data Normalization**: Z-score normalization applied to patient features
- **BM25 Approach**: Treats patients as documents and food categories as terms

## Future Work

- [ ] Add support for more diverse food categories
- [ ] Improve model interpretability
- [ ] Integrate real-time glucose monitoring data
- [ ] Develop web interface for healthcare providers
- [ ] Implement ensemble methods combining recommendation + glucose prediction
