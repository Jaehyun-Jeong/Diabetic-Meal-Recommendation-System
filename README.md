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
│   ├── evaluated_meals.csv      # Meal data with nutritional information
│   └── total_metrics.csv        # Patient characteristics data
├── saved_models/                # Directory for saved trained models
├── DataLoader.py               # Data loading and preprocessing utilities
├── model.py                    # Main model implementations (BM25CosSim, LMF)
├── preprocess.py              # Data preprocessing and normalization
├── main.py                    # Main execution script
├── main_BM25CosSim.py         # BM25+CosSim specific main script
├── main_delta_g.py            # Delta glucose specific analysis
├── train.py                   # Model training with hyperparameter tuning
└── train_LMF.py               # LMF model training script
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

## Installation

```bash
# Clone the repository
git clone git@github.com:Jaehyun-Jeong/Diabetic-Meal-Recommendation-System.git
cd Diabetic-Meal-Recommendation-System

# Install required packages
pip install pandas numpy scipy scikit-learn implicit tqdm joblib
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
# Run grid search for optimal parameters
python train.py

# Train LMF model
python train_LMF.py
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

## Future Work

- [ ] Implement regression models for delta_g and g_max prediction
- [ ] Add support for more diverse food categories
- [ ] Improve model interpretability
- [ ] Integrate real-time glucose monitoring data
- [ ] Develop web interface for healthcare providers
