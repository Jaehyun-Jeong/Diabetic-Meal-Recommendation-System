import numpy as np
import pandas as pd

from DataLoader import load_data

def cleansing_data():

    df = load_data()

    features = [
        'carbs', 'calories', 'protein', 'fat', 'g0',
        'Age', 'Gender', 'BMI', 'Body weight ', 'Height ',
        'A1c PDL (Lab)'
    ]
    target = ['delta_g']

    # Cleansing
    df = df[features + target].dropna()

    df.loc[df['A1c PDL (Lab)'] >= 6.0, 'has_diabetes'] = 1.0
    df.loc[df['A1c PDL (Lab)'] < 6.0, 'has_diabetes'] = 0.0
    df = df.drop('A1c PDL (Lab)', axis=1)

    features = [
        'carbs', 'calories', 'protein', 'fat', 'g0',
        'Age', 'Gender', 'BMI', 'Body weight ', 'Height ',
        'has_diabetes'
    ]

    '''
    df.loc[df['meal_type'] == 'dinner', 'meal_type'] = "Dinner"
    df.loc[df['meal_type'] == 'lunch', 'meal_type'] = "Lunch"
    df.loc[df['meal_type'] == 'Snacks', 'meal_type'] = "Snack"
    df.loc[df['meal_type'] == 'snack', 'meal_type'] = "Snack"
    df.loc[df['meal_type'] == 'snack 1', 'meal_type'] = "Snack"
    df.loc[df['meal_type'] == 'breakfast', 'meal_type'] = "Breakfast"
    '''

    x_train = df[features]
    y_train = df[target]

    categorical_cols = x_train.select_dtypes(include=['object', 'category']).columns
    x_train = pd.get_dummies(x_train, columns=categorical_cols, dtype=float)

    x_train.to_csv('./dataset/x.clean.pruned.v3.csv', index=False)
    y_train.to_csv('./dataset/delta_g.clean.pruned.v3.csv', index=False)

    return df

if __name__ == "__main__":
    df = cleansing_data()
