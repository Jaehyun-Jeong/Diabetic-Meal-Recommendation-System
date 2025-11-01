import numpy as np
import pandas as pd

from DataLoader import load_data

def cleansing_data():

    # csv를 로드
    df = load_data()

    # 사용할 컬럼을 선택
    features = [
        'carbs', 'calories', 'protein', 'fat', 'g0',
        'Age', 'Gender', 'BMI', 'Body weight ', 'Height ',
        'A1c PDL (Lab)'
    ]
    # 예측할 타겟을 선택
    target = ['delta_g']

    # Cleansing
    # NA: Not Applicable 가 있는 열을 제거
    df = df[features + target].dropna()

    # 'A1c PDL (Lab)' 이 6.0 이상이면 당뇨
    df.loc[df['A1c PDL (Lab)'] >= 6.0, 'has_diabetes'] = 1.0
    df.loc[df['A1c PDL (Lab)'] < 6.0, 'has_diabetes'] = 0.0
    # 'A1c PDL (Lab)' 열을 제거, 유저가 제공 불가능한 데이터므로 사용 불가능
    df = df.drop('A1c PDL (Lab)', axis=1)

    # 실제로 얻어진 컬럼
    features = [
        'carbs', 'calories', 'protein', 'fat', 'g0',
        'Age', 'Gender', 'BMI', 'Body weight ', 'Height ',
        'has_diabetes'
    ]

    # 만약 식사 정보를 넣고싶으면 주석 제거
    '''
    df.loc[df['meal_type'] == 'dinner', 'meal_type'] = "Dinner"
    df.loc[df['meal_type'] == 'lunch', 'meal_type'] = "Lunch"
    df.loc[df['meal_type'] == 'Snacks', 'meal_type'] = "Snack"
    df.loc[df['meal_type'] == 'snack', 'meal_type'] = "Snack"
    df.loc[df['meal_type'] == 'snack 1', 'meal_type'] = "Snack"
    df.loc[df['meal_type'] == 'breakfast', 'meal_type'] = "Breakfast"
    '''

    # x, y 나누기
    x_train = df[features]
    y_train = df[target]

    # category 데이터는 one-hot encoding
    categorical_cols = x_train.select_dtypes(include=['object', 'category']).columns
    x_train = pd.get_dummies(x_train, columns=categorical_cols, dtype=float)

    # 데이터 저장
    x_train.to_csv('./dataset/x.clean.pruned.v3.csv', index=False)
    y_train.to_csv('./dataset/delta_g.clean.pruned.v3.csv', index=False)

    return df

if __name__ == "__main__":
    df = cleansing_data()
