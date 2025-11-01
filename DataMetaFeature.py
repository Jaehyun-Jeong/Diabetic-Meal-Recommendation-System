from typing import Tuple
from datetime import datetime
import numpy as np
import pandas as pd

from DataLoader import load_data


# Ex)
# 2025-05-01 14:23:00
# -> (4, 13) (Month, Day, Hour) Counting from 0
def time_preprocess(time_str: str) -> Tuple[int, int, int]:

    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")

    return dt.month, dt.day, dt.weekday(), dt.hour


def time_fourier_feature(
    df: pd.DataFrame,
    feature_name: str = "meal_time",
) -> pd.DataFrame:

    time_upper_bounds = {
        "month": 11,
        "day": 29,  # It differs by month
        "weekday": 6,
        "hour": 23,
    }

    df[["meal_month", "meal_day", "meal_weekday", "meal_hour"]] = (
        df[feature_name].apply(time_preprocess).apply(pd.Series)
    )

    # fourier feature
    df["sin_meal_month"] = np.sin(
        2 * np.pi * df["meal_month"] / time_upper_bounds["month"]
    )
    df["cos_meal_month"] = np.cos(
        2 * np.pi * df["meal_month"] / time_upper_bounds["month"]
    )
    df["sin_meal_day"] = np.sin(2 * np.pi * df["meal_day"] / time_upper_bounds["day"])
    df["cos_meal_day"] = np.cos(2 * np.pi * df["meal_day"] / time_upper_bounds["day"])
    df["sin_meal_weekday"] = np.sin(
        2 * np.pi * df["meal_weekday"] / time_upper_bounds["weekday"]
    )
    df["cos_meal_weekday"] = np.cos(
        2 * np.pi * df["meal_weekday"] / time_upper_bounds["weekday"]
    )
    df["sin_meal_hour"] = np.sin(
        2 * np.pi * df["meal_hour"] / time_upper_bounds["hour"]
    )
    df["cos_meal_hour"] = np.cos(
        2 * np.pi * df["meal_hour"] / time_upper_bounds["hour"]
    )

    return df


# time
# month
# day of the week
def add_time_info_data():

    df = load_data()

    features = [
        "meal_time",
        "carbs",
        "calories",
        "protein",
        "fat",
        "g0",
        "Age",
        "Gender",
        "BMI",
        "Body weight ",
        "Height ",
        "A1c PDL (Lab)",
    ]
    target = ["delta_g"]

    # Cleansing
    df = df[features + target].dropna()
    df = time_fourier_feature(df)

    df.loc[df["A1c PDL (Lab)"] >= 6.0, "has_diabetes"] = 1.0
    df.loc[df["A1c PDL (Lab)"] < 6.0, "has_diabetes"] = 0.0
    df = df.drop("A1c PDL (Lab)", axis=1)

    features = [
        "carbs",
        "calories",
        "protein",
        "fat",
        "g0",
        "Age",
        "Gender",
        "BMI",
        "Body weight ",
        "Height ",
        "has_diabetes",
        "sin_meal_hour",
        "cos_meal_hour",
        "sin_meal_day",
        "cos_meal_day",
        "sin_meal_month",
        "cos_meal_month",
    ]

    """
    df.loc[df['meal_type'] == 'dinner', 'meal_type'] = "Dinner"
    df.loc[df['meal_type'] == 'lunch', 'meal_type'] = "Lunch"
    df.loc[df['meal_type'] == 'Snacks', 'meal_type'] = "Snack"
    df.loc[df['meal_type'] == 'snack', 'meal_type'] = "Snack"
    df.loc[df['meal_type'] == 'snack 1', 'meal_type'] = "Snack"
    df.loc[df['meal_type'] == 'breakfast', 'meal_type'] = "Breakfast"
    """

    x_train = df[features]
    y_train = df[target]

    categorical_cols = x_train.select_dtypes(include=["object", "category"]).columns
    x_train = pd.get_dummies(x_train, columns=categorical_cols, dtype=float)

    x_train.to_csv("./dataset/x.clean.pruned.hour.day.month.csv", index=False)
    y_train.to_csv("./dataset/delta_g.clean.pruned.hour.day.month.csv", index=False)

    return df


if __name__ == "__main__":
    df = add_time_info_data()
