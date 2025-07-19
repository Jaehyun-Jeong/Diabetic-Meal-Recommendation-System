from typing import Dict, List


FOOD_CATEGORIES = ['과일군', '곡류군', '혼합식품', '어육류군', '우유군', '채소군', '지방군']


class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, df):
        self.mean = df.mean()
        self.std = df.std()

    def transform(self, df):
        return (df - self.mean) / self.std

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)


def idx_to_category(pred: Dict[int, List[int]]):
    for patient_id, recommend in pred.items():
        recommend = [FOOD_CATEGORIES[index] for index in recommend]
        pred[patient_id] = recommend
    return pred
