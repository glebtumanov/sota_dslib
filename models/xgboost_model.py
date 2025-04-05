from xgboost import XGBClassifier
from spark_sota_modeling.modeling import BaseModel

class XGBoostModel(BaseModel):
    def train(self, train, valid, target, features, cat_features=[]):
        model = XGBClassifier(**params)
        model.fit(train[features], train[target], eval_set=[(valid[features], valid[target])], verbose=False)
        return model

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X, y):
        # Implement evaluation logic
        pass
