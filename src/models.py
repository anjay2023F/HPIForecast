"""Model zoo. Baselines are computed directly from features in evaluate.py."""
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASELINES = ["Persistence", "Historical mean"]


def get_models(seed: int = 42) -> dict:
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=seed),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=seed),
    }
