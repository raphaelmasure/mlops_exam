import sys
import mlflow
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import s3fs
from constants import TEXT_FEATURE, Y, DATA_PATH, LABEL_PREFIX
from utils import write_training_data

# Load data
def load_data():
    df = pd.read_csv("/home/onyxia/work/mlops_exam/data/DSA-2025_clean_data.tsv", sep='\t')
    return df.sample(frac=0.1)



def train(remote_server_uri, experiment_name, run_name, n_estimators, max_depth):
    """
    Train an XGBoost model and log results to MLflow.
    """
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name):
        df = load_data()

        features = ['chol', 'crp', 'phos']
        X = df[features]
        y = df[Y]
        
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        xgb_model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')

        param_grid = {
            'n_estimators': [50, 100, 150], 
            'max_depth': [3, 5, 7],        
        }

        grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)

        model_path = f"models/{run_name}.json"
        best_model.save_model(model_path)
        mlflow.log_artifact(model_path)

        print("Training complete. Best parameters:", grid_search.best_params_)
        print("Test accuracy:", accuracy)
        print("Test precision:", precision)

if __name__ == "__main__":
    remote_server_uri = sys.argv[1]
    experiment_name = sys.argv[2]
    run_name = sys.argv[3]
    n_estimators = int(sys.argv[4])
    max_depth = int(sys.argv[5])

    train(remote_server_uri, experiment_name, run_name, n_estimators, max_depth)