import sys
import mlflow
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Load data
def load_data():
    df = pd.read_csv("/home/onyxia/work/mlops_exam/data/DSA-2025_clean_data.tsv", sep='\t')
    return df.sample(frac=0.1)

def train(remote_server_uri, experiment_name, run_name):
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name):
        df = load_data()
        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=0)
        
        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        param_grid = {
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
        }
        
        model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train_vec, y_train)
        
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test_vec)
        acc = accuracy_score(y_test, predictions)
        
        for param, value in grid_search.best_params_.items():
            mlflow.log_param(param, value)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(best_model, "xgboost_model")
        
        print(f"Best model accuracy: {acc}")

if __name__ == "__main__":
    remote_server_uri = sys.argv[1]
    experiment_name = sys.argv[2]
    run_name = sys.argv[3]
    
    train(remote_server_uri, experiment_name, run_name)
