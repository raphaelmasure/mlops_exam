import mlflow

model_name = "xgboost"
version = 1

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{version}"
)

list_libs = ["15", "10", "5"]

results = model.predict(list_libs, params={"k": 1})
print(results)