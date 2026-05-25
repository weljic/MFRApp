from pdm.config import DATA_PATH
from pdm.data import prepare_data, split_train_test
from pdm.model import build_model, train_model, evaluate_model
from pdm.serialize import save_model

def main():
    X, y = prepare_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    model = build_model()
    model = train_model(model, X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    print("Metrics:", metrics)

    save_model(model, "models/model.joblib")

if __name__ == "__main__":
    main()