import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from .data import load_data
from .features import preprocess
from .config import TARGET, OUTPUT_DIR

def main():
    train, test = load_data()

    train = preprocess(train)
    test = preprocess(test)

    X_train = train.drop(columns=[TARGET, "Name", "Ticket", "Cabin"])
    y_train = train[TARGET]

    X_test = test.drop(columns=["Name", "Ticket", "Cabin"])

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": preds
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)

if __name__ == "__main__":
    main()