import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def main():

    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    rng = np.random.default_rng(42)
    mask = rng.choice([True, False], size=X.shape, p=[0.05, 0.95])
    X_nan = X.mask(mask)


    X_train, X_test, y_train, y_test = train_test_split(
        X_nan, y, test_size=0.2, random_state=42, stratify=y
    )


    modelos = {
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        "SVM": SVC(random_state=42),
        "KNN": KNeighborsClassifier()
    }

    mejores_scores = {}
    pipelines = {}


    for nombre, modelo in modelos.items():
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("classifier", modelo)
        ])

        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
        mean_score = scores.mean()
        mejores_scores[nombre] = mean_score
        pipelines[nombre] = pipeline

        print(f"Modelo: {nombre} -> CV Accuracy: {mean_score:.4f}")


    mejor_modelo = max(mejores_scores, key=mejores_scores.get)
    print("\nMejor modelo seleccionado:", mejor_modelo)

    pipeline_final = pipelines[mejor_modelo]
    pipeline_final.fit(X_train, y_train)


    y_pred = pipeline_final.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy en test con {mejor_modelo}: {test_acc:.4f}")


    joblib.dump(pipeline_final, "iris_best_model.pkl")
    print("\nPipeline exportado como iris_best_model.pkl")

if __name__ == "__main__":
    main()
