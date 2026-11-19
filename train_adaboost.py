import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score

def train_adaboost_model(train_path, test_path, model_output_path, metrics_output_path):
    # 1. Load the splits
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Separate target variable
    X_train = train_df.drop(columns=['ProdTaken'])
    y_train = train_df['ProdTaken']
    X_test = test_df.drop(columns=['ProdTaken'])
    y_test = test_df['ProdTaken']

    # Identify columns by type
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()

    # 2. Build Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    # Base weak learner with depth 1 or 2 (AdaBoost traditionally uses shallow trees)
    base_estimator = DecisionTreeClassifier(max_depth=2, random_state=42, class_weight='balanced')

    # Define the estimator pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', AdaBoostClassifier(estimator=base_estimator, random_state=42))
    ])

    # 3. Parameter tuning grid
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 1.0]
    }

    # Run Grid Search with cross-validation targeting f1-score
    print("Starting Grid Search hyperparameter tuning for AdaBoost Classifier...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best Hyperparameters: {best_params}")

    # 4. Evaluation
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("\n=== AdaBoost Classifier Test Evaluation Results ===")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"ROC AUC: {test_roc_auc:.4f}")

    # 5. Save the best model and parameters/metrics logs
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(best_model, model_output_path)
    print(f"Saved trained model to {model_output_path}")

    run_summary = {
        'best_params': best_params,
        'metrics': {
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'test_roc_auc': test_roc_auc
        }
    }

    with open(metrics_output_path, 'w') as f:
        json.dump(run_summary, f, indent=4)
    print(f"Saved tuning and evaluation metrics to {metrics_output_path}")

if __name__ == '__main__':
    train_csv = 'visit_with_us_mlops/data/processed/train.csv'
    test_csv = 'visit_with_us_mlops/data/processed/test.csv'
    model_file = 'visit_with_us_mlops/models/best_model.pkl'
    metrics_file = 'visit_with_us_mlops/models/metrics.json'
    train_adaboost_model(train_csv, test_csv, model_file, metrics_file)
