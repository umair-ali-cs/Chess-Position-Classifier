# model.py
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline                       
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, f1_score  
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("⚠ LightGBM not installed. Run: pip install lightgbm")


FEATURE_COLUMNS = [
    "material_diff",
    "mobility",
    "king_in_check",       
    "pawn_structure",
    "attacked_pieces",
    "center_control",
    "king_safety",
    "game_phase",
    "passed_pawns",        
    "bishop_pair",         
    "open_file_rooks",     
    "weighted_mobility",   
]


def load_dataset(csv_path="chess_dataset.csv"):
    """Loads the CSV and returns X, y, feature names."""
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLUMNS]
    y = df["label"]
    return X, y, FEATURE_COLUMNS


def plot_confusion_matrix(y_true, y_pred, labels, model_name="Model", save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.show()
    plt.close()


def tune_knn(X_train, y_train, cv=5):
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights':     ['uniform', 'distance'],
        'metric':      ['euclidean', 'manhattan']
    }
    grid = GridSearchCV(KNeighborsClassifier(), param_grid,
                        cv=cv, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best KNN params: {grid.best_params_}")
    print(f"Best cross-val F1 (macro): {grid.best_score_:.4f}")
    return grid.best_estimator_


def tune_decision_tree(X_train, y_train, cv=5):
    param_grid = {
        'max_depth':        [3, 5, 7, 10, None],
        'min_samples_split':[2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion':        ['gini', 'entropy'],
        'class_weight':     ['balanced', None],
    }
    grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid,
                        cv=cv, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best Decision Tree params: {grid.best_params_}")
    print(f"Best cross-val F1 (macro): {grid.best_score_:.4f}")
    return grid.best_estimator_


def tune_random_forest(X_train, y_train, cv=5):
    param_grid = {
        'n_estimators':     [100, 200],
        'max_depth':        [10, 15, None],
        'min_samples_split':[5, 10],
        'class_weight':     ['balanced', None]
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid,
                        cv=cv, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best RF params: {grid.best_params_}")
    print(f"Best cross-val F1 (macro): {grid.best_score_:.4f}")
    return grid.best_estimator_


def tune_lgbm(X_train, y_train, cv=5):
    if not LGBM_AVAILABLE:
        print("Skipping LightGBM tuning — not installed.")
        return None

    param_grid = {
        'n_estimators':  [300, 500],
        'learning_rate': [0.05, 0.1],
        'num_leaves':    [31, 63],
        'max_depth':     [-1, 10],
        'class_weight':  ['balanced', None],
    }
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message='X does not have valid feature names'
        )
        grid = GridSearchCV(
            LGBMClassifier(random_state=42, verbose=-1),
            param_grid, cv=cv, scoring='f1_macro', n_jobs=-1
        )
        grid.fit(X_train,y_train)

    print(f"Best LightGBM params: {grid.best_params_}")
    print(f"Best cross-val F1 (macro): {grid.best_score_:.4f}")
    return grid.best_estimator_


def prepare_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def evaluate_model(model, X_test, y_test, model_name="Model",
                   plot_cm=True, save_cm_path=None):
    y_pred = model.predict(X_test)
    print(f"\n{'='*40}")
    print(f"  {model_name}")
    print(f"{'='*40}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    accuracy = np.mean(y_pred == y_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Accuracy: {accuracy:.4f}  |  F1 macro: {f1:.4f}")
    if plot_cm:
        labels = sorted(set(y_test) | set(y_pred))
        plot_confusion_matrix(y_test, y_pred, labels, model_name, save_cm_path)
    return y_pred, f1  


def save_model(model, scaler, model_path="model.pkl", scaler_path="scaler.pkl"):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")


def load_model(model_path="model.pkl", scaler_path="scaler.pkl"):
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_position(features_dict, model, scaler):
    """features_dict: output of extract_features()"""
    row = pd.DataFrame([features_dict])[FEATURE_COLUMNS]
    row_scaled = scaler.transform(row)
    return model.predict(row_scaled)[0]


def plot_feature_importance(model, feature_names, title="Feature Importance"):
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute.")
        return
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    print("\n--- Feature Importance Values ---")
    for i in indices:
        print(f"{feature_names[i]:<20} : {importances[i]:.4f}")
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig(r'plots\feature_importance_trained.png', bbox_inches='tight')
    plt.close()
    print("Feature importance plot saved.")


def plot_decision_tree(model, feature_names, class_names,
                       save_path=r"plots\decision_tree.png"):
    if not isinstance(model, DecisionTreeClassifier):
        print("Model is not a Decision Tree, cannot plot tree structure.")
        return
    plt.figure(figsize=(20, 10))
    plot_tree(model, filled=True, feature_names=feature_names,
              class_names=class_names, rounded=True, fontsize=10)
    plt.title("Decision Tree Structure")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Decision tree plot saved to {save_path}")
    plt.show()
    plt.close()


def plot_learning_curve(estimator, X, y, cv=5,
                        save_path=r"plots\learning_curve.png"):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores,  axis=1)
    test_mean  = np.mean(test_scores,  axis=1)
    test_std   = np.std(test_scores,   axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue',  label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1, color='blue')
    plt.plot(train_sizes, test_mean,  'o-', color='green', label='Cross-validation score')
    plt.fill_between(train_sizes, test_mean - test_std,
                     test_mean + test_std,  alpha=0.1, color='green')
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Learning curve saved to {save_path}")
    plt.show()
    plt.close()
