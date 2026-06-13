# main.py
import argparse
import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from model import (
    load_dataset,
    prepare_data,
    tune_knn,
    tune_decision_tree,
    tune_random_forest,
    tune_lgbm,
    evaluate_model,
    save_model,
    load_model,
    predict_position,
    plot_feature_importance,
    plot_learning_curve,
    DecisionTreeClassifier,
    LGBM_AVAILABLE,
)
from features import extract_features
import chess


def ensure_plots_dir():
    if not os.path.exists("plots"):
        os.makedirs("plots")
        print("Created 'plots' directory.")


def train_pipeline(csv_path):
    ensure_plots_dir()

    print(f"Loading dataset from {csv_path}...")
    X, y, feature_cols = load_dataset(csv_path)
    print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features")
    print("Class distribution:")
    print(y.value_counts())

    print("\nPreparing data (train/test split + scaling)...")
    X_train, X_test, y_train, y_test, scaler = prepare_data(X, y)

    # ── train and evaluate all models ──────────────────────────────────────
    print("\nTuning KNN...")
    best_knn = tune_knn(X_train, y_train)
    _, f1_knn = evaluate_model(best_knn, X_test, y_test, "Tuned KNN",
                               plot_cm=True, save_cm_path="plots/knn_cm.png")

    print("\nTuning Decision Tree...")
    best_dt = tune_decision_tree(X_train, y_train)
    _, f1_dt = evaluate_model(best_dt, X_test, y_test, "Tuned DT",
                              plot_cm=True, save_cm_path="plots/dt_cm.png")

    print("\nTuning Random Forest...")
    best_rf = tune_random_forest(X_train, y_train)
    _, f1_rf = evaluate_model(best_rf, X_test, y_test, "Tuned RF",
                              plot_cm=True, save_cm_path="plots/rf_cm.png")

    # LightGBM (if available)
    best_lgbm = None
    f1_lgbm = 0.0
    if LGBM_AVAILABLE:
        print("\nTuning LightGBM...")
        best_lgbm = tune_lgbm(X_train, y_train)
        if best_lgbm is not None:
            _, f1_lgbm = evaluate_model(best_lgbm, X_test, y_test, "Tuned LightGBM",
                                        plot_cm=True, save_cm_path="plots/lgbm_cm.png")

    # ── select best model by F1 macro ──────────────────────────────────────
    candidates = {
        "Random Forest": (best_rf, f1_rf),
        "Decision Tree": (best_dt, f1_dt),
        "KNN": (best_knn, f1_knn),
    }
    if best_lgbm is not None:
        candidates["LightGBM"] = (best_lgbm, f1_lgbm)

    best_name = max(candidates, key=lambda k: candidates[k][1])
    best_model, best_f1 = candidates[best_name]
    print(f"\n✓ {best_name} selected (F1 macro = {best_f1:.4f})")

    save_model(best_model, scaler)

    if hasattr(best_model, 'feature_importances_'):
        plot_feature_importance(best_model, feature_cols)
        if isinstance(best_model, DecisionTreeClassifier):
            from model import plot_decision_tree
            plot_decision_tree(best_model, feature_cols,
                               ['Easy', 'Medium', 'Hard'],
                               save_path="plots/dt_tree.png")

    print("\nPlotting learning curve (uses raw features + pipeline)...")
    lc_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', best_model),
    ])
    plot_learning_curve(lc_pipeline, X, y, cv=5,
                        save_path="plots/lc.png")

    print("\nTraining complete. Model saved as 'model.pkl', scaler as 'scaler.pkl'.")


def predict_single_fen(fen, model, scaler):
    """Predict difficulty for a single FEN string."""
    try:
        board = chess.Board(fen)
    except Exception:
        raise ValueError("Invalid FEN string")
    features = extract_features(board)
    prediction = predict_position(features, model, scaler)
    return prediction


def predict_mode(args):
    """Load model and scaler, then predict difficulty for given FEN(s)."""
    try:
        model, scaler = load_model()
    except FileNotFoundError:
        print("Error: Model files not found. Please run training first (main.py --mode train).")
        sys.exit(1)

    if args.interactive:
        print("Chess Difficulty Predictor - Interactive Mode")
        print("Enter FEN strings (or 'quit' to exit):")
        while True:
            fen = input("FEN> ").strip()
            if fen.lower() in ("quit", "exit", "q"):
                break
            if not fen:
                print("Please enter a valid FEN.")
                continue
            try:
                pred = predict_single_fen(fen, model, scaler)
                print(f"Prediction: {pred}")
            except ValueError as e:
                print(f"Error: {e}")
    else:
        if args.fen_file:
            try:
                with open(args.fen_file, 'r') as f:
                    fens = [line.strip() for line in f if line.strip()]
            except Exception as e:
                print(f"Error reading file: {e}")
                sys.exit(1)
            for fen in fens:
                try:
                    pred = predict_single_fen(fen, model, scaler)
                    print(f"{fen} => {pred}")
                except ValueError as e:
                    print(f"{fen} => Error: {e}")
        elif args.fen:
            try:
                pred = predict_single_fen(args.fen, model, scaler)
                print(f"Prediction: {pred}")
            except ValueError as e:
                print(f"Error: {e}")
                sys.exit(1)
        else:
            print("Please provide --fen or --fen-file (or --interactive).")


def main():
    parser = argparse.ArgumentParser(description="Chess Difficulty Predictor")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "predict"],
                        help="Mode: train the model or predict difficulty")
    parser.add_argument("--csv", type=str, default="chess_dataset.csv",
                        help="Path to labeled CSV (for training)")
    parser.add_argument("--fen", type=str, help="Single FEN string to predict")
    parser.add_argument("--fen-file", type=str, help="File containing one FEN per line")
    parser.add_argument("--interactive", action="store_true",
                        help="Run interactive prediction loop")
    args = parser.parse_args()

    if args.mode == "train":
        train_pipeline(args.csv)
    else:
        predict_mode(args)


if __name__ == "__main__":
    main()