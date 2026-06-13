# prepare_dataset.py
import pandas as pd
import chess
from features import extract_features

def label_from_eval(fen, eval_score, thresholds=(1.0, 3.0),
                    difficulty_mode="position_hard"):  
    board = chess.Board(fen)
    stm_eval = eval_score if board.turn == chess.WHITE else -eval_score
    advantage = abs(stm_eval)

    if difficulty_mode == "position_hard":
        if advantage > thresholds[1]:
            return "Easy"
        elif advantage > thresholds[0]:
            return "Medium"
        else:
            return "Hard"

    elif difficulty_mode == "tactical_hard":
        if advantage > thresholds[1]:
            return "Hard"
        elif advantage > thresholds[0]:
            return "Medium"
        else:
            return "Easy"

    else:
        raise ValueError(f"Unknown difficulty_mode: '{difficulty_mode}'. "
                         f"Choose 'position_hard' or 'tactical_hard'.")


def main():
    input_csv  = "chess_dataset.csv"
    output_csv = "chess_dataset_labeled.csv"

    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} positions.")

    required = {"fen", "eval"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns: {missing}")

    print("Extracting features...")
    features_list = []
    for fen in df["fen"]:
        board = chess.Board(fen)
        features_list.append(extract_features(board))
    features_df = pd.DataFrame(features_list)

    print("Assigning labels (mode=position_hard)...")
    features_df["label"] = df.apply(
        lambda row: label_from_eval(row["fen"], row["eval"],
                                    difficulty_mode="position_hard"),
        axis=1
    )

    features_df["fen"]  = df["fen"]
    features_df["eval"] = df["eval"]

    features_df.to_csv(output_csv, index=False)
    print(f"\nSaved labeled dataset to {output_csv}")
    print("Class distribution:")
    print(features_df["label"].value_counts())

    total = len(features_df)
    for label, count in features_df["label"].value_counts().items():
        pct = count / total * 100
        if pct < 10:
            print(f"  ⚠ WARNING: '{label}' has only {pct:.1f}% of samples — "
                  f"consider SMOTE or class_weight='balanced'.")

    print("\nSample rows:")
    print(features_df[["fen", "eval", "label"]].head(10))


if __name__ == "__main__":
    main()
