from flask import Flask, render_template, request, jsonify
import chess
import joblib
import os
import sys

app = Flask(__name__)

# Load model and scaler at startup
model = None
scaler = None

def load_model():
    global model, scaler
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("WARNING: model.pkl or scaler.pkl not found. Run 'python main.py --mode train' first.")

def extract_features(board):
    from features import extract_features as _extract
    return _extract(board)

def predict(fen):
    board = chess.Board(fen)
    features = extract_features(board)
    from features import extract_features as _ef
    import pandas as pd
    FEATURE_COLUMNS = [
        "material_diff", "mobility", "king_in_check", "pawn_structure",
        "attacked_pieces", "center_control", "king_safety", "game_phase",
        "passed_pawns", "bishop_pair", "open_file_rooks", "weighted_mobility",
    ]
    row = pd.DataFrame([features])[FEATURE_COLUMNS]
    row_scaled = scaler.transform(row)
    return model.predict(row_scaled)[0]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()
    fen = data.get("fen", "").strip()
    if not fen:
        return jsonify({"error": "FEN is required"}), 400
    try:
        board = chess.Board(fen)
    except Exception:
        return jsonify({"error": "Invalid FEN string"}), 400
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 500
    try:
        result = predict(fen)
        return jsonify({"prediction": result, "fen": fen})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/board", methods=["POST"])
def get_board():
    data = request.get_json()
    fen = data.get("fen", "").strip()
    if not fen:
        fen = chess.STARTING_FEN
    try:
        board = chess.Board(fen)
        pieces = {}
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                pieces[square] = {
                    "type": piece.piece_type,
                    "color": "white" if piece.color == chess.WHITE else "black",
                    "symbol": piece.unicode_symbol()
                }
        return jsonify({"pieces": pieces, "valid": True})
    except Exception:
        return jsonify({"valid": False, "error": "Invalid FEN"})

if __name__ == "__main__":
    load_model()
    app.run(debug=True)
