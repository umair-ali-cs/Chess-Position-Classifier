import chess

piece_values = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
}

def calculate_material(board):
    material = 0
    for piece_type in piece_values:
        material += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        material -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
    return material

def total_material(board):
    total = 0
    for piece_type in piece_values:
        total += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        total += len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
    return total

def mobility(board):
    board_copy = board.copy()
    board_copy.turn = chess.WHITE
    white_m = board_copy.legal_moves.count()
    board_copy.turn = chess.BLACK
    black_m = board_copy.legal_moves.count()
    return white_m - black_m

def king_in_check(board):
    board_w = board.copy(); board_w.turn = chess.WHITE
    board_b = board.copy(); board_b.turn = chess.BLACK
    white_check = int(board_w.is_check())
    black_check = int(board_b.is_check())
    return white_check - black_check

def pawn_structure(board):
    def count_weaknesses(color):
        pawns = board.pieces(chess.PAWN, color)
        opp_pawns = board.pieces(chess.PAWN, not color)
        files = [chess.square_file(sq) for sq in pawns]

        doubled = sum(files.count(f) - 1 for f in set(files) if files.count(f) > 1)
        isolated = sum(1 for f in set(files) if (f-1) not in files and (f+1) not in files)

        passed = 0
        for sq in pawns:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            adjacent_files = {f-1, f, f+1} & set(range(8))
            if color == chess.WHITE:
                blocking = [s for s in opp_pawns if chess.square_file(s) in adjacent_files and chess.square_rank(s) > r]
            else:
                blocking = [s for s in opp_pawns if chess.square_file(s) in adjacent_files and chess.square_rank(s) < r]
            if not blocking:
                passed += 1

        return doubled + isolated - passed

    return count_weaknesses(chess.WHITE) - count_weaknesses(chess.BLACK)

def num_attacked_pieces(board):
    def attacked_value(color):
        attacker = not color
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                if board.is_attacked_by(attacker, square):
                    score += piece_values.get(piece.piece_type, 0)
        return score
    return attacked_value(chess.WHITE) - attacked_value(chess.BLACK)

def king_safety(board):
    def shield_score(color):
        king_sq = board.king(color)
        if king_sq is None:
            return 0
        king_file = chess.square_file(king_sq)
        king_rank = chess.square_rank(king_sq)
        direction = 1 if color == chess.WHITE else -1
        shield = 0
        for df in [-1, 0, 1]:
            f = king_file + df
            r = king_rank + direction
            if 0 <= f <= 7 and 0 <= r <= 7:
                sq = chess.square(f, r)
                p = board.piece_at(sq)
                if p and p.piece_type == chess.PAWN and p.color == color:
                    shield += 1
        return shield
    return shield_score(chess.WHITE) - shield_score(chess.BLACK)

def piece_count(board):
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    white = sum(len(board.pieces(pt, chess.WHITE)) for pt in piece_types)
    black = sum(len(board.pieces(pt, chess.BLACK)) for pt in piece_types)
    return white, black

def game_phase(board):
    total = total_material(board)
    max_material = 78
    return 1.0 - min(total / max_material, 1.0)

CENTER_SQUARES = [chess.D4, chess.D5, chess.E4, chess.E5]
def center_control(board):
    score = 0
    for sq in CENTER_SQUARES:
        score += len(board.attackers(chess.WHITE, sq))
        score -= len(board.attackers(chess.BLACK, sq))
    return score

def passed_pawn_count(board):
    def count_passed(color):
        pawns = board.pieces(chess.PAWN, color)
        opp_pawns = board.pieces(chess.PAWN, not color)
        count = 0
        for sq in pawns:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            adjacent_files = {max(0, f - 1), f, min(7, f + 1)}
            if color == chess.WHITE:
                blocking = [s for s in opp_pawns
                            if chess.square_file(s) in adjacent_files
                            and chess.square_rank(s) > r]
            else:
                blocking = [s for s in opp_pawns
                            if chess.square_file(s) in adjacent_files
                            and chess.square_rank(s) < r]
            if not blocking:
                count += 1
        return count
    return count_passed(chess.WHITE) - count_passed(chess.BLACK)

def bishop_pair_advantage(board):
    white_pair = int(len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2)
    black_pair = int(len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2)
    return white_pair - black_pair

def open_file_rooks(board):
    white_pawn_files = {chess.square_file(sq) for sq in board.pieces(chess.PAWN, chess.WHITE)}
    black_pawn_files = {chess.square_file(sq) for sq in board.pieces(chess.PAWN, chess.BLACK)}

    def rook_score(color):
        own_files   = white_pawn_files if color == chess.WHITE else black_pawn_files
        opp_files   = black_pawn_files if color == chess.WHITE else white_pawn_files
        score = 0.0
        for sq in board.pieces(chess.ROOK, color):
            f = chess.square_file(sq)
            if f not in own_files and f not in opp_files:
                score += 1.0   
            elif f not in own_files:
                score += 0.5   
        return score

    return rook_score(chess.WHITE) - rook_score(chess.BLACK)

def weighted_mobility(board):
    piece_move_weight = {
        chess.PAWN:   0.5,
        chess.KNIGHT: 2.0,
        chess.BISHOP: 1.5,
        chess.ROOK:   1.0,
        chess.QUEEN:  0.75, 
        chess.KING:   0.25,
    }

    def score_for(color):
        total = 0.0
        board_copy = board.copy()
        board_copy.turn = color
        for move in board_copy.legal_moves:
            piece = board_copy.piece_at(move.from_square)
            if piece:
                total += piece_move_weight.get(piece.piece_type, 1.0)
        return total

    return score_for(chess.WHITE) - score_for(chess.BLACK)

def extract_move_features(board, move):
    piece = board.piece_at(move.from_square)
    return {
        "piece_type":        piece.piece_type if piece else 0,
        "is_capture":        int(board.is_capture(move)),
        "captured_value":    piece_values.get(
                                board.piece_at(move.to_square).piece_type, 0
                             ) if board.is_capture(move) else 0,
        "gives_check":       int(board.gives_check(move)),
        "is_promotion":      int(move.promotion is not None),
        "move_distance":     chess.square_distance(move.from_square, move.to_square),
        "to_center":         int(move.to_square in [chess.D4, chess.D5, chess.E4, chess.E5]),
        "moves_into_attack": int(board.is_attacked_by(not board.turn, move.to_square)),
        "escapes_attack":    int(board.is_attacked_by(not board.turn, move.from_square)),
    }
