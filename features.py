import chess

from utils import (
    calculate_material,
    mobility,
    weighted_mobility,        
    num_attacked_pieces,
    pawn_structure,
    passed_pawn_count,      
    king_in_check,
    center_control,
    king_safety,
    bishop_pair_advantage,  
    open_file_rooks,        
    game_phase,
)


def extract_features(board):
    return {
        "material_diff":     calculate_material(board),
        "mobility":          mobility(board),
        "king_in_check":     king_in_check(board),
        "pawn_structure":    pawn_structure(board),
        "attacked_pieces":   num_attacked_pieces(board),
        "center_control":    center_control(board),
        "king_safety":       king_safety(board),
        "game_phase":        game_phase(board),
        "passed_pawns":      passed_pawn_count(board),
        "bishop_pair":       bishop_pair_advantage(board),
        "open_file_rooks":   open_file_rooks(board),
        "weighted_mobility": weighted_mobility(board),
    }
