from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
from itertools import permutations
import random

app = Flask(__name__)
CORS(app)

# ============== Load Dictionary ==============

WORDS = set()
WORDS_BY_LENGTH = {}  # words grouped by length for faster lookup

dict_path = os.path.join(os.path.dirname(__file__), 'scrabble_words.txt')
print("Loading dictionary...")
with open(dict_path, 'r') as f:
    for line in f:
        word = line.strip().upper()
        if word and len(word) >= 2:
            WORDS.add(word)
            length = len(word)
            if length not in WORDS_BY_LENGTH:
                WORDS_BY_LENGTH[length] = set()
            WORDS_BY_LENGTH[length].add(word)

print(f"Loaded {len(WORDS)} words")

# Letter values
LETTER_VALUES = {
    'A':1,'B':3,'C':3,'D':2,'E':1,'F':4,'G':2,'H':4,'I':1,'J':8,
    'K':5,'L':1,'M':3,'N':1,'O':1,'P':3,'Q':10,'R':1,'S':1,'T':1,
    'U':1,'V':4,'W':4,'X':8,'Y':4,'Z':10,'?':0
}

# Bonus squares
TW = {(0,0),(0,7),(0,14),(7,0),(7,14),(14,0),(14,7),(14,14)}
DW = {(1,1),(2,2),(3,3),(4,4),(1,13),(2,12),(3,11),(4,10),
      (13,1),(12,2),(11,3),(10,4),(13,13),(12,12),(11,11),(10,10),(7,7)}
TL = {(1,5),(1,9),(5,1),(5,5),(5,9),(5,13),(9,1),(9,5),(9,9),(9,13),(13,5),(13,9)}
DL = {(0,3),(0,11),(2,6),(2,8),(3,0),(3,7),(3,14),(6,2),(6,6),(6,8),(6,12),
      (7,3),(7,11),(8,2),(8,6),(8,8),(8,12),(11,0),(11,7),(11,14),(12,6),(12,8),(14,3),(14,11)}


def calculate_word_score(word, positions, new_positions):
    """Calculate score for a word given letter positions"""
    word_multiplier = 1
    letter_score = 0
    
    for i, (row, col) in enumerate(positions):
        letter = word[i]
        is_new = (row, col) in new_positions
        base_value = LETTER_VALUES.get(letter, 0)
        
        if is_new:
            if (row, col) in TL:
                base_value *= 3
            elif (row, col) in DL:
                base_value *= 2
            if (row, col) in TW:
                word_multiplier *= 3
            elif (row, col) in DW:
                word_multiplier *= 2
        
        letter_score += base_value
    
    total = letter_score * word_multiplier
    if len(new_positions) == 7:
        total += 50
    return total


def get_cross_word(board, row, col, dr, dc):
    """Get the cross word at position perpendicular to direction"""
    # Perpendicular direction
    pdr, pdc = dc, dr
    
    # Find start of cross word
    sr, sc = row, col
    while 0 <= sr - pdr < 15 and 0 <= sc - pdc < 15:
        if board[sr - pdr][sc - pdc] is None:
            break
        sr -= pdr
        sc -= pdc
    
    # Build cross word
    word = ""
    r, c = sr, sc
    while 0 <= r < 15 and 0 <= c < 15 and (board[r][c] is not None or (r == row and c == col)):
        if board[r][c] is not None:
            word += board[r][c]
        elif r == row and c == col:
            word += "?"  # Placeholder for new tile
        else:
            break
        r += pdr
        c += pdc
    
    return word if len(word) > 1 else ""


def find_ai_moves(board, rack, difficulty='hard'):
    """Find valid moves for AI"""
    moves = []
    rack_letters = list(rack.upper())
    
    # Check if board is empty (first move)
    is_first_move = all(board[r][c] is None for r in range(15) for c in range(15))
    
    if is_first_move:
        # First move must go through center
        anchors = [(7, 7, True)]  # row, col, is_anchor_start
    else:
        # Find anchor squares
        anchors = []
        for r in range(15):
            for c in range(15):
                if board[r][c] is None:
                    # Check adjacent
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 15 and 0 <= nc < 15 and board[nr][nc] is not None:
                            anchors.append((r, c, True))
                            break
    
    # For each anchor, try placing words
    for anchor_row, anchor_col, _ in anchors[:20]:  # Limit anchors for performance
        # Try horizontal and vertical
        for dr, dc in [(0, 1), (1, 0)]:
            direction = 'H' if dc == 1 else 'V'
            
            # Try different word lengths
            for word_len in range(2, min(8, len(rack_letters) + 1)):
                if word_len not in WORDS_BY_LENGTH:
                    continue
                
                # Try words that could pass through this anchor
                # Generate candidate words from rack
                for perm in permutations(rack_letters, min(word_len, len(rack_letters))):
                    candidate = ''.join(perm)
                    
                    # Check if any word starts with or contains this
                    for word in WORDS_BY_LENGTH.get(len(candidate), set()):
                        if set(word) <= set(rack_letters + [board[r][c] for r in range(15) for c in range(15) if board[r][c]]):
                            # Try to place this word at anchor
                            for offset in range(len(word)):
                                start_r = anchor_row - (offset * dr)
                                start_c = anchor_col - (offset * dc)
                                
                                if start_r < 0 or start_c < 0:
                                    continue
                                if start_r + (len(word) - 1) * dr >= 15:
                                    continue
                                if start_c + (len(word) - 1) * dc >= 15:
                                    continue
                                
                                # Check if word fits
                                valid = True
                                tiles_needed = []
                                positions = []
                                new_positions = set()
                                temp_rack = rack_letters.copy()
                                
                                for i, letter in enumerate(word):
                                    r = start_r + i * dr
                                    c = start_c + i * dc
                                    positions.append((r, c))
                                    
                                    if board[r][c] is not None:
                                        if board[r][c] != letter:
                                            valid = False
                                            break
                                    else:
                                        if letter in temp_rack:
                                            temp_rack.remove(letter)
                                            tiles_needed.append({'row': r, 'col': c, 'letter': letter, 'is_blank': False})
                                            new_positions.add((r, c))
                                        elif '?' in temp_rack:
                                            temp_rack.remove('?')
                                            tiles_needed.append({'row': r, 'col': c, 'letter': letter, 'is_blank': True})
                                            new_positions.add((r, c))
                                        else:
                                            valid = False
                                            break
                                
                                if valid and len(tiles_needed) > 0:
                                    # Must connect to existing tiles (unless first move)
                                    if not is_first_move:
                                        connects = False
                                        for r, c in positions:
                                            if board[r][c] is not None:
                                                connects = True
                                                break
                                            for adjr, adjc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                                                if 0 <= adjr < 15 and 0 <= adjc < 15:
                                                    if board[adjr][adjc] is not None and (adjr, adjc) not in positions:
                                                        connects = True
                                                        break
                                        if not connects:
                                            continue
                                    else:
                                        # First move must include center
                                        if (7, 7) not in positions:
                                            continue
                                    
                                    score = calculate_word_score(word, positions, new_positions)
                                    moves.append({
                                        'word': word,
                                        'start_row': start_r,
                                        'start_col': start_c,
                                        'direction': direction,
                                        'tiles': tiles_needed,
                                        'score': score
                                    })
                    
                    if len(moves) > 100:  # Limit for performance
                        break
                if len(moves) > 100:
                    break
            if len(moves) > 100:
                break
        if len(moves) > 100:
            break
    
    # Sort by score
    moves.sort(key=lambda m: m['score'], reverse=True)
    
    # Based on difficulty, pick move
    if not moves:
        return None
    
    if difficulty == 'easy':
        # Pick random from bottom half
        idx = random.randint(len(moves)//2, len(moves)-1) if len(moves) > 1 else 0
        return moves[idx]
    elif difficulty == 'medium':
        # Pick from top 30%
        idx = random.randint(0, max(0, len(moves)//3))
        return moves[idx]
    else:
        # Hard - pick best
        return moves[0]


# ============== Routes ==============

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/validate', methods=['POST'])
def validate_word():
    data = request.json
    word = data.get('word', '').upper()
    return jsonify({'word': word, 'valid': word in WORDS})

@app.route('/api/validate_words', methods=['POST'])
def validate_words():
    data = request.json
    words = data.get('words', [])
    results = {w: w.upper() in WORDS for w in words}
    return jsonify({'results': results, 'all_valid': all(results.values())})

@app.route('/api/check', methods=['GET'])
def check_word():
    word = request.args.get('word', '').upper()
    return jsonify({'word': word, 'valid': word in WORDS})

@app.route('/api/stats')
def stats():
    return jsonify({'word_count': len(WORDS), 'letter_values': LETTER_VALUES})

@app.route('/api/ai_move', methods=['POST'])
def ai_move():
    """Find AI move given board and rack"""
    data = request.json
    board_data = data.get('board', [])
    rack = data.get('rack', [])
    difficulty = data.get('difficulty', 'hard')
    
    # Convert board format: list of {row, col, letter} to 2D array
    board = [[None]*15 for _ in range(15)]
    for tile in board_data:
        r, c = tile['row'], tile['col']
        board[r][c] = tile['letter'].upper()
    
    rack_str = ''.join(rack).upper()
    
    # Find best move
    move = find_ai_moves(board, rack_str, difficulty)
    
    if move:
        return jsonify({
            'success': True,
            'word': move['word'],
            'start_row': move['start_row'],
            'start_col': move['start_col'],
            'direction': move['direction'],
            'tiles': move['tiles'],
            'score': move['score']
        })
    else:
        return jsonify({'success': False, 'message': 'No valid moves found'})

@app.route('/api/hint', methods=['POST'])
def hint():
    """Give player a hint"""
    return ai_move()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
