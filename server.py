from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import random
import requests
from collections import Counter

app = Flask(__name__)
CORS(app)

# ============== Load Dictionary ==============

WORDS = set()
WORDS_BY_LEN = {}
WORDS_BY_LETTERS = {}  # Map sorted letters -> list of words (for anagram lookup)

dict_path = os.path.join(os.path.dirname(__file__), 'scrabble_words.txt')
print("Loading dictionary...")
with open(dict_path, 'r') as f:
    for line in f:
        word = line.strip().upper()
        if word and 2 <= len(word) <= 15:
            WORDS.add(word)
            L = len(word)
            if L not in WORDS_BY_LEN:
                WORDS_BY_LEN[L] = []
            WORDS_BY_LEN[L].append(word)
            
            # Index by sorted letters for faster lookup
            key = ''.join(sorted(word))
            if key not in WORDS_BY_LETTERS:
                WORDS_BY_LETTERS[key] = []
            WORDS_BY_LETTERS[key].append(word)

print(f"Loaded {len(WORDS)} words")

# Letter values and scores
LETTER_VALUES = {
    'A':1,'B':3,'C':3,'D':2,'E':1,'F':4,'G':2,'H':4,'I':1,'J':8,
    'K':5,'L':1,'M':3,'N':1,'O':1,'P':3,'Q':10,'R':1,'S':1,'T':1,
    'U':1,'V':4,'W':4,'X':8,'Y':4,'Z':10,'?':0
}

TW = {(0,0),(0,7),(0,14),(7,0),(7,14),(14,0),(14,7),(14,14)}
DW = {(1,1),(2,2),(3,3),(4,4),(1,13),(2,12),(3,11),(4,10),
      (13,1),(12,2),(11,3),(10,4),(13,13),(12,12),(11,11),(10,10),(7,7)}
TL = {(1,5),(1,9),(5,1),(5,5),(5,9),(5,13),(9,1),(9,5),(9,9),(9,13),(13,5),(13,9)}
DL = {(0,3),(0,11),(2,6),(2,8),(3,0),(3,7),(3,14),(6,2),(6,6),(6,8),(6,12),
      (7,3),(7,11),(8,2),(8,6),(8,8),(8,12),(11,0),(11,7),(11,14),(12,6),(12,8),(14,3),(14,11)}


def can_form_word(word, available_letters):
    """Check if word can be formed from available letters (including blanks)"""
    available = Counter(available_letters)
    blanks = available.get('?', 0)
    
    for letter in word:
        if available[letter] > 0:
            available[letter] -= 1
        elif blanks > 0:
            blanks -= 1
        else:
            return False
    return True


def score_word(word, positions, new_positions, blank_positions=set()):
    """Score a word with multipliers"""
    word_mult = 1
    letter_score = 0
    
    for i, (r, c) in enumerate(positions):
        letter = word[i]
        is_new = (r, c) in new_positions
        is_blank = (r, c) in blank_positions
        val = 0 if is_blank else LETTER_VALUES.get(letter, 0)
        
        if is_new:
            if (r, c) in TL:
                val *= 3
            elif (r, c) in DL:
                val *= 2
            if (r, c) in TW:
                word_mult *= 3
            elif (r, c) in DW:
                word_mult *= 2
        
        letter_score += val
    
    return letter_score * word_mult


def get_cross_word(board, row, col, dr, dc, new_letter):
    """Get perpendicular word formed at position"""
    pdr, pdc = (0, 1) if dr == 1 else (1, 0)
    
    # Find start of cross word
    sr, sc = row, col
    while sr - pdr >= 0 and sc - pdc >= 0 and board[sr - pdr][sc - pdc]:
        sr -= pdr
        sc -= pdc
    
    # Build word
    word = ""
    positions = []
    r, c = sr, sc
    while 0 <= r < 15 and 0 <= c < 15:
        if r == row and c == col:
            word += new_letter
            positions.append((r, c))
        elif board[r][c]:
            word += board[r][c]
            positions.append((r, c))
        else:
            break
        r += pdr
        c += pdc
    
    return (word, positions) if len(word) > 1 else None


def get_letters_in_line(board, anchor_r, anchor_c, dr, dc):
    """Get all letters currently in a line through the anchor"""
    letters = []
    # Go backwards
    r, c = anchor_r - dr, anchor_c - dc
    while 0 <= r < 15 and 0 <= c < 15 and board[r][c]:
        letters.append(board[r][c])
        r -= dr
        c -= dc
    # Go forwards
    r, c = anchor_r + dr, anchor_c + dc
    while 0 <= r < 15 and 0 <= c < 15 and board[r][c]:
        letters.append(board[r][c])
        r += dr
        c += dc
    return letters


def find_all_moves(board, rack):
    """Find all valid moves - optimized AI"""
    moves = []
    rack_list = list(rack.upper())
    rack_set = set(rack_list)
    has_blank = '?' in rack_list
    
    is_first = all(board[r][c] is None for r in range(15) for c in range(15))
    
    # Find anchors
    anchors = set()
    if is_first:
        anchors.add((7, 7))
    else:
        for r in range(15):
            for c in range(15):
                if board[r][c] is None:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 15 and 0 <= nc < 15 and board[nr][nc]:
                            anchors.add((r, c))
                            break
    
    # Pre-filter words that could possibly be formed with rack
    # A word is possible if all its letters come from rack + board
    board_letters = [board[r][c] for r in range(15) for c in range(15) if board[r][c]]
    all_available = rack_list + board_letters
    
    possible_words = set()
    for word in WORDS:
        if len(word) <= len(rack_list) + 8:  # Can't be longer than rack + reasonable board use
            if can_form_word(word, all_available):
                possible_words.add(word)
    
    print(f"Filtered to {len(possible_words)} possible words from {len(WORDS)}")
    
    for anchor_r, anchor_c in anchors:
        for dr, dc in [(0, 1), (1, 0)]:
            direction = 'H' if dc == 1 else 'V'
            
            # Get board letters in this line for this anchor
            line_letters = get_letters_in_line(board, anchor_r, anchor_c, dr, dc)
            available_for_line = rack_list + line_letters
            
            # How far back can we start?
            max_prefix = 0
            r, c = anchor_r - dr, anchor_c - dc
            while r >= 0 and c >= 0 and board[r][c] is None and max_prefix < len(rack_list):
                max_prefix += 1
                r -= dr
                c -= dc
            
            for word in possible_words:
                # Quick filter: word must be formable with rack + line letters
                if not can_form_word(word, available_for_line):
                    continue
                
                for offset in range(min(len(word), max_prefix + 1)):
                    start_r = anchor_r - offset * dr
                    start_c = anchor_c - offset * dc
                    
                    if start_r < 0 or start_c < 0:
                        continue
                    end_r = start_r + (len(word) - 1) * dr
                    end_c = start_c + (len(word) - 1) * dc
                    if end_r >= 15 or end_c >= 15:
                        continue
                    
                    # Try to place word
                    temp_rack = rack_list.copy()
                    valid = True
                    tiles_to_place = []
                    positions = []
                    new_positions = set()
                    blank_positions = set()
                    covers_anchor = False
                    uses_rack = False
                    
                    for i, letter in enumerate(word):
                        r = start_r + i * dr
                        c = start_c + i * dc
                        positions.append((r, c))
                        
                        if r == anchor_r and c == anchor_c:
                            covers_anchor = True
                        
                        if board[r][c]:
                            if board[r][c] != letter:
                                valid = False
                                break
                        else:
                            if letter in temp_rack:
                                temp_rack.remove(letter)
                                tiles_to_place.append({'row': r, 'col': c, 'letter': letter, 'is_blank': False})
                                new_positions.add((r, c))
                                uses_rack = True
                            elif '?' in temp_rack:
                                temp_rack.remove('?')
                                tiles_to_place.append({'row': r, 'col': c, 'letter': letter, 'is_blank': True})
                                new_positions.add((r, c))
                                blank_positions.add((r, c))
                                uses_rack = True
                            else:
                                valid = False
                                break
                    
                    if not valid or not covers_anchor or not uses_rack:
                        continue
                    
                    if is_first and (7, 7) not in positions:
                        continue
                    
                    # Validate cross words
                    cross_valid = True
                    total_cross_score = 0
                    
                    for tile in tiles_to_place:
                        cross = get_cross_word(board, tile['row'], tile['col'], dr, dc, tile['letter'])
                        if cross:
                            cword, cpos = cross
                            if cword not in WORDS:
                                cross_valid = False
                                break
                            cross_new = {(tile['row'], tile['col'])}
                            cross_blank = {(tile['row'], tile['col'])} if tile['is_blank'] else set()
                            total_cross_score += score_word(cword, cpos, cross_new, cross_blank)
                    
                    if not cross_valid:
                        continue
                    
                    main_score = score_word(word, positions, new_positions, blank_positions)
                    total_score = main_score + total_cross_score
                    
                    if len(tiles_to_place) == 7:
                        total_score += 50
                    
                    moves.append({
                        'word': word,
                        'start_row': start_r,
                        'start_col': start_c,
                        'direction': direction,
                        'tiles': tiles_to_place,
                        'score': total_score
                    })
    
    return moves


def get_ai_move(board, rack, difficulty='hard'):
    """Get AI move based on difficulty"""
    moves = find_all_moves(board, rack)
    
    if not moves:
        return None
    
    # Sort by score descending
    moves.sort(key=lambda m: m['score'], reverse=True)
    
    # Remove duplicates
    seen = set()
    unique_moves = []
    for m in moves:
        key = (m['word'], m['start_row'], m['start_col'], m['direction'])
        if key not in seen:
            seen.add(key)
            unique_moves.append(m)
    
    if not unique_moves:
        return None
    
    print(f"Found {len(unique_moves)} unique moves. Top 5:")
    for m in unique_moves[:5]:
        print(f"  {m['word']}: {m['score']} pts")
    
    if difficulty == 'easy':
        # Pick from bottom 40%
        start_idx = int(len(unique_moves) * 0.6)
        idx = random.randint(start_idx, len(unique_moves) - 1) if start_idx < len(unique_moves) else 0
    elif difficulty == 'medium':
        # Pick from top 50% but not the best
        end_idx = max(1, int(len(unique_moves) * 0.5))
        idx = random.randint(1, end_idx) if end_idx > 1 else 0
    else:
        # Hard - best move
        idx = 0
    
    return unique_moves[idx]


# Claude API for definitions and hints
CLAUDE_API_KEY = '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'

@app.route('/api/define', methods=['POST'])
def define_word():
    """Get word definition from Claude"""
    data = request.json
    word = data.get('word', '').upper()
    
    try:
        response = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers={
                'Content-Type': 'application/json',
                'x-api-key': CLAUDE_API_KEY,
                'anthropic-version': '2023-06-01'
            },
            json={
                'model': 'claude-sonnet-4-20250514',
                'max_tokens': 200,
                'messages': [{
                    'role': 'user',
                    'content': f'Define the word "{word}" briefly in 1-2 sentences. If it\'s a valid Scrabble word but obscure, explain what it means. Just give the definition, no preamble.'
                }]
            },
            timeout=10
        )
        result = response.json()
        definition = result.get('content', [{}])[0].get('text', 'Definition not found')
        return jsonify({'success': True, 'definition': definition})
    except Exception as e:
        print(f"Define error: {e}")
        return jsonify({'success': False, 'definition': 'Could not fetch definition'})


@app.route('/api/funny_hint', methods=['POST'])
def funny_hint():
    """Get funny hint from Claude without revealing the word"""
    data = request.json
    word = data.get('word', '')
    rack = data.get('rack', [])
    
    try:
        response = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers={
                'Content-Type': 'application/json',
                'x-api-key': CLAUDE_API_KEY,
                'anthropic-version': '2023-06-01'
            },
            json={
                'model': 'claude-sonnet-4-20250514',
                'max_tokens': 150,
                'messages': [{
                    'role': 'user',
                    'content': f'You\'re a witty Scrabble coach. The player has these letters: {", ".join(rack)}. The best word they can play is "{word}" but DON\'T say the word! Give them a clever, funny hint or clue to help them figure it out. Be playful and brief (1-2 sentences). Don\'t use the word itself or obvious anagrams.'
                }]
            },
            timeout=10
        )
        result = response.json()
        hint = result.get('content', [{}])[0].get('text', 'Hmm, I\'m stumped too!')
        return jsonify({'success': True, 'hint': hint})
    except Exception as e:
        print(f"Hint error: {e}")
        return jsonify({'success': False, 'hint': 'My wit seems to have wandered off...'})


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
    
    board = [[None]*15 for _ in range(15)]
    for tile in board_data:
        r, c = tile['row'], tile['col']
        board[r][c] = tile['letter'].upper()
    
    rack_str = ''.join(rack).upper()
    
    move = get_ai_move(board, rack_str, difficulty)
    
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
    return ai_move()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
