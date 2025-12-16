from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load dictionary once at startup
WORDS = set()
dict_path = os.path.join(os.path.dirname(__file__), 'scrabble_words.txt')
with open(dict_path, 'r') as f:
    WORDS = set(word.strip().upper() for word in f)

print(f"Loaded {len(WORDS)} words")

# Letter values
LETTER_VALUES = {
    'A':1,'B':3,'C':3,'D':2,'E':1,'F':4,'G':2,'H':4,'I':1,'J':8,
    'K':5,'L':1,'M':3,'N':1,'O':1,'P':3,'Q':10,'R':1,'S':1,'T':1,
    'U':1,'V':4,'W':4,'X':8,'Y':4,'Z':10,'?':0
}

# Tile distribution
TILE_DISTRIBUTION = {
    'A':9,'B':2,'C':2,'D':4,'E':12,'F':2,'G':3,'H':2,'I':9,'J':1,
    'K':1,'L':4,'M':2,'N':6,'O':8,'P':2,'Q':1,'R':6,'S':4,'T':6,
    'U':4,'V':2,'W':2,'X':1,'Y':2,'Z':1,'?':2
}

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/validate', methods=['POST'])
def validate_word():
    """Validate a single word"""
    data = request.json
    word = data.get('word', '').upper()
    is_valid = word in WORDS
    return jsonify({
        'word': word,
        'valid': is_valid
    })

@app.route('/api/validate_words', methods=['POST'])
def validate_words():
    """Validate multiple words at once"""
    data = request.json
    words = data.get('words', [])
    results = {}
    all_valid = True
    for word in words:
        w = word.upper()
        valid = w in WORDS
        results[word] = valid
        if not valid:
            all_valid = False
    return jsonify({
        'results': results,
        'all_valid': all_valid
    })

@app.route('/api/score', methods=['POST'])
def calculate_score():
    """Calculate score for a word with position bonuses"""
    data = request.json
    word = data.get('word', '').upper()
    positions = data.get('positions', [])  # List of {letter, row, col, is_blank}
    
    # Board bonus positions
    TW = [(0,0),(0,7),(0,14),(7,0),(7,14),(14,0),(14,7),(14,14)]
    DW = [(1,1),(2,2),(3,3),(4,4),(1,13),(2,12),(3,11),(4,10),
          (13,1),(12,2),(11,3),(10,4),(13,13),(12,12),(11,11),(10,10),(7,7)]
    TL = [(1,5),(1,9),(5,1),(5,5),(5,9),(5,13),(9,1),(9,5),(9,9),(9,13),(13,5),(13,9)]
    DL = [(0,3),(0,11),(2,6),(2,8),(3,0),(3,7),(3,14),(6,2),(6,6),(6,8),(6,12),
          (7,3),(7,11),(8,2),(8,6),(8,8),(8,12),(11,0),(11,7),(11,14),(12,6),(12,8),(14,3),(14,11)]
    
    word_multiplier = 1
    letter_score = 0
    
    for pos in positions:
        letter = pos.get('letter', '').upper()
        row = pos.get('row', 0)
        col = pos.get('col', 0)
        is_blank = pos.get('is_blank', False)
        is_new = pos.get('is_new', True)  # Only new tiles get bonuses
        
        # Blank tiles = 0 points
        base_value = 0 if is_blank else LETTER_VALUES.get(letter, 0)
        
        if is_new:
            # Apply letter multipliers
            if (row, col) in TL:
                base_value *= 3
            elif (row, col) in DL:
                base_value *= 2
            
            # Track word multipliers
            if (row, col) in TW:
                word_multiplier *= 3
            elif (row, col) in DW:
                word_multiplier *= 2
        
        letter_score += base_value
    
    total = letter_score * word_multiplier
    
    # Bingo bonus (all 7 tiles)
    tiles_played = sum(1 for p in positions if p.get('is_new', True))
    if tiles_played == 7:
        total += 50
    
    return jsonify({
        'word': word,
        'score': total,
        'word_multiplier': word_multiplier,
        'bingo': tiles_played == 7
    })

@app.route('/api/check', methods=['GET'])
def check_word():
    """Quick word check via GET"""
    word = request.args.get('word', '').upper()
    return jsonify({
        'word': word,
        'valid': word in WORDS
    })

@app.route('/api/stats')
def stats():
    """Dictionary stats"""
    return jsonify({
        'word_count': len(WORDS),
        'letter_values': LETTER_VALUES,
        'tile_distribution': TILE_DISTRIBUTION
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
