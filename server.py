from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import os
import random
import requests
import string
from collections import Counter

app = Flask(__name__)
CORS(app)
# Let Flask-SocketIO auto-detect best async mode (will use simple-websocket)
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=60, ping_interval=25)

# ============== Game Rooms ==============
# Stores active game rooms: room_code -> game_state
game_rooms = {}

def generate_room_code():
    """Generate 4-letter room code"""
    while True:
        code = ''.join(random.choices(string.ascii_uppercase, k=4))
        if code not in game_rooms:
            return code

def create_tile_bag():
    """Create standard Scrabble tile bag"""
    distribution = {
        'A': 9, 'B': 2, 'C': 2, 'D': 4, 'E': 12, 'F': 2, 'G': 3, 'H': 2,
        'I': 9, 'J': 1, 'K': 1, 'L': 4, 'M': 2, 'N': 6, 'O': 8, 'P': 2,
        'Q': 1, 'R': 6, 'S': 4, 'T': 6, 'U': 4, 'V': 2, 'W': 2, 'X': 1,
        'Y': 2, 'Z': 1, '?': 2
    }
    bag = []
    for letter, count in distribution.items():
        bag.extend([letter] * count)
    random.shuffle(bag)
    return bag

def draw_tiles(bag, count):
    """Draw tiles from bag"""
    drawn = []
    for _ in range(min(count, len(bag))):
        drawn.append(bag.pop())
    return drawn

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
                print(f"  {letter} at ({r},{c}): TL x3 = {val}")
            elif (r, c) in DL:
                val *= 2
                print(f"  {letter} at ({r},{c}): DL x2 = {val}")
            else:
                print(f"  {letter} at ({r},{c}): {val}")
            if (r, c) in TW:
                word_mult *= 3
                print(f"  ({r},{c}): TW word_mult now {word_mult}")
            elif (r, c) in DW:
                word_mult *= 2
                print(f"  ({r},{c}): DW word_mult now {word_mult}")
        else:
            print(f"  {letter} at ({r},{c}): existing tile = {val}")
        
        letter_score += val
    
    total = letter_score * word_mult
    print(f"  Word {word}: {letter_score} * {word_mult} = {total}")
    return total


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
                    
                    # CRITICAL: Check if we're extending an existing word
                    # Get the full connected word in this direction
                    full_word_start_r, full_word_start_c = start_r, start_c
                    full_word_end_r, full_word_end_c = end_r, end_c
                    
                    # Check for tiles BEFORE our word
                    check_r, check_c = start_r - dr, start_c - dc
                    while 0 <= check_r < 15 and 0 <= check_c < 15 and board[check_r][check_c]:
                        full_word_start_r, full_word_start_c = check_r, check_c
                        check_r -= dr
                        check_c -= dc
                    
                    # Check for tiles AFTER our word
                    check_r, check_c = end_r + dr, end_c + dc
                    while 0 <= check_r < 15 and 0 <= check_c < 15 and board[check_r][check_c]:
                        full_word_end_r, full_word_end_c = check_r, check_c
                        check_r += dr
                        check_c += dc
                    
                    # Build the full connected word
                    full_word = ""
                    curr_r, curr_c = full_word_start_r, full_word_start_c
                    while True:
                        if (curr_r, curr_c) in new_positions:
                            # This is a tile we're placing
                            for tile in tiles_to_place:
                                if tile['row'] == curr_r and tile['col'] == curr_c:
                                    full_word += tile['letter']
                                    break
                        elif board[curr_r][curr_c]:
                            full_word += board[curr_r][curr_c]
                        else:
                            # Gap - shouldn't happen if logic is right
                            break
                        
                        if curr_r == full_word_end_r and curr_c == full_word_end_c:
                            break
                        curr_r += dr
                        curr_c += dc
                    
                    # Validate the full connected word
                    if full_word not in WORDS:
                        continue  # Invalid extended word!
                    
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


# Claude API for definitions and hints - uses environment variable
CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY', '')

@app.route('/api/define', methods=['POST'])
def define_word():
    """Get word definition from Claude"""
    data = request.json
    word = data.get('word', '').upper()
    
    # First check if it's in our Scrabble dictionary
    is_valid_scrabble = word in WORDS
    
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
                    'content': f'Define the word "{word}" briefly in 1-2 sentences. This is a valid Scrabble word. If you\'re not sure what it means, say it\'s a valid Scrabble word and give your best guess at its meaning or origin. Do not say it\'s not a real word - it is in the official Scrabble dictionary.'
                }]
            },
            timeout=10
        )
        result = response.json()
        definition = result.get('content', [{}])[0].get('text', 'Definition not found')
        
        # Add note if it's a valid Scrabble word
        if is_valid_scrabble and "not a word" in definition.lower():
            definition = f"Valid Scrabble word. {definition}"
            
        return jsonify({'success': True, 'definition': definition, 'valid_scrabble': is_valid_scrabble})
    except Exception as e:
        print(f"Define error: {e}")
        if is_valid_scrabble:
            return jsonify({'success': True, 'definition': 'Valid Scrabble word (definition unavailable)', 'valid_scrabble': True})
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


# ============== WebSocket Events ==============

@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")
    emit('connected', {'sid': request.sid})


@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")
    # Remove player from any rooms they were in
    for room_code, game in list(game_rooms.items()):
        for player_id, player in list(game.get('players', {}).items()):
            if player.get('sid') == request.sid:
                player['connected'] = False
                # Notify room
                socketio.emit('player_disconnected', {
                    'player_id': player_id,
                    'name': player.get('name', 'Unknown')
                }, room=room_code)


@socketio.on('create_room')
def handle_create_room(data):
    """Create a new game room"""
    print(f"[CREATE_ROOM] Received request: {data}")
    print(f"[CREATE_ROOM] From SID: {request.sid}")
    
    room_code = generate_room_code()
    player_name = data.get('name', 'Player 1')
    
    game_rooms[room_code] = {
        'code': room_code,
        'host_sid': request.sid,
        'status': 'waiting',  # waiting, playing, finished
        'players': {
            request.sid: {
                'sid': request.sid,
                'name': player_name,
                'rack': [],
                'score': 0,
                'connected': True,
                'order': 0
            }
        },
        'board_viewers': [],  # SIDs of iPad board viewers
        'board': [[None]*15 for _ in range(15)],
        'bag': create_tile_bag(),
        'current_player_idx': 0,
        'player_order': [request.sid],
        'turn_number': 0,
        'last_move': None,
        'pass_count': 0
    }
    
    join_room(room_code)
    
    emit('room_created', {
        'room_code': room_code,
        'player_id': request.sid,
        'players': [{
            'id': request.sid,
            'name': player_name,
            'score': 0
        }]
    })
    
    print(f"[CREATE_ROOM] Room {room_code} created by {player_name}")
    print(f"[CREATE_ROOM] Active rooms: {list(game_rooms.keys())}")


@socketio.on('join_room')
def handle_join_room(data):
    """Join existing room as player"""
    print(f"[JOIN_ROOM] Received request: {data}")
    print(f"[JOIN_ROOM] From SID: {request.sid}")
    print(f"[JOIN_ROOM] Active rooms: {list(game_rooms.keys())}")
    
    room_code = data.get('room_code', '').upper()
    player_name = data.get('name', 'Player')
    mode = data.get('mode', 'player')  # 'player' or 'board'
    
    if room_code not in game_rooms:
        print(f"[JOIN_ROOM] ERROR: Room {room_code} not found")
        emit('error', {'message': f'Room {room_code} not found'})
        return
    
    print(f"[JOIN_ROOM] Found room {room_code}")
    game = game_rooms[room_code]
    
    if mode == 'board':
        # iPad joining as board viewer
        game['board_viewers'].append(request.sid)
        join_room(room_code)
        
        # Send current board state
        emit('board_joined', {
            'room_code': room_code,
            'board': game['board'],
            'players': [{'id': pid, 'name': p['name'], 'score': p['score']} 
                       for pid, p in game['players'].items()],
            'status': game['status'],
            'current_player_idx': game['current_player_idx'],
            'bag_count': len(game['bag'])
        })
        print(f"Board viewer joined room {room_code}")
        return
    
    # Check if this is a reconnection (player with same name was disconnected)
    reconnecting_player_id = None
    for pid, p in game['players'].items():
        if p['name'] == player_name and not p.get('connected', True):
            reconnecting_player_id = pid
            break
    
    if reconnecting_player_id:
        # Reconnecting player - update their SID
        old_player = game['players'].pop(reconnecting_player_id)
        old_player['sid'] = request.sid
        old_player['connected'] = True
        game['players'][request.sid] = old_player
        
        # Update player_order
        idx = game['player_order'].index(reconnecting_player_id)
        game['player_order'][idx] = request.sid
        
        join_room(room_code)
        
        player_list = [{'id': pid, 'name': p['name'], 'score': p['score']} 
                       for pid, p in game['players'].items()]
        
        # Send game state to reconnecting player
        emit('room_joined', {
            'room_code': room_code,
            'player_id': request.sid,
            'players': player_list
        })
        
        # If game is in progress, send full state
        if game['status'] == 'playing':
            emit('game_started', {
                'your_rack': old_player['rack'],
                'current_player_idx': game['current_player_idx'],
                'current_player_id': game['player_order'][game['current_player_idx']],
                'players': player_list,
                'bag_count': len(game['bag']),
                'board': game['board']
            })
        
        print(f"{player_name} reconnected to room {room_code}")
        return
    
    # New player joining
    if game['status'] != 'waiting':
        emit('error', {'message': 'Game already in progress'})
        return
    
    if len(game['players']) >= 4:
        emit('error', {'message': 'Room is full (max 4 players)'})
        return
    
    # Add player
    game['players'][request.sid] = {
        'sid': request.sid,
        'name': player_name,
        'rack': [],
        'score': 0,
        'connected': True,
        'order': len(game['players'])
    }
    game['player_order'].append(request.sid)
    
    join_room(room_code)
    
    # Build player list
    player_list = [{'id': pid, 'name': p['name'], 'score': p['score']} 
                   for pid, p in game['players'].items()]
    
    # Notify joiner
    emit('room_joined', {
        'room_code': room_code,
        'player_id': request.sid,
        'players': player_list
    })
    
    # Notify everyone in room
    socketio.emit('player_joined', {
        'player_id': request.sid,
        'name': player_name,
        'players': player_list
    }, room=room_code)
    
    print(f"{player_name} joined room {room_code}")


@socketio.on('start_game')
def handle_start_game(data):
    """Host starts the game"""
    room_code = data.get('room_code', '').upper()
    
    if room_code not in game_rooms:
        emit('error', {'message': 'Room not found'})
        return
    
    game = game_rooms[room_code]
    
    if request.sid != game['host_sid']:
        emit('error', {'message': 'Only host can start game'})
        return
    
    if len(game['players']) < 2:
        emit('error', {'message': 'Need at least 2 players'})
        return
    
    # Deal tiles to all players
    for player_id in game['player_order']:
        game['players'][player_id]['rack'] = draw_tiles(game['bag'], 7)
    
    game['status'] = 'playing'
    game['current_player_idx'] = 0
    
    # Send each player their rack privately
    for player_id, player in game['players'].items():
        socketio.emit('game_started', {
            'your_rack': player['rack'],
            'current_player_idx': 0,
            'current_player_id': game['player_order'][0],
            'players': [{'id': pid, 'name': p['name'], 'score': p['score']} 
                       for pid, p in game['players'].items()],
            'bag_count': len(game['bag'])
        }, room=player_id)
    
    # Send to board viewers
    for viewer_sid in game['board_viewers']:
        socketio.emit('game_started', {
            'current_player_idx': 0,
            'current_player_id': game['player_order'][0],
            'players': [{'id': pid, 'name': p['name'], 'score': p['score']} 
                       for pid, p in game['players'].items()],
            'bag_count': len(game['bag'])
        }, room=viewer_sid)
    
    print(f"Game started in room {room_code}")


@socketio.on('play_move')
def handle_play_move(data):
    """Player plays tiles"""
    room_code = data.get('room_code', '').upper()
    tiles = data.get('tiles', [])  # [{row, col, letter, is_blank}]
    
    if room_code not in game_rooms:
        emit('error', {'message': 'Room not found'})
        return
    
    game = game_rooms[room_code]
    
    if game['status'] != 'playing':
        emit('error', {'message': 'Game not in progress'})
        return
    
    current_player_id = game['player_order'][game['current_player_idx']]
    if request.sid != current_player_id:
        emit('error', {'message': 'Not your turn'})
        return
    
    player = game['players'][request.sid]
    
    # Validate and place tiles
    # Build words and validate them
    words_formed = data.get('words', [])
    
    # Check all words are valid
    invalid_words = [w for w in words_formed if w.upper() not in WORDS]
    if invalid_words:
        emit('move_rejected', {'message': f'Invalid word(s): {", ".join(invalid_words)}'})
        return
    
    # Calculate score server-side for consistency
    new_positions = set((t['row'], t['col']) for t in tiles)
    blank_positions = set((t['row'], t['col']) for t in tiles if t.get('is_blank'))
    
    total_score = 0
    
    # Build the board state after placing tiles (temporarily)
    temp_board = [row[:] for row in game['board']]  # Copy board
    for tile in tiles:
        temp_board[tile['row']][tile['col']] = tile['letter']
    
    # Score all words formed
    for word in words_formed:
        # Find word positions on board
        word_upper = word.upper()
        word_positions = []
        
        # Search horizontally and vertically for this word
        for r in range(15):
            for c in range(15):
                # Check horizontal
                if c + len(word_upper) <= 15:
                    match = True
                    positions = []
                    for i, letter in enumerate(word_upper):
                        cell = temp_board[r][c + i]
                        cell_letter = cell if isinstance(cell, str) else (cell.get('letter') if cell else None)
                        if cell_letter != letter:
                            match = False
                            break
                        positions.append((r, c + i))
                    if match and len(positions) == len(word_upper):
                        word_positions = positions
                        break
                # Check vertical
                if r + len(word_upper) <= 15:
                    match = True
                    positions = []
                    for i, letter in enumerate(word_upper):
                        cell = temp_board[r + i][c]
                        cell_letter = cell if isinstance(cell, str) else (cell.get('letter') if cell else None)
                        if cell_letter != letter:
                            match = False
                            break
                        positions.append((r + i, c))
                    if match and len(positions) == len(word_upper):
                        word_positions = positions
                        break
            if word_positions:
                break
        
        if word_positions:
            word_score = score_word(word_upper, word_positions, new_positions, blank_positions)
            total_score += word_score
    
    # Bingo bonus (all 7 tiles)
    if len(tiles) == 7:
        total_score += 50
    
    score = total_score
    
    # Place tiles on board
    for tile in tiles:
        game['board'][tile['row']][tile['col']] = tile['letter']
        # Remove from player's rack
        letter_to_remove = '?' if tile.get('is_blank') else tile['letter']
        if letter_to_remove in player['rack']:
            player['rack'].remove(letter_to_remove)
    
    # Update score
    player['score'] += score
    
    # Draw new tiles
    new_tiles = draw_tiles(game['bag'], len(tiles))
    player['rack'].extend(new_tiles)
    
    # Reset pass count
    game['pass_count'] = 0
    
    # Store last move
    game['last_move'] = {
        'player_id': request.sid,
        'player_name': player['name'],
        'tiles': tiles,
        'words': words_formed,
        'score': score
    }
    
    # Next turn
    game['current_player_idx'] = (game['current_player_idx'] + 1) % len(game['player_order'])
    game['turn_number'] += 1
    
    next_player_id = game['player_order'][game['current_player_idx']]
    
    # Check for game end
    game_over = False
    if len(player['rack']) == 0 and len(game['bag']) == 0:
        game_over = True
        game['status'] = 'finished'
    
    # Send updated state to everyone
    player_list = [{'id': pid, 'name': p['name'], 'score': p['score']} 
                   for pid, p in game['players'].items()]
    
    # Send to all players
    for pid, p in game['players'].items():
        socketio.emit('move_played', {
            'board': game['board'],
            'last_move': game['last_move'],
            'your_rack': p['rack'],
            'players': player_list,
            'current_player_idx': game['current_player_idx'],
            'current_player_id': next_player_id,
            'bag_count': len(game['bag']),
            'game_over': game_over
        }, room=pid)
    
    # Send to board viewers
    for viewer_sid in game['board_viewers']:
        socketio.emit('move_played', {
            'board': game['board'],
            'last_move': game['last_move'],
            'players': player_list,
            'current_player_idx': game['current_player_idx'],
            'current_player_id': next_player_id,
            'bag_count': len(game['bag']),
            'game_over': game_over
        }, room=viewer_sid)
    
    print(f"Move played in {room_code}: {words_formed} for {score} pts")


@socketio.on('pass_turn')
def handle_pass_turn(data):
    """Player passes their turn"""
    room_code = data.get('room_code', '').upper()
    
    if room_code not in game_rooms:
        emit('error', {'message': 'Room not found'})
        return
    
    game = game_rooms[room_code]
    current_player_id = game['player_order'][game['current_player_idx']]
    
    if request.sid != current_player_id:
        emit('error', {'message': 'Not your turn'})
        return
    
    game['pass_count'] += 1
    game['current_player_idx'] = (game['current_player_idx'] + 1) % len(game['player_order'])
    
    # Game ends if all players pass twice
    game_over = game['pass_count'] >= len(game['players']) * 2
    if game_over:
        game['status'] = 'finished'
    
    next_player_id = game['player_order'][game['current_player_idx']]
    player_list = [{'id': pid, 'name': p['name'], 'score': p['score']} 
                   for pid, p in game['players'].items()]
    
    # Notify everyone
    socketio.emit('turn_passed', {
        'player_id': request.sid,
        'player_name': game['players'][request.sid]['name'],
        'current_player_idx': game['current_player_idx'],
        'current_player_id': next_player_id,
        'players': player_list,
        'game_over': game_over
    }, room=room_code)


@socketio.on('get_state')
def handle_get_state(data):
    """Get current game state (for reconnection)"""
    room_code = data.get('room_code', '').upper()
    
    if room_code not in game_rooms:
        emit('error', {'message': 'Room not found'})
        return
    
    game = game_rooms[room_code]
    player = game['players'].get(request.sid)
    
    emit('game_state', {
        'room_code': room_code,
        'board': game['board'],
        'your_rack': player['rack'] if player else [],
        'players': [{'id': pid, 'name': p['name'], 'score': p['score']} 
                   for pid, p in game['players'].items()],
        'current_player_idx': game['current_player_idx'],
        'current_player_id': game['player_order'][game['current_player_idx']] if game['player_order'] else None,
        'status': game['status'],
        'bag_count': len(game['bag']),
        'last_move': game['last_move']
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)
