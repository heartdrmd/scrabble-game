import eventlet
eventlet.monkey_patch()
from flask import Flask, jsonify, request, send_from_directory
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import os
import random
import requests
import string
import time
from collections import Counter

# ============================================
# FIX 1: Player list ordering after reconnection
# ============================================
# PROBLEM: When a player disconnects and reconnects, their player_id changes.
# The old code built player_list from game['players'].items() which is a dict.
# Python dicts preserve insertion order, but reconnection changes the order
# (old key removed, new key added at end). This caused the turn indicator
# to show the wrong player's name even though the game logic was correct.
#
# SOLUTION: All player_list construction now uses game['player_order'] as
# the source of truth, not game['players'].items(). The player_order list
# maintains the correct turn order regardless of reconnections.
# ============================================
# FIX 2: Official Scrabble end-game scoring
# ============================================
# PROBLEM: End-game tile penalty was not implemented.
#
# SOLUTION: Now applies official Scrabble rules:
# - When someone plays out: others LOSE their tile values, winner GAINS them
# - When everyone passes: everyone LOSES their own tile values (no one gains)
# ============================================
# FIX 3: Disconnect timeout and kick handling
# ============================================
# PROBLEM: In timer-OFF games, if a player disconnects during their turn,
# the game would be stuck forever waiting for them.
#
# SOLUTION:
# - Timer ON: Turn timer handles skips automatically
# - Timer OFF: After 2.5 minutes of disconnect, auto-skip their turn
# - After 2 consecutive disconnect skips, player is KICKED from game
# - Kicked players keep their score (for display) but can't take turns
# - If fewer than 2 players remain, game ends automatically
# ============================================

app = Flask(__name__)
CORS(app)
# Use eventlet for production WebSocket support
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=60, ping_interval=25, async_mode='eventlet')


# ============== Disconnect Timeout Settings ==============
DISCONNECT_TIMEOUT_SECONDS = 150  # 2.5 minutes before auto-skip for disconnected player
MAX_DISCONNECT_SKIPS = 2          # After this many skips while disconnected, kick player


# ============== GADDAG Data Structure ==============

class GADDAGNode:
    """A node in the GADDAG tree"""
    __slots__ = ['children', 'is_word_end']  # Memory optimization
    
    def __init__(self):
        self.children = {}      # letter -> GADDAGNode
        self.is_word_end = False  # True if this completes a valid word

DELIMITER = '>'  # Separates reversed-prefix from suffix

class GADDAG:
    """
    GADDAG for fast Scrabble move generation.
    
    For word "CART":
      - C>ART   (start at C, go right)
      - AC>RT   (start at A, left to C, right to RT)
      - RAC>T   (start at R, left to AC, right to T)
      - TRAC>   (start at T, left to RAC)
    """
    
    def __init__(self):
        self.root = GADDAGNode()
        self.word_count = 0
    
    def add_word(self, word):
        """Add a word to the GADDAG with all possible anchor points"""
        word = word.upper()
        n = len(word)
        
        # For each possible anchor position (each letter in the word)
        for anchor_pos in range(n):
            node = self.root
            
            # 1. Add REVERSED prefix (letters BEFORE anchor, going backwards)
            for i in range(anchor_pos - 1, -1, -1):
                letter = word[i]
                if letter not in node.children:
                    node.children[letter] = GADDAGNode()
                node = node.children[letter]
            
            # 2. Add delimiter (marks transition from left to right)
            if DELIMITER not in node.children:
                node.children[DELIMITER] = GADDAGNode()
            node = node.children[DELIMITER]
            
            # 3. Add suffix (letters FROM anchor onwards, going forward)
            for i in range(anchor_pos, n):
                letter = word[i]
                if letter not in node.children:
                    node.children[letter] = GADDAGNode()
                node = node.children[letter]
            
            # Mark end of word
            node.is_word_end = True
        
        self.word_count += 1
    
    def build_from_words(self, words):
        """Build GADDAG from word list"""
        start = time.time()
        for word in words:
            if 2 <= len(word) <= 15:
                self.add_word(word)
        elapsed = time.time() - start
        print(f"GADDAG built: {self.word_count} words in {elapsed:.2f}s")
        return self

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
GADDAG_ROOT = None  # Will be initialized at startup

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

# Build GADDAG for fast move generation
print("Building GADDAG...")
gaddag = GADDAG()
GADDAG_ROOT = gaddag.build_from_words(WORDS)

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
    
    total = letter_score * word_mult
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


def get_cross_check_set(board, row, col, dr, dc):
    """
    Get the set of letters that can legally be placed at (row, col)
    based on perpendicular (cross) words.
    Returns None if any letter is valid (no perpendicular constraints).
    Returns a set of valid letters otherwise.
    """
    pdr, pdc = (0, 1) if dr == 1 else (1, 0)
    
    # Check if there are any perpendicular tiles
    has_before = (row - pdr >= 0 and col - pdc >= 0 and board[row - pdr][col - pdc])
    has_after = (row + pdr < 15 and col + pdc < 15 and board[row + pdr][col + pdc])
    
    if not has_before and not has_after:
        return None  # Any letter is valid
    
    # Build prefix (tiles before this position in perpendicular direction)
    prefix = ""
    r, c = row - pdr, col - pdc
    while r >= 0 and c >= 0 and board[r][c]:
        prefix = board[r][c] + prefix
        r -= pdr
        c -= pdc
    
    # Build suffix (tiles after this position in perpendicular direction)
    suffix = ""
    r, c = row + pdr, col + pdc
    while 0 <= r < 15 and 0 <= c < 15 and board[r][c]:
        suffix += board[r][c]
        r += pdr
        c += pdc
    
    # Find all letters that form valid words
    valid_letters = set()
    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        word = prefix + letter + suffix
        if word in WORDS:
            valid_letters.add(letter)
    
    return valid_letters


def find_all_moves_gaddag(board, rack):
    """
    Find all valid moves using GADDAG - MUCH faster than brute force!
    
    Instead of checking 178k words, we walk the GADDAG tree,
    only exploring paths that match our available letters.
    """
    moves = []
    rack_list = list(rack.upper())
    rack_counter = Counter(rack_list)
    
    is_first = all(board[r][c] is None for r in range(15) for c in range(15))
    
    # Find anchors (empty squares next to filled squares)
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
    
    # Pre-compute cross-check sets for each empty square
    cross_checks = {}
    for r in range(15):
        for c in range(15):
            if board[r][c] is None:
                # Horizontal play: check vertical cross-words
                cross_checks[(r, c, 0, 1)] = get_cross_check_set(board, r, c, 0, 1)
                # Vertical play: check horizontal cross-words
                cross_checks[(r, c, 1, 0)] = get_cross_check_set(board, r, c, 1, 0)
    
    def gen_moves(anchor_r, anchor_c, dr, dc):
        """Generate all moves through this anchor in given direction"""
        direction = 'H' if dc == 1 else 'V'
        
        # How many empty squares before anchor can we use?
        max_left = 0
        r, c = anchor_r - dr, anchor_c - dc
        while r >= 0 and c >= 0 and board[r][c] is None and max_left < len(rack_list):
            # Check it's not another anchor (would be handled by that anchor)
            if (r, c) in anchors and (r, c) != (anchor_r, anchor_c):
                break
            max_left += 1
            r -= dr
            c -= dc
        
        def extend_right(node, pos_r, pos_c, word, positions, tiles_placed, remaining_rack, blank_positions):
            """Extend word to the right (after delimiter)"""
            # Current position valid?
            if pos_r < 0 or pos_r >= 15 or pos_c < 0 or pos_c >= 15:
                return
            
            current_tile = board[pos_r][pos_c]
            
            if current_tile:
                # Board has a tile here - must use it
                if current_tile in node.children:
                    new_node = node.children[current_tile]
                    new_word = word + current_tile
                    new_positions = positions + [(pos_r, pos_c)]
                    
                    # Check if word complete and we placed at least one tile
                    if new_node.is_word_end and tiles_placed:
                        # Validate and add move
                        add_move_if_valid(new_word, new_positions, tiles_placed, blank_positions, direction)
                    
                    # Continue extending
                    extend_right(new_node, pos_r + dr, pos_c + dc, new_word, new_positions, 
                                tiles_placed, remaining_rack, blank_positions)
            else:
                # Empty square - try placing tiles from rack
                cross_set = cross_checks.get((pos_r, pos_c, dr, dc))
                
                for letter, child_node in node.children.items():
                    if letter == DELIMITER:
                        continue
                    
                    # Check cross-word constraint
                    if cross_set is not None and letter not in cross_set:
                        continue
                    
                    # Try using this letter from rack
                    if remaining_rack.get(letter, 0) > 0:
                        new_rack = remaining_rack.copy()
                        new_rack[letter] -= 1
                        new_word = word + letter
                        new_positions = positions + [(pos_r, pos_c)]
                        new_tiles = tiles_placed + [{'row': pos_r, 'col': pos_c, 'letter': letter, 'is_blank': False}]
                        
                        if child_node.is_word_end:
                            add_move_if_valid(new_word, new_positions, new_tiles, blank_positions, direction)
                        
                        extend_right(child_node, pos_r + dr, pos_c + dc, new_word, new_positions,
                                    new_tiles, new_rack, blank_positions)
                    
                    # Try using blank tile
                    if remaining_rack.get('?', 0) > 0:
                        new_rack = remaining_rack.copy()
                        new_rack['?'] -= 1
                        new_word = word + letter
                        new_positions = positions + [(pos_r, pos_c)]
                        new_tiles = tiles_placed + [{'row': pos_r, 'col': pos_c, 'letter': letter, 'is_blank': True}]
                        new_blanks = blank_positions | {(pos_r, pos_c)}
                        
                        if child_node.is_word_end:
                            add_move_if_valid(new_word, new_positions, new_tiles, new_blanks, direction)
                        
                        extend_right(child_node, pos_r + dr, pos_c + dc, new_word, new_positions,
                                    new_tiles, new_rack, new_blanks)
        
        def extend_left(node, pos_r, pos_c, left_part, left_positions, tiles_placed, remaining_rack, blank_positions, tiles_placed_left):
            """Build prefix going left, then switch to going right at delimiter."""
            
            # Try switching to right extension (go through delimiter)
            if DELIMITER in node.children:
                delim_node = node.children[DELIMITER]
                # Start extending right from anchor position
                extend_right(delim_node, anchor_r, anchor_c, left_part, left_positions,
                            tiles_placed, remaining_rack, blank_positions)
            
            new_r, new_c = pos_r - dr, pos_c - dc
            if new_r < 0 or new_c < 0 or new_r >= 15 or new_c >= 15:
                return
            
            current_tile = board[new_r][new_c]
            
            if current_tile:
                # Board has an EXISTING tile - traverse it in GADDAG
                if current_tile in node.children:
                    new_node = node.children[current_tile]
                    new_left = current_tile + left_part
                    new_positions = [(new_r, new_c)] + left_positions
                    extend_left(new_node, new_r, new_c, new_left, new_positions,
                               tiles_placed, remaining_rack, blank_positions, tiles_placed_left)
            else:
                # Empty square - can we place MORE new tiles going left?
                if tiles_placed_left >= max_left:
                    return  # Limit reached for NEW placements
                
                # Don't cross into another anchor's territory
                if (new_r, new_c) in anchors and (new_r, new_c) != (anchor_r, anchor_c):
                    return
                
                cross_set = cross_checks.get((new_r, new_c, dr, dc))
                
                for letter, child_node in node.children.items():
                    if letter == DELIMITER:
                        continue
                    
                    if cross_set is not None and letter not in cross_set:
                        continue
                    
                    if remaining_rack.get(letter, 0) > 0:
                        new_rack = remaining_rack.copy()
                        new_rack[letter] -= 1
                        new_left = letter + left_part
                        new_positions = [(new_r, new_c)] + left_positions
                        new_tiles = [{'row': new_r, 'col': new_c, 'letter': letter, 'is_blank': False}] + tiles_placed
                        
                        extend_left(child_node, new_r, new_c, new_left, new_positions,
                                   new_tiles, new_rack, blank_positions, tiles_placed_left + 1)
                    
                    if remaining_rack.get('?', 0) > 0:
                        new_rack = remaining_rack.copy()
                        new_rack['?'] -= 1
                        new_left = letter + left_part
                        new_positions = [(new_r, new_c)] + left_positions
                        new_tiles = [{'row': new_r, 'col': new_c, 'letter': letter, 'is_blank': True}] + tiles_placed
                        new_blanks = blank_positions | {(new_r, new_c)}
                        
                        extend_left(child_node, new_r, new_c, new_left, new_positions,
                                   new_tiles, new_rack, new_blanks, tiles_placed_left + 1)
        
        def add_move_if_valid(word, positions, tiles_placed, blank_positions, direction):
            """Validate cross words AND main word extension, add move if all valid"""
            if not tiles_placed:
                return
            
            # First move must cover center
            if is_first and (7, 7) not in positions:
                return
            
            new_positions = set((t['row'], t['col']) for t in tiles_placed)
            
            # Check if we're extending an existing word
            start_r, start_c = positions[0]
            end_r, end_c = positions[-1]
            
            # Find the FULL word including any existing tiles before/after
            full_word_start_r, full_word_start_c = start_r, start_c
            full_word_end_r, full_word_end_c = end_r, end_c
            
            # Check for existing tiles BEFORE our word
            check_r, check_c = start_r - dr, start_c - dc
            while 0 <= check_r < 15 and 0 <= check_c < 15 and board[check_r][check_c]:
                full_word_start_r, full_word_start_c = check_r, check_c
                check_r -= dr
                check_c -= dc
            
            # Check for existing tiles AFTER our word
            check_r, check_c = end_r + dr, end_c + dc
            while 0 <= check_r < 15 and 0 <= check_c < 15 and board[check_r][check_c]:
                full_word_end_r, full_word_end_c = check_r, check_c
                check_r += dr
                check_c += dc
            
            # Build the FULL connected word (including existing tiles)
            full_word = ""
            full_positions = []
            curr_r, curr_c = full_word_start_r, full_word_start_c
            
            # Create a map of tiles we're placing for quick lookup
            tiles_map = {(t['row'], t['col']): t['letter'] for t in tiles_placed}
            
            while True:
                full_positions.append((curr_r, curr_c))
                
                if (curr_r, curr_c) in tiles_map:
                    # This is a tile we're placing
                    full_word += tiles_map[(curr_r, curr_c)]
                elif board[curr_r][curr_c]:
                    # Existing tile on board
                    full_word += board[curr_r][curr_c]
                else:
                    # Gap - shouldn't happen
                    break
                
                if curr_r == full_word_end_r and curr_c == full_word_end_c:
                    break
                curr_r += dr
                curr_c += dc
            
            # Validate the FULL connected word
            if full_word not in WORDS:
                return  # Invalid!
            
            # Validate all cross words
            total_cross_score = 0
            
            for tile in tiles_placed:
                cross = get_cross_word(board, tile['row'], tile['col'], dr, dc, tile['letter'])
                if cross:
                    cword, cpos = cross
                    if cword not in WORDS:
                        return  # Invalid cross word
                    cross_blanks = {(tile['row'], tile['col'])} if tile['is_blank'] else set()
                    total_cross_score += score_word(cword, cpos, {(tile['row'], tile['col'])}, cross_blanks)
            
            # Score the FULL main word (including existing tiles!)
            main_score = score_word(full_word, full_positions, new_positions, blank_positions)
            total_score = main_score + total_cross_score
            
            # Bingo bonus
            if len(tiles_placed) == 7:
                total_score += 50
            
            moves.append({
                'word': full_word,
                'start_row': full_positions[0][0],
                'start_col': full_positions[0][1],
                'direction': direction,
                'tiles': tiles_placed,
                'score': total_score
            })
        
        # Start the search from GADDAG root
        extend_left(GADDAG_ROOT.root, anchor_r, anchor_c, "", [], [], rack_counter.copy(), set(), 0)
    
    # Generate moves for each anchor in each direction
    for anchor_r, anchor_c in anchors:
        gen_moves(anchor_r, anchor_c, 0, 1)  # Horizontal
        gen_moves(anchor_r, anchor_c, 1, 0)  # Vertical
    
    return moves


# Keep old function name as alias for compatibility
def find_all_moves(board, rack):
    return find_all_moves_gaddag(board, rack)


def get_ai_move(board, rack, difficulty='hard'):
    """Get AI move based on difficulty - uses GADDAG for fast search"""
    start_time = time.time()
    
    moves = find_all_moves_gaddag(board, rack)
    
    elapsed = time.time() - start_time
    print(f"GADDAG found {len(moves)} moves in {elapsed:.3f}s")
    
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

# Definition cache - persists for server lifetime
DEFINITION_CACHE = {}

def get_free_dictionary_definition(word):
    """Fallback: Get definition from free dictionary API"""
    try:
        response = requests.get(
            f'https://api.dictionaryapi.dev/api/v2/entries/en/{word.lower()}',
            timeout=5
        )
        if response.ok:
            data = response.json()
            if data and len(data) > 0:
                meanings = data[0].get('meanings', [])
                if meanings:
                    definitions = meanings[0].get('definitions', [])
                    if definitions:
                        return definitions[0].get('definition', None)
    except Exception as e:
        print(f"Free dictionary fallback error: {e}")
    return None

def call_claude_api(messages, max_tokens=200, max_retries=2):
    """Call Claude API with retry logic"""
    for attempt in range(max_retries):
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
                    'max_tokens': max_tokens,
                    'messages': messages
                },
                timeout=10
            )
            if response.ok:
                result = response.json()
                text = result.get('content', [{}])[0].get('text', None)
                if text:
                    return text
        except Exception as e:
            print(f"Claude API attempt {attempt + 1} failed: {e}")
        
        # Brief pause before retry (only if not last attempt)
        if attempt < max_retries - 1:
            time.sleep(0.5)
    
    return None  # All retries failed

@app.route('/api/define', methods=['POST'])
def define_word():
    """Get word definition - with caching, retry, and fallback"""
    data = request.json
    word = data.get('word', '').upper()
    
    # First check if it's in our Scrabble dictionary
    is_valid_scrabble = word in WORDS
    
    # Check cache first
    if word in DEFINITION_CACHE:
        print(f"Definition cache hit: {word}")
        return jsonify({'success': True, 'definition': DEFINITION_CACHE[word], 'valid_scrabble': is_valid_scrabble, 'cached': True})
    
    # Try Claude API with retry
    definition = call_claude_api([{
        'role': 'user',
        'content': f'Define the word "{word}" briefly in 1-2 sentences. This is a valid Scrabble word. If you\'re not sure what it means, say it\'s a valid Scrabble word and give your best guess at its meaning or origin. Do not say it\'s not a real word - it is in the official Scrabble dictionary.'
    }], max_tokens=200, max_retries=2)
    
    if definition:
        # Add note if it's a valid Scrabble word but Claude doubted it
        if is_valid_scrabble and "not a word" in definition.lower():
            definition = f"Valid Scrabble word. {definition}"
        
        # Cache the successful definition
        DEFINITION_CACHE[word] = definition
        print(f"Definition cached: {word}")
        return jsonify({'success': True, 'definition': definition, 'valid_scrabble': is_valid_scrabble})
    
    # Claude failed - try free dictionary fallback
    print(f"Claude failed for {word}, trying free dictionary...")
    fallback_def = get_free_dictionary_definition(word)
    
    if fallback_def:
        # Cache the fallback too
        DEFINITION_CACHE[word] = fallback_def
        print(f"Fallback definition cached: {word}")
        return jsonify({'success': True, 'definition': fallback_def, 'valid_scrabble': is_valid_scrabble, 'fallback': True})
    
    # All sources failed
    if is_valid_scrabble:
        return jsonify({'success': True, 'definition': 'Valid Scrabble word (definition unavailable)', 'valid_scrabble': True})
    return jsonify({'success': False, 'definition': 'Could not fetch definition'})


@app.route('/api/funny_hint', methods=['POST'])
def funny_hint():
    """Get funny hint from Claude with retry logic"""
    data = request.json
    word = data.get('word', '')
    rack = data.get('rack', [])
    
    # Use retry helper
    hint = call_claude_api([{
        'role': 'user',
        'content': f'You\'re a witty Scrabble coach. The player has these letters: {", ".join(rack)}. The best word they can play is "{word}" but DON\'T say the word! Give them a clever, funny hint or clue to help them figure it out. Be playful and brief (1-2 sentences). Don\'t use the word itself or obvious anagrams.'
    }], max_tokens=150, max_retries=2)
    
    if hint:
        return jsonify({'success': True, 'hint': hint})
    else:
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
        return jsonify({
            'success': False, 
            'message': 'No valid moves found',
            'should_pass': True  # Signal to client that passing is recommended
        })

@app.route('/api/hint', methods=['POST'])
def hint():
    return ai_move()


# ============== Disconnect Timeout Functions ==============

def kick_player(room_code, game, player_id, reason="disconnected"):
    """Kick a player from the game (they stay in players dict for score, but removed from play)"""
    if player_id not in game['players']:
        return False
    
    player = game['players'][player_id]
    player_name = player.get('name', 'Player')
    
    print(f"[KICK] Kicking {player_name} from {room_code} (reason: {reason})")
    
    # Mark as kicked (keep in players dict for score display)
    player['kicked'] = True
    player['connected'] = False
    
    # Remove from player_order (they can't take turns anymore)
    if player_id in game['player_order']:
        kicked_idx = game['player_order'].index(player_id)
        game['player_order'].remove(player_id)
        
        # Adjust current_player_idx if needed
        if len(game['player_order']) > 0:
            if game['current_player_idx'] >= len(game['player_order']):
                game['current_player_idx'] = 0
            elif kicked_idx < game['current_player_idx']:
                game['current_player_idx'] -= 1
    
    # Check if game should end (fewer than 2 active players)
    active_players = len(game['player_order'])
    if active_players < 2:
        game['status'] = 'finished'
        print(f"[KICK] Game ended - only {active_players} player(s) left")
    
    # Notify everyone
    # Build player list in CORRECT order (using player_order, not dict order)
    active_list = [{'id': pid, 'name': game['players'][pid]['name'], 'score': game['players'][pid]['score'], 'kicked': False} 
                   for pid in game['player_order']]
    kicked_list = [{'id': pid, 'name': p['name'], 'score': p['score'], 'kicked': True} 
                   for pid, p in game['players'].items() if p.get('kicked')]
    players_list = active_list + kicked_list
    
    socketio.emit('player_kicked', {
        'player_id': player_id,
        'player_name': player_name,
        'reason': reason,
        'players': players_list,
        'game_over': game['status'] == 'finished',
        'active_players': active_players
    }, room=room_code)
    
    return True


def auto_pass_turn(room_code, game):
    """Auto-pass turn when disconnect timeout expires"""
    # Safety check
    if not game.get('player_order'):
        return
    
    current_player_id = game['player_order'][game['current_player_idx']]
    current_player = game['players'].get(current_player_id, {})
    player_name = current_player.get('name', 'Player')
    was_disconnected = not current_player.get('connected', True)
    
    print(f"[AUTO-PASS] Auto-passing turn for {player_name} in room {room_code} (disconnected: {was_disconnected})")
    
    # Track disconnect skips
    if was_disconnected:
        current_player['disconnect_skips'] = current_player.get('disconnect_skips', 0) + 1
        print(f"[AUTO-PASS] {player_name} disconnect_skips now = {current_player['disconnect_skips']}")
        
        # Kick if too many disconnect skips
        if current_player['disconnect_skips'] >= MAX_DISCONNECT_SKIPS:
            kick_player(room_code, game, current_player_id, reason=f"disconnected for {MAX_DISCONNECT_SKIPS} turns")
            
            # After kick, check if game ended
            if game['status'] == 'finished':
                return
            
            # After kick, current_player_idx already points to next player
            # Just notify
            if len(game['player_order']) > 0:
                next_player_id = game['player_order'][game['current_player_idx']]
                players_list = [{'id': pid, 'name': game['players'][pid]['name'], 'score': game['players'][pid]['score']} 
                               for pid in game['player_order']]
                socketio.emit('turn_passed', {
                    'player_id': current_player_id,
                    'player_name': player_name,
                    'reason': 'player_kicked',
                    'current_player_id': next_player_id,
                    'players': players_list,
                    'game_over': False
                }, room=room_code)
            return
    else:
        # Player is connected but timed out - reset their disconnect_skips
        current_player['disconnect_skips'] = 0
    
    # Increment pass count
    game['pass_count'] = game.get('pass_count', 0) + 1
    
    # Move to next player
    game['current_player_idx'] = (game['current_player_idx'] + 1) % len(game['player_order'])
    next_player_id = game['player_order'][game['current_player_idx']]
    
    # Check for game over (all players passed twice consecutively)
    game_over = game['pass_count'] >= len(game['player_order']) * 2
    if game_over:
        game['status'] = 'finished'
        
        # OFFICIAL SCRABBLE END-GAME SCORING (all pass scenario):
        # Everyone subtracts their remaining tile values (no one gains)
        for pid, p in game['players'].items():
            if not p.get('kicked'):
                rack_value = sum(LETTER_VALUES.get(tile, 0) for tile in p['rack'])
                p['score'] -= rack_value
                print(f"[END GAME] (disconnect all pass): {p['name']} loses {rack_value} pts for remaining tiles")
    
    # Notify all players
    players_list = [{'id': pid, 'name': game['players'][pid]['name'], 'score': game['players'][pid]['score']} 
                    for pid in game['player_order']]
    
    # Build clear reason message
    skip_count = current_player.get('disconnect_skips', 0)
    remaining_chances = MAX_DISCONNECT_SKIPS - skip_count
    reason_message = f"disconnected - turn skipped ({remaining_chances} chance{'s' if remaining_chances != 1 else ''} left)"
    
    socketio.emit('turn_passed', {
        'player_id': current_player_id,
        'player_name': player_name,
        'reason': 'disconnect_timeout',
        'reason_message': reason_message,
        'skip_count': skip_count,
        'current_player_id': next_player_id,
        'players': players_list,
        'game_over': game_over
    }, room=room_code)
    
    # Send explicit game_ended event if game is over
    if game_over:
        winner_id = max(game['player_order'], key=lambda pid: game['players'][pid]['score'])
        winner = game['players'][winner_id]
        
        socketio.emit('game_ended', {
            'reason': 'all_passed',
            'winner_id': winner_id,
            'winner_name': winner['name'],
            'winner_score': winner['score'],
            'final_scores': players_list
        }, room=room_code)
        
        print(f"[GAME ENDED] (disconnect all pass) {winner['name']} wins with {winner['score']} pts!")


def check_disconnect_timeouts():
    """Check for disconnected players who need to be auto-passed (LIVE mode, timer-OFF games only)"""
    current_time = time.time()
    
    for room_code, game in list(game_rooms.items()):
        if game.get('status') != 'playing':
            continue
        
        # Skip SLOW games - no disconnect timeout
        if game.get('game_speed') == 'slow':
            continue
        
        # Safety check: make sure player_order is not empty
        if not game.get('player_order'):
            continue
        
        current_player_id = game['player_order'][game['current_player_idx']]
        current_player = game['players'].get(current_player_id, {})
        
        # Only check if current player is disconnected
        if not current_player.get('connected', True):
            disconnect_time = current_player.get('disconnect_time')
            if disconnect_time:
                elapsed = current_time - disconnect_time
                if elapsed >= DISCONNECT_TIMEOUT_SECONDS:
                    print(f"[DISCONNECT TIMEOUT] {current_player.get('name', 'Player')} disconnected for {elapsed:.0f}s, auto-passing")
                    auto_pass_turn(room_code, game)


# ============== WebSocket Events ==============

# Background task to cleanup abandoned games AND check disconnect timeouts
def background_cleanup():
    """Run cleanup every 60 seconds, check disconnect timeouts every 5 seconds"""
    check_counter = 0
    while True:
        socketio.sleep(5)  # Check every 5 seconds
        check_counter += 1
        
        # Check disconnect timeouts every 5 seconds
        check_disconnect_timeouts()
        
        # Run cleanup every 60 seconds (12 x 5 seconds)
        if check_counter >= 12:
            check_counter = 0
            cleanup_abandoned_games()
            print(f"[CLEANUP] Active rooms: {len(game_rooms)}")


# Track if background cleanup has started
_cleanup_started = False

@socketio.on('connect')
def handle_connect():
    global _cleanup_started
    # Start cleanup task on first connection
    if not _cleanup_started:
        _cleanup_started = True
        socketio.start_background_task(background_cleanup)
        print("[STARTUP] Background cleanup task started")
    
    print(f"Client connected: {request.sid}")
    emit('connected', {'sid': request.sid})


@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")
    # Mark player as disconnected but NEVER delete the game
    for room_code, game in list(game_rooms.items()):
        for player_id, player in list(game.get('players', {}).items()):
            if player.get('sid') == request.sid:
                player['connected'] = False
                player['disconnect_time'] = time.time()  # Track when they disconnected
                game['last_activity'] = time.time()  # Track last activity
                # Notify room
                socketio.emit('player_disconnected', {
                    'player_id': player_id,
                    'name': player.get('name', 'Unknown')
                }, room=room_code)
                print(f"[DISCONNECT] Player {player.get('name', 'Unknown')} left room {room_code}")


def cleanup_abandoned_games():
    """ULTRA CONSERVATIVE cleanup - NEVER lose active games"""
    now = time.time()
    for room_code in list(game_rooms.keys()):
        game = game_rooms.get(room_code)
        if not game:
            continue
        
        created = game.get('created_at', now)
        age_hours = (now - created) / 3600
        all_disconnected = all(not p.get('connected', False) for p in game['players'].values())
        
        # ONLY cleanup in these VERY specific cases:
        
        # 1. Empty waiting room (no players ever joined) older than 1 hour
        if game['status'] == 'waiting' and len(game['players']) <= 1 and all_disconnected and age_hours > 1:
            del game_rooms[room_code]
            print(f"Room {room_code}: Cleaned up (empty waiting room >1h)")
            continue
        
        # 2. Finished games older than 4 hours (game is over anyway)
        if game['status'] == 'finished' and age_hours > 4:
            del game_rooms[room_code]
            print(f"Room {room_code}: Cleaned up (finished game >4h)")
            continue
        
        # 3. ANY game older than 48 hours (2 days) - even active ones
        #    (No Scrabble game realistically lasts 2 days)
        if age_hours > 48:
            del game_rooms[room_code]
            print(f"Room {room_code}: Cleaned up (>48h old)")
            continue
        
        # NEVER delete active 'playing' games that are < 48 hours old!


@socketio.on('create_room')
def handle_create_room(data):
    """Create a new game room"""
    print(f"[CREATE_ROOM] Received request: {data}")
    print(f"[CREATE_ROOM] From SID: {request.sid}")
    
    room_code = generate_room_code()
    player_name = data.get('name', 'Player 1')
    game_speed = data.get('game_speed', 'live')  # 'live' = disconnect timeout, 'slow' = no timeout
    
    game_rooms[room_code] = {
        'code': room_code,
        'host_sid': request.sid,
        'status': 'waiting',  # waiting, playing, finished
        'created_at': time.time(),
        'game_speed': game_speed,  # 'live' or 'slow'
        'players': {
            request.sid: {
                'sid': request.sid,
                'name': player_name,
                'rack': [],
                'score': 0,
                'connected': True,
                'order': 0,
                'disconnect_skips': 0,      # Consecutive turns skipped while disconnected
                'disconnect_time': None     # When player disconnected (for timeout)
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
        'host_id': request.sid,
        'game_speed': game_speed,
        'players': [{
            'id': request.sid,
            'name': player_name,
            'score': 0
        }]
    })
    
    print(f"[CREATE_ROOM] Room {room_code} created by {player_name} (speed: {game_speed})")
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
            'players': [{'id': pid, 'name': game['players'][pid]['name'], 'score': game['players'][pid]['score']} 
                       for pid in game['player_order']],
            'status': game['status'],
            'current_player_idx': game['current_player_idx'],
            'bag_count': len(game['bag'])
        })
        print(f"Board viewer joined room {room_code}")
        return
    
    # Check for reconnection (player with same name - connected OR disconnected)
    # This handles the case where wifi cuts quickly and old socket hasn't disconnected yet
    # FORCE TAKEOVER: If same name exists with different socket ID, take over that slot
    reconnecting_player_id = None
    for pid, p in game['players'].items():
        if p['name'] == player_name and pid != request.sid:
            reconnecting_player_id = pid
            break
    
    if reconnecting_player_id:
        old_player = game['players'].get(reconnecting_player_id, {})
        
        # Check if player was KICKED - don't allow reconnection
        if old_player.get('kicked'):
            emit('error', {'message': 'You were removed from this game for being disconnected'})
            print(f"[JOIN_ROOM] {player_name} tried to rejoin but was kicked")
            return
        
        # Reconnecting player - FORCE TAKEOVER of old socket
        old_player = game['players'].pop(reconnecting_player_id)
        old_player['sid'] = request.sid
        old_player['connected'] = True
        old_player['disconnect_skips'] = 0      # Reset - they're back!
        old_player['disconnect_time'] = None    # Reset - no longer disconnected
        game['players'][request.sid] = old_player
        
        # Cancel any pending cleanup
        game.pop('cleanup_after', None)
        
        # Update player_order
        idx = game['player_order'].index(reconnecting_player_id)
        game['player_order'][idx] = request.sid
        
        # Update host_sid if this was the host
        if game['host_sid'] == reconnecting_player_id:
            game['host_sid'] = request.sid
            print(f"Host {player_name} reconnected with new SID")
        
        # Leave old socket room, join with new socket
        leave_room(room_code, sid=reconnecting_player_id)
        join_room(room_code)
        
        # Build player list in CORRECT order (using player_order, not dict order)
        player_list = [{'id': pid, 'name': game['players'][pid]['name'], 'score': game['players'][pid]['score']} 
                       for pid in game['player_order']]
        
        # Check if this player is the host
        is_host = game['host_sid'] == request.sid
        
        # Notify others that player reconnected
        socketio.emit('player_reconnected', {
            'player_id': request.sid,
            'name': player_name,
            'host_id': game['host_sid'],
            'players': player_list
        }, room=room_code, skip_sid=request.sid)
        
        # Send game state to reconnecting player
        emit('room_joined', {
            'room_code': room_code,
            'player_id': request.sid,
            'host_id': game['host_sid'],
            'is_host': is_host,
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
        
        print(f"[RECONNECT] {player_name} FORCE TAKEOVER to room {room_code} (old socket: {reconnecting_player_id[:8]})")
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
        'order': len(game['players']),
        'disconnect_skips': 0,      # Consecutive turns skipped while disconnected
        'disconnect_time': None     # When player disconnected (for timeout)
    }
    game['player_order'].append(request.sid)
    
    join_room(room_code)
    
    # Build player list in CORRECT order (using player_order, not dict order)
    player_list = [{'id': pid, 'name': game['players'][pid]['name'], 'score': game['players'][pid]['score']} 
                   for pid in game['player_order']]
    
    # Notify joiner
    emit('room_joined', {
        'room_code': room_code,
        'player_id': request.sid,
        'host_id': game['host_sid'],
        'players': player_list
    })
    
    # Notify everyone in room
    socketio.emit('player_joined', {
        'player_id': request.sid,
        'name': player_name,
        'host_id': game['host_sid'],
        'players': player_list
    }, room=room_code)
    
    print(f"{player_name} joined room {room_code}")


@socketio.on('start_game')
def handle_start_game(data):
    """Host starts the game"""
    room_code = data.get('room_code', '').upper()
    wrong_word_rule = data.get('wrong_word_rule', 1)  # Default: 1 retry allowed
    
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
    
    # Store game settings
    game['wrong_word_rule'] = wrong_word_rule  # 0 = zero tolerance, 1 = one retry
    game['wrong_attempts'] = {}  # {player_id: count} - reset each turn
    
    # Deal tiles to all players
    for player_id in game['player_order']:
        game['players'][player_id]['rack'] = draw_tiles(game['bag'], 7)
    
    game['status'] = 'playing'
    game['current_player_idx'] = 0
    
    # Build player list in CORRECT order (using player_order, not dict order)
    player_list = [{'id': pid, 'name': game['players'][pid]['name'], 'score': game['players'][pid]['score']} 
                   for pid in game['player_order']]
    
    # Send each player their rack privately
    for player_id, player in game['players'].items():
        socketio.emit('game_started', {
            'your_rack': player['rack'],
            'current_player_idx': 0,
            'current_player_id': game['player_order'][0],
            'players': player_list,
            'bag_count': len(game['bag']),
            'wrong_word_rule': wrong_word_rule
        }, room=player_id)
    
    # Send to board viewers
    for viewer_sid in game['board_viewers']:
        socketio.emit('game_started', {
            'current_player_idx': 0,
            'current_player_id': game['player_order'][0],
            'players': player_list,
            'bag_count': len(game['bag']),
            'wrong_word_rule': wrong_word_rule
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
        # Get wrong word rule settings
        wrong_word_rule = game.get('wrong_word_rule', 1)  # Default 1 retry
        
        # Track wrong attempts for this player
        if 'wrong_attempts' not in game:
            game['wrong_attempts'] = {}
        
        current_attempts = game['wrong_attempts'].get(request.sid, 0)
        game['wrong_attempts'][request.sid] = current_attempts + 1
        
        attempts_allowed = wrong_word_rule  # 0 or 1
        attempts_left = attempts_allowed - game['wrong_attempts'][request.sid] + 1
        
        # Broadcast wrong word to all players
        socketio.emit('wrong_word', {
            'player_id': request.sid,
            'player_name': player['name'],
            'words': invalid_words,
            'attempts_left': max(0, attempts_left)
        }, room=room_code)
        
        # Check if turn should be lost
        if game['wrong_attempts'][request.sid] > attempts_allowed:
            # Turn lost - advance to next player
            game['wrong_attempts'][request.sid] = 0  # Reset for next turn
            
            # Advance turn
            game['current_player_idx'] = (game['current_player_idx'] + 1) % len(game['player_order'])
            next_player_id = game['player_order'][game['current_player_idx']]
            
            # Reset wrong attempts for new player
            game['wrong_attempts'][next_player_id] = 0
            
            # Notify all players about turn loss
            socketio.emit('turn_lost', {
                'player_id': request.sid,
                'player_name': player['name'],
                'reason': f'Invalid word(s): {", ".join(invalid_words)}',
                'current_player_idx': game['current_player_idx'],
                'current_player_id': next_player_id,
                'players': [{'id': pid, 'name': game['players'][pid]['name'], 'score': game['players'][pid]['score']} 
                           for pid in game['player_order']]
            }, room=room_code)
            return
        
        # Still have retries - just reject the move
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
    
    # DEBUG: Log rack and bag state for game-end diagnosis
    print(f"[DEBUG] After move by {player['name']}: rack={player['rack']} (len={len(player['rack'])}), bag_count={len(game['bag'])}")
    
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
    
    # Reset wrong attempts for the new player's turn
    if 'wrong_attempts' in game:
        game['wrong_attempts'][next_player_id] = 0
    
    # Check for game end
    game_over = False
    print(f"[DEBUG] Checking game end: rack_empty={len(player['rack'])==0}, bag_empty={len(game['bag'])==0}")
    if len(player['rack']) == 0 and len(game['bag']) == 0:
        game_over = True
        game['status'] = 'finished'
        
        # OFFICIAL SCRABBLE END-GAME SCORING:
        # Player who went out gains all opponents' remaining tile values
        # Opponents subtract their remaining tile values from their scores
        total_remaining = 0
        for pid, p in game['players'].items():
            if pid != request.sid:  # Not the winner
                rack_value = sum(LETTER_VALUES.get(tile, 0) for tile in p['rack'])
                p['score'] -= rack_value  # Loser subtracts their tiles
                total_remaining += rack_value
        
        # Winner adds all remaining tile values
        player['score'] += total_remaining
        
        print(f"END GAME: {player['name']} went out, gained {total_remaining} pts from opponents' tiles")
    
    # Send updated state to everyone
    # Build player list in CORRECT order (using player_order, not dict order)
    player_list = [{'id': pid, 'name': game['players'][pid]['name'], 'score': game['players'][pid]['score']} 
                   for pid in game['player_order']]
    
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
    
    # Send explicit game_ended event if game is over
    if game_over:
        # Determine winner (highest score)
        winner_id = max(game['player_order'], key=lambda pid: game['players'][pid]['score'])
        winner = game['players'][winner_id]
        
        socketio.emit('game_ended', {
            'reason': 'player_finished',
            'winner_id': winner_id,
            'winner_name': winner['name'],
            'winner_score': winner['score'],
            'final_scores': player_list,
            'finished_by': player['name']
        }, room=room_code)
        
        print(f"GAME ENDED: {winner['name']} wins with {winner['score']} pts!")
    
    print(f"Move played in {room_code}: {words_formed} for {score} pts")


@socketio.on('exchange_tiles')
def handle_exchange_tiles(data):
    """Player exchanges tiles with the bag"""
    room_code = data.get('room_code', '').upper()
    tiles_to_exchange = data.get('tiles', [])
    
    if room_code not in game_rooms:
        emit('error', {'message': 'Room not found'})
        return
    
    game = game_rooms[room_code]
    current_player_id = game['player_order'][game['current_player_idx']]
    
    if request.sid != current_player_id:
        emit('error', {'message': 'Not your turn'})
        return
    
    if len(game['bag']) < len(tiles_to_exchange):
        emit('error', {'message': f'Not enough tiles in bag ({len(game["bag"])} remaining)'})
        return
    
    player = game['players'][request.sid]
    
    # Remove tiles from rack and add to bag
    for letter in tiles_to_exchange:
        if letter in player['rack']:
            player['rack'].remove(letter)
            game['bag'].append(letter)
    
    # Shuffle bag
    import random
    random.shuffle(game['bag'])
    
    # Draw new tiles
    new_tiles = game['bag'][:len(tiles_to_exchange)]
    game['bag'] = game['bag'][len(tiles_to_exchange):]
    player['rack'].extend(new_tiles)
    
    # Reset pass count (exchange is not a pass)
    game['pass_count'] = 0
    
    # Next turn
    game['current_player_idx'] = (game['current_player_idx'] + 1) % len(game['player_order'])
    next_player_id = game['player_order'][game['current_player_idx']]
    
    # Reset wrong attempts for the new player's turn
    if 'wrong_attempts' in game:
        game['wrong_attempts'][next_player_id] = 0
    
    # Build player list in CORRECT order (using player_order, not dict order)
    player_list = [{'id': pid, 'name': game['players'][pid]['name'], 'score': game['players'][pid]['score']} 
                   for pid in game['player_order']]
    
    # Notify the player who exchanged (with their new rack)
    emit('tiles_exchanged', {
        'your_rack': player['rack'],
        'tiles_exchanged': len(tiles_to_exchange),
        'current_player_idx': game['current_player_idx'],
        'current_player_id': next_player_id,
        'players': player_list,
        'bag_count': len(game['bag'])
    })
    
    # Notify everyone else
    socketio.emit('player_exchanged', {
        'player_id': request.sid,
        'player_name': player['name'],
        'tiles_exchanged': len(tiles_to_exchange),
        'current_player_idx': game['current_player_idx'],
        'current_player_id': next_player_id,
        'players': player_list,
        'bag_count': len(game['bag'])
    }, room=room_code, skip_sid=request.sid)


@socketio.on('used_help')
def handle_used_help(data):
    """Notify all players when someone uses Hint or Clue"""
    room_code = data.get('room_code', '').upper()
    help_type = data.get('type', 'hint')  # 'hint' or 'clue'
    
    if room_code not in game_rooms:
        return
    
    game = game_rooms[room_code]
    player = game['players'].get(request.sid)
    
    if not player:
        return
    
    # Notify everyone else
    socketio.emit('player_used_help', {
        'player_id': request.sid,
        'player_name': player['name'],
        'type': help_type
    }, room=room_code, skip_sid=request.sid)
    
    print(f"{player['name']} used {help_type} in room {room_code}")


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
    
    # Game ends if all players pass twice (6 consecutive passes for 2 players)
    game_over = game['pass_count'] >= len(game['players']) * 2
    if game_over:
        game['status'] = 'finished'
        
        # OFFICIAL SCRABBLE END-GAME SCORING (all pass scenario):
        # Everyone subtracts their remaining tile values (no one gains)
        for pid, p in game['players'].items():
            rack_value = sum(LETTER_VALUES.get(tile, 0) for tile in p['rack'])
            p['score'] -= rack_value
            print(f"END GAME (all pass): {p['name']} loses {rack_value} pts for remaining tiles")
    
    next_player_id = game['player_order'][game['current_player_idx']]
    
    # Reset wrong attempts for the new player's turn
    if 'wrong_attempts' in game:
        game['wrong_attempts'][next_player_id] = 0
    
    # Build player list in CORRECT order (using player_order, not dict order)
    player_list = [{'id': pid, 'name': game['players'][pid]['name'], 'score': game['players'][pid]['score']} 
                   for pid in game['player_order']]
    
    # Notify everyone
    socketio.emit('turn_passed', {
        'player_id': request.sid,
        'player_name': game['players'][request.sid]['name'],
        'current_player_idx': game['current_player_idx'],
        'current_player_id': next_player_id,
        'players': player_list,
        'game_over': game_over
    }, room=room_code)
    
    # Send explicit game_ended event if game is over (all passed)
    if game_over:
        # Determine winner (highest score)
        winner_id = max(game['player_order'], key=lambda pid: game['players'][pid]['score'])
        winner = game['players'][winner_id]
        
        socketio.emit('game_ended', {
            'reason': 'all_passed',
            'winner_id': winner_id,
            'winner_name': winner['name'],
            'winner_score': winner['score'],
            'final_scores': player_list
        }, room=room_code)
        
        print(f"GAME ENDED (all passed): {winner['name']} wins with {winner['score']} pts!")


@socketio.on('get_state')
def handle_get_state(data):
    """Get current game state (for reconnection)"""
    room_code = data.get('room_code', '').upper()
    
    if room_code not in game_rooms:
        emit('error', {'message': 'Room not found'})
        return
    
    game = game_rooms[room_code]
    player = game['players'].get(request.sid)
    
    # Build player list in CORRECT order (using player_order, not dict order)
    player_list = [{'id': pid, 'name': game['players'][pid]['name'], 'score': game['players'][pid]['score']} 
                   for pid in game['player_order']]
    
    emit('game_state', {
        'room_code': room_code,
        'board': game['board'],
        'your_rack': player['rack'] if player else [],
        'players': player_list,
        'current_player_idx': game['current_player_idx'],
        'current_player_id': game['player_order'][game['current_player_idx']] if game['player_order'] else None,
        'status': game['status'],
        'bag_count': len(game['bag']),
        'last_move': game['last_move']
    })


if __name__ == '__main__':
    # For local development only - production uses gunicorn
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
