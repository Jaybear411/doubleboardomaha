#!/usr/bin/env python3
"""Flask app to play Double Board Omaha against trained bot"""

from flask import Flask, render_template, jsonify, request, session
import torch
import random
import os
from nfsp import NFSPAgent, encode_state, get_legal_actions, apply_action, action_name, STATE_DIM
from game import PokerGame

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load trained bot
bot = NFSPAgent(state_dim=STATE_DIM, device='cpu')
try:
    bot.q_network.load_state_dict(torch.load('q_network_p1.pth', map_location='cpu', weights_only=True))
    bot.policy_network.load_state_dict(torch.load('policy_network_p1.pth', map_location='cpu', weights_only=True))
    bot.q_network.eval()
    bot.policy_network.eval()
    print("✓ Loaded trained model")
except FileNotFoundError:
    print("⚠ Model not found, using random")


def card_to_str(card):
    """Convert card number to string like 'As', 'Kh'"""
    ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    suits = ['s', 'h', 'd', 'c']
    return ranks[(card - 1) % 13] + suits[(card - 1) // 13]


def init_game_state():
    """Initialize new game"""
    cards = list(range(1, 53))
    random.shuffle(cards)
    
    return {
        'cards': cards,
        'player_hand': cards[:4],
        'bot_hand': cards[4:8],
        'board1': cards[8:11],
        'board2': cards[11:14],
        'street': 'FLOP',
        'street_idx': 0,
        'pot': 2,
        'player_stack': 99,
        'bot_stack': 99,
        'dealer': random.choice([0, 1]),  # 0=player dealer, 1=bot dealer
        'current_bet': 0,
        'game_over': False,
        'winner': None,
        'message': '',
        'bot_strategy': 'sl',
        'action_on': None,
        'street_actions': [],
    }


def get_bot_action(state, legal):
    """Get bot's action"""
    bot_state = encode_state(
        state['bot_hand'],
        [state['board1'], state['board2']],
        state['pot'],
        [state['bot_stack'], state['player_stack']],
        0,  # Position 0 for agent
        state['street'],
        state['current_bet']
    )
    
    strategy = state.get('bot_strategy', 'sl')
    with torch.no_grad():
        action, _ = bot.select_action(bot_state, legal, mode=strategy)
    return action


def process_action(state, actor, action):
    """Process an action, return (bet_amount, is_fold)"""
    if actor == 'player':
        hero_stack = state['player_stack']
        villain_stack = state['bot_stack']
    else:
        hero_stack = state['bot_stack']
        villain_stack = state['player_stack']
    
    return apply_action(action, state['current_bet'], hero_stack, villain_stack, state['pot'])


def advance_street(state):
    """Advance to next street"""
    cards = state['cards']
    
    if state['street'] == 'FLOP':
        state['street'] = 'TURN'
        state['street_idx'] = 1
        state['board1'] = state['board1'] + [cards[14]]
        state['board2'] = state['board2'] + [cards[15]]
    elif state['street'] == 'TURN':
        state['street'] = 'RIVER'
        state['street_idx'] = 2
        state['board1'] = state['board1'] + [cards[16]]
        state['board2'] = state['board2'] + [cards[17]]
        do_showdown(state)
        return
    
    state['current_bet'] = 0
    state['street_actions'] = []


def do_showdown(state):
    """Evaluate hands"""
    game = PokerGame([100, 100], 1, 0)
    
    best_b1_player, _ = game.omaha_hand_strength(state['player_hand'], state['board1'])
    best_b2_player, _ = game.omaha_hand_strength(state['player_hand'], state['board2'])
    best_b1_bot, _ = game.omaha_hand_strength(state['bot_hand'], state['board1'])
    best_b2_bot, _ = game.omaha_hand_strength(state['bot_hand'], state['board2'])
    
    player_wins, bot_wins = 0, 0
    results = []
    
    if best_b1_player > best_b1_bot:
        player_wins += 0.5
        results.append("You win Board 1!")
    elif best_b1_player < best_b1_bot:
        bot_wins += 0.5
        results.append("Bot wins Board 1")
    else:
        player_wins += 0.25
        bot_wins += 0.25
        results.append("Board 1 tie")
    
    if best_b2_player > best_b2_bot:
        player_wins += 0.5
        results.append("You win Board 2!")
    elif best_b2_player < best_b2_bot:
        bot_wins += 0.5
        results.append("Bot wins Board 2")
    else:
        player_wins += 0.25
        bot_wins += 0.25
        results.append("Board 2 tie")
    
    pot = state['pot']
    state['player_stack'] += int(player_wins * pot)
    state['bot_stack'] += int(bot_wins * pot)
    
    state['game_over'] = True
    if player_wins > bot_wins:
        state['winner'] = 'player'
        state['message'] = f"You win! {' | '.join(results)}"
    elif bot_wins > player_wins:
        state['winner'] = 'bot'
        state['message'] = f"Bot wins. {' | '.join(results)}"
    else:
        state['winner'] = 'tie'
        state['message'] = f"Tie! {' | '.join(results)}"


def get_client_state(state):
    """State for client"""
    # Get legal actions for player
    if state['action_on'] == 'player' and not state['game_over']:
        legal = get_legal_actions(
            state['current_bet'],
            state['player_stack'],
            state['bot_stack'],
            state['pot']
        )
    else:
        legal = []
    
    facing_bet = state['current_bet'] > 0
    
    return {
        'player_hand': [card_to_str(c) for c in state['player_hand']],
        'bot_hand': [card_to_str(c) for c in state['bot_hand']] if state['game_over'] else ['??'] * 4,
        'board1': [card_to_str(c) for c in state['board1']],
        'board2': [card_to_str(c) for c in state['board2']],
        'street': state['street'],
        'pot': state['pot'],
        'player_stack': state['player_stack'],
        'bot_stack': state['bot_stack'],
        'current_bet': state['current_bet'],
        'game_over': state['game_over'],
        'winner': state['winner'],
        'message': state['message'],
        'action_on': state['action_on'],
        'legal_actions': legal,
        'facing_bet': facing_bet,
        'dealer': 'You' if state['dealer'] == 0 else 'Bot',
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/new_game', methods=['POST'])
def new_game():
    """Start new game"""
    data = request.json or {}
    strategy = data.get('strategy', 'sl')
    
    state = init_game_state()
    state['bot_strategy'] = strategy
    
    # Who acts first (non-dealer)
    first = 'bot' if state['dealer'] == 0 else 'player'
    state['action_on'] = first
    
    # If bot first, get its action
    if first == 'bot':
        legal = get_legal_actions(0, state['bot_stack'], state['player_stack'], state['pot'])
        bot_action = get_bot_action(state, legal)
        state['street_actions'].append(('bot', bot_action))
        
        bet_amount, _ = process_action(state, 'bot', bot_action)
        
        if bet_amount > 0:
            state['bot_stack'] -= bet_amount
            state['pot'] += bet_amount
            state['current_bet'] = bet_amount
            state['message'] = f"Bot {action_name(bot_action, False)} ({bet_amount})"
        else:
            state['message'] = "Bot checks"
        
        state['action_on'] = 'player'
    
    session['game_state'] = state
    return jsonify(get_client_state(state))


@app.route('/api/action', methods=['POST'])
def player_action():
    """Handle player action"""
    state = session.get('game_state')
    if not state or state['game_over']:
        return jsonify({'error': 'No active game'}), 400
    
    data = request.json
    action = data.get('action')
    
    if state['action_on'] != 'player':
        return jsonify({'error': 'Not your turn'}), 400
    
    legal = get_legal_actions(
        state['current_bet'],
        state['player_stack'],
        state['bot_stack'],
        state['pot']
    )
    
    if action not in legal:
        return jsonify({'error': f'Illegal action {action}. Legal: {legal}'}), 400
    
    facing_bet = state['current_bet'] > 0
    state['street_actions'].append(('player', action))
    
    bet_amount, is_fold = process_action(state, 'player', action)
    
    if is_fold:
        state['bot_stack'] += state['pot']
        state['game_over'] = True
        state['winner'] = 'bot'
        state['message'] = "You fold. Bot wins."
        session['game_state'] = state
        return jsonify(get_client_state(state))
    
    if bet_amount > 0:
        state['player_stack'] -= bet_amount
        state['pot'] += bet_amount
        
        if state['current_bet'] > 0 and bet_amount <= state['current_bet']:
            # Call - end of action
            state['message'] = f"You call {bet_amount}"
            state['current_bet'] = 0
            
            if state['street'] == 'RIVER':
                do_showdown(state)
            else:
                advance_street(state)
                _start_new_street(state)
        else:
            # Bet or raise
            state['message'] = f"You {action_name(action, facing_bet)} ({bet_amount})"
            state['current_bet'] = bet_amount
            
            # Bot responds
            _bot_responds(state)
    else:
        # Check
        state['message'] = "You check"
        
        if len(state['street_actions']) >= 2:
            # Both checked
            if state['street'] == 'RIVER':
                do_showdown(state)
            else:
                advance_street(state)
                _start_new_street(state)
        else:
            # Bot acts
            _bot_responds(state)
    
    session['game_state'] = state
    return jsonify(get_client_state(state))


def _bot_responds(state):
    """Bot responds to player action"""
    legal = get_legal_actions(
        state['current_bet'],
        state['bot_stack'],
        state['player_stack'],
        state['pot']
    )
    
    facing_bet = state['current_bet'] > 0
    bot_action = get_bot_action(state, legal)
    state['street_actions'].append(('bot', bot_action))
    
    bet_amount, is_fold = process_action(state, 'bot', bot_action)
    
    if is_fold:
        state['player_stack'] += state['pot']
        state['game_over'] = True
        state['winner'] = 'player'
        state['message'] += f" → Bot folds. You win!"
        return
    
    if bet_amount > 0:
        state['bot_stack'] -= bet_amount
        state['pot'] += bet_amount
        
        if state['current_bet'] > 0 and bet_amount <= state['current_bet']:
            # Bot calls
            state['message'] += f" → Bot calls {bet_amount}"
            state['current_bet'] = 0
            
            if state['street'] == 'RIVER':
                do_showdown(state)
            else:
                advance_street(state)
                _start_new_street(state)
        else:
            # Bot raises
            state['message'] += f" → Bot {action_name(bot_action, facing_bet)} ({bet_amount})"
            state['current_bet'] = bet_amount
            state['action_on'] = 'player'
    else:
        # Bot checks
        state['message'] += " → Bot checks"
        
        if len(state['street_actions']) >= 2:
            if state['street'] == 'RIVER':
                do_showdown(state)
            else:
                advance_street(state)
                _start_new_street(state)
        else:
            state['action_on'] = 'player'


def _start_new_street(state):
    """Start new street"""
    if state['game_over']:
        return
    
    first = 'bot' if state['dealer'] == 0 else 'player'
    state['action_on'] = first
    state['street_actions'] = []
    
    if first == 'bot':
        legal = get_legal_actions(0, state['bot_stack'], state['player_stack'], state['pot'])
        bot_action = get_bot_action(state, legal)
        state['street_actions'].append(('bot', bot_action))
        
        bet_amount, _ = process_action(state, 'bot', bot_action)
        
        if bet_amount > 0:
            state['bot_stack'] -= bet_amount
            state['pot'] += bet_amount
            state['current_bet'] = bet_amount
            state['message'] += f" → {state['street']}: Bot {action_name(bot_action, False)} ({bet_amount})"
        else:
            state['message'] += f" → {state['street']}: Bot checks"
        
        state['action_on'] = 'player'


if __name__ == '__main__':
    app.run(debug=True, port=5000)
