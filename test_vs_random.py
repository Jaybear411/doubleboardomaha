#!/usr/bin/env python3
"""Test trained NFSP model against random player"""

import torch
import random
import asyncio
from nfsp import NFSPAgent, encode_state, get_legal_actions, apply_action, STATE_DIM
from game import PokerGame


class RandomPlayer:
    """Picks random actions"""
    def get_action(self, legal_actions):
        return random.choice(legal_actions)


class TrainedPlayer:
    """Trained NFSP model"""
    def __init__(self):
        self.agent = NFSPAgent(state_dim=STATE_DIM, device='cpu')
        
        try:
            self.agent.q_network.load_state_dict(
                torch.load('q_network_p1.pth', map_location='cpu', weights_only=True)
            )
            self.agent.policy_network.load_state_dict(
                torch.load('policy_network_p1.pth', map_location='cpu', weights_only=True)
            )
            self.agent.q_network.eval()
            self.agent.policy_network.eval()
            self.agent.epsilon = 0  # No exploration during testing
            print("✓ Loaded trained model\n")
        except FileNotFoundError:
            print("⚠ Model files not found, using random\n")
    
    def get_action(self, hand, boards, pot, stacks, position, street, bet_to_call, legal_actions):
        state = encode_state(hand, boards, pot, stacks, position, street, bet_to_call)
        with torch.no_grad():
            # Use RL (Q-network) - the trained policy
            action, _ = self.agent.select_action(state, legal_actions, mode='rl')
        return action


async def play_hand(trained, random_player, trained_pos=0):
    """Play one hand, return chip profit for trained player"""
    stacks = [100, 100]
    initial = [100, 100]
    bb = 1
    dealer = random.choice([0, 1])
    
    game = PokerGame(stacks[:], bb, dealer)
    
    cards = list(range(1, 53))
    random.shuffle(cards)
    hands = game.deal_hands(cards)
    boards = game.deal_flops(cards)
    
    pot = 2 * bb
    game.stacks[0] -= bb
    game.stacks[1] -= bb
    
    for street_idx, street_name in enumerate(['FLOP', 'TURN', 'RIVER']):
        if street_idx == 1:
            boards = game.deal_turns(cards, boards[0], boards[1])
        elif street_idx == 2:
            boards = game.deal_rivers(cards, boards[0], boards[1])
        
        first = 1 - dealer
        second = dealer
        current_bet = 0
        
        for action_num in range(4):
            if action_num % 2 == 0:
                actor = first
            else:
                actor = second
            
            hero_stack = game.stacks[actor]
            villain_stack = game.stacks[1 - actor]
            
            if hero_stack <= 0:
                break
            
            legal = get_legal_actions(current_bet, hero_stack, villain_stack, pot)
            
            if actor == trained_pos:
                action = trained.get_action(
                    hands[actor], boards, pot,
                    [hero_stack, villain_stack],
                    actor, street_name, current_bet, legal
                )
            else:
                action = random_player.get_action(legal)
            
            bet_amount, is_fold = apply_action(action, current_bet, hero_stack, villain_stack, pot)
            
            if is_fold:
                winner = 1 - actor
                game.stacks[winner] += pot
                return game.stacks[trained_pos] - initial[trained_pos]
            
            if bet_amount > 0:
                game.stacks[actor] -= bet_amount
                pot += bet_amount
                
                if current_bet > 0:
                    if bet_amount <= current_bet:
                        current_bet = 0
                        break
                    else:
                        current_bet = bet_amount
                else:
                    current_bet = bet_amount
            else:
                if action_num > 0 and current_bet == 0:
                    break
    
    # Showdown
    best_b1_p1, _ = game.omaha_hand_strength(hands[0], boards[0])
    best_b2_p1, _ = game.omaha_hand_strength(hands[0], boards[1])
    best_b1_p2, _ = game.omaha_hand_strength(hands[1], boards[0])
    best_b2_p2, _ = game.omaha_hand_strength(hands[1], boards[1])
    
    w1, w2 = 0, 0
    if best_b1_p1 > best_b1_p2: w1 += 0.5
    elif best_b1_p1 < best_b1_p2: w2 += 0.5
    else: w1 += 0.25; w2 += 0.25
    
    if best_b2_p1 > best_b2_p2: w1 += 0.5
    elif best_b2_p1 < best_b2_p2: w2 += 0.5
    else: w1 += 0.25; w2 += 0.25
    
    game.stacks[0] += int(w1 * pot)
    game.stacks[1] += int(w2 * pot)
    
    return game.stacks[trained_pos] - initial[trained_pos]


async def test_vs_random(num_hands=1000):
    print("=" * 60)
    print("TESTING: Trained Model vs Random Player")
    print("=" * 60 + "\n")
    
    trained = TrainedPlayer()
    random_player = RandomPlayer()
    
    total = 0
    results = []
    
    print(f"Playing {num_hands} hands...\n")
    
    for i in range(num_hands):
        pos = i % 2
        profit = await play_hand(trained, random_player, pos)
        total += profit
        results.append(profit)
        
        if (i + 1) % 200 == 0:
            ev = total / (i + 1)
            print(f"  {i+1:4d} hands | Total: {total:+6.1f} | EV: {ev:+.2f} bb/hand")
    
    ev = total / num_hands
    std = (sum((r - ev) ** 2 for r in results) / len(results)) ** 0.5
    
    wins = sum(1 for r in results if r > 0)
    losses = sum(1 for r in results if r < 0)
    ties = sum(1 for r in results if r == 0)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total hands:    {num_hands}")
    print(f"Total chips:    {total:+.1f}")
    print(f"EV/hand:        {ev:+.3f} bb/hand")
    print(f"Std dev:        {std:.3f}")
    print(f"Win/Loss/Tie:   {wins}/{losses}/{ties}")
    
    print("\n" + "-" * 60)
    if ev > 0.5:
        print(f"✓ Crushing random! (+{ev:.2f} bb/hand)")
    elif ev > 0.2:
        print(f"✓ Clearly better (+{ev:.2f} bb/hand)")
    elif ev > -0.05:
        print(f"≈ Roughly equal ({ev:+.2f} bb/hand)")
    else:
        print(f"⚠ Losing to random ({ev:+.2f} bb/hand)")
    print("=" * 60)


def main():
    import sys
    num_hands = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    asyncio.run(test_vs_random(num_hands))


if __name__ == "__main__":
    main()
