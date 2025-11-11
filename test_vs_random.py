#!/usr/bin/env python3
#test our trained model against a bot that picks random actions

import torch
import random
import asyncio
from cfr_min import NFSPAgent, encode_state
from game import PokerGame

class RandomPlayer:
    #Picks random actions
    
    def get_action(self, legal_actions):
        return random.choice(legal_actions)

class TrainedPlayer:
    #Trained NFSP model
    
    def __init__(self):
        self.agent = NFSPAgent(device='cpu')
        
        try:
            self.agent.q_network.load_state_dict(
                torch.load('q_network_p1.pth', map_location='cpu')
            )
            self.agent.policy_network.load_state_dict(
                torch.load('policy_network_p1.pth', map_location='cpu')
            )
            self.agent.q_network.eval()
            self.agent.policy_network.eval()
            print("✓ Loaded trained model\n")
        except FileNotFoundError:
            print("⚠ Warning: Model files not found, using random initialization\n")
    
    def get_action(self, hand, boards, pot, stacks, position, street, bet_to_call, legal_actions):
        #Use policy network (average strategy)
        state = encode_state(hand, boards, pot, stacks, position, street, bet_to_call)
        
        with torch.no_grad():
            action, _ = self.agent.select_action(state, legal_actions, mode='sl')
        
        return action

async def play_hand(trained_player, random_player, trained_position=0):
    #Play one hand, return chip profit for trained player
    stacks = [100, 100]
    initial_stacks = [100, 100]
    big_blind = 1
    dealer = random.choice([0, 1])
    
    game = PokerGame(stacks[:], big_blind, dealer)
    
    cards = list(range(1, 53))
    random.shuffle(cards)
    hands = game.deal_hands(cards)
    boards = game.deal_flops(cards)
    
    pot = 2 * big_blind
    game.stacks[0] -= big_blind
    game.stacks[1] -= big_blind
    
    for street_idx, street_name in enumerate(['FLOP', 'TURN']):
        if street_idx == 1:
            boards = game.deal_turns(cards, boards[0], boards[1])
        
        first_player = 1 - dealer
        second_player = dealer
        if first_player == trained_position:
            action1 = trained_player.get_action(
                hands[first_player], boards, pot,
                [game.stacks[first_player], game.stacks[second_player]],
                first_player, street_name, 0, [0, 1]
            )
        else:
            action1 = random_player.get_action([0, 1])
        
        bet_to_call = 0
        if action1 == 1:
            bet_amount = min(pot, game.stacks[first_player], game.stacks[second_player])
            game.stacks[first_player] -= bet_amount
            pot += bet_amount
            bet_to_call = bet_amount
        
        legal2 = [2, 3] if bet_to_call > 0 else [0, 1]
        if second_player == trained_position:
            action2 = trained_player.get_action(
                hands[second_player], boards, pot,
                [game.stacks[second_player], game.stacks[first_player]],
                second_player, street_name, bet_to_call, legal2
            )
        else:
            action2 = random_player.get_action(legal2)
        
        if bet_to_call > 0:
            if action2 == 3:
                winner = first_player
                game.stacks[winner] += pot
                return game.stacks[trained_position] - initial_stacks[trained_position]
            elif action2 == 2:
                call_amount = min(bet_to_call, game.stacks[second_player])
                game.stacks[second_player] -= call_amount
                pot += call_amount
        else:
            if action2 == 1:
                bet_amount = min(pot, game.stacks[second_player], game.stacks[first_player])
                game.stacks[second_player] -= bet_amount
                pot += bet_amount
                if first_player == trained_position:
                    action3 = trained_player.get_action(
                        hands[first_player], boards, pot,
                        [game.stacks[first_player], game.stacks[second_player]],
                        first_player, street_name, bet_amount, [2, 3]
                    )
                else:
                    action3 = random_player.get_action([2, 3])
                
                if action3 == 3:
                    winner = second_player
                    game.stacks[winner] += pot
                    return game.stacks[trained_position] - initial_stacks[trained_position]
                else:
                    game.stacks[first_player] -= bet_amount
                    pot += bet_amount
    
    boards = game.deal_rivers(cards, boards[0], boards[1])
    
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
    
    return game.stacks[trained_position] - initial_stacks[trained_position]

async def test_vs_random(num_hands=1000):
    
    print("="*60)
    print("TESTING: Trained Model vs Random Player")
    print("="*60)
    
    trained = TrainedPlayer()
    random_player = RandomPlayer()
    
    total_chips = 0
    chip_results = []
    
    print(f"Playing {num_hands} hands...\n")
    
    for i in range(num_hands):
        trained_position = i % 2
        chip_profit = await play_hand(trained, random_player, trained_position)
        
        total_chips += chip_profit
        chip_results.append(chip_profit)
        
        if (i + 1) % 100 == 0:
            mean_ev = total_chips / (i + 1)
            mse = sum((r - mean_ev) ** 2 for r in chip_results) / len(chip_results)
            print(f"  {i+1:4d} hands | Total chips: {total_chips:+6.1f} | "
                  f"EV/hand: {mean_ev:+6.2f} bb/hand | MSE: {mse:.2f}")
    
    mean_ev = total_chips / num_hands
    mse = sum((r - mean_ev) ** 2 for r in chip_results) / len(chip_results)
    std_dev = mse ** 0.5
    
    wins = sum(1 for r in chip_results if r > 0)
    losses = sum(1 for r in chip_results if r < 0)
    ties = sum(1 for r in chip_results if r == 0)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total hands:      {num_hands}")
    print(f"Total chips:      {total_chips:+.1f} chips")
    print(f"Mean EV/hand:     {mean_ev:+.3f} bb/hand")
    print(f"Std deviation:    {std_dev:.3f} bb/hand")
    print(f"MSE:              {mse:.3f}")
    print(f"\nWin/Loss/Tie:     {wins}/{losses}/{ties} ({wins/num_hands*100:.1f}%/{losses/num_hands*100:.1f}%/{ties/num_hands*100:.1f}%)")
    
    print("\n" + "-"*60)
    if mean_ev > 0.5:
        print(f"✓ Crushing random! (+{mean_ev:.2f} bb/hand)")
    elif mean_ev > 0.2:
        print(f"✓ Clearly better (+{mean_ev:.2f} bb/hand)")
    elif mean_ev > 0.05:
        print(f"~ Slightly better (+{mean_ev:.2f} bb/hand)")
    elif mean_ev > -0.05:
        print(f"≈ Roughly equal ({mean_ev:+.2f} bb/hand)")
    else:
        print(f"⚠ Losing to random ({mean_ev:+.2f} bb/hand)")
    
    print(f"\nExpected vs random: +0.3 to +0.5 bb/hand")
    print("="*60)

def main():
    import sys
    
    # Get number of hands from command line, default 1000
    num_hands = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    
    asyncio.run(test_vs_random(num_hands))

if __name__ == "__main__":
    main()

