"""Test random against heuristic linear player"""

import torch
import random
import asyncio
from nfsp import NFSPAgent, encode_state, get_legal_actions, apply_action, STATE_DIM
from game import PokerGame


class LinearPlayer:
    #rule-based: bet/raise with 2 pair+, otherwise check/fold
    def __init__(self):
        self.game = PokerGame([100, 100], 1, 0)
    
    def get_action(self, hand, boards, legal_actions, bet_to_call):
        best_rank = 0
        
        if boards and len(boards) > 0:
            if boards[0] and len(boards[0]) >= 3:
                rank1, _ = self.game.omaha_hand_strength(hand, boards[0])
                best_rank = max(best_rank, rank1[0])
            
            if len(boards) > 1 and boards[1] and len(boards[1]) >= 3:
                rank2, _ = self.game.omaha_hand_strength(hand, boards[1])
                best_rank = max(best_rank, rank2[0])
        
        has_strong = best_rank >= 2  # two pair or better
        
        if bet_to_call > 0:
            if has_strong:
                if 1 in legal_actions and random.random() < 0.4:
                    return 1  # raise pot
                return 0  # call
            else:
                return 3  # fold (fold)
        else:
            if has_strong:
                if random.random() < 0.6:
                    return 2  # bet pot
                else:
                    return 1  # bet half
            else:
                return 0  # check


class RandomPlayer:
    #picks random actions
    def get_action(self, legal_actions):
        return random.choice(legal_actions)


async def play_hand(random_player, linear, random_pos=0):
    #play one hand, return chip profit for random player
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
            
            if actor == random_pos:
                action = random_player.get_action(legal)
            else:
                action = linear.get_action(hands[actor], boards, legal, current_bet)
            
            bet_amount, is_fold = apply_action(action, current_bet, hero_stack, villain_stack, pot)
            
            if is_fold:
                winner = 1 - actor
                game.stacks[winner] += pot
                return game.stacks[random_pos] - initial[random_pos]
            
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
    
    return game.stacks[random_pos] - initial[random_pos]


async def test_vs_linear(num_hands=1000):
    print("Random vs Linear Player (simplified)")    
    random_player = RandomPlayer()
    linear = LinearPlayer()
    
    total = 0
    results = []
    
    print(f"Playing {num_hands} hands")
    
    for i in range(num_hands):
        pos = i % 2
        profit = await play_hand(random_player, linear, pos)
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
    

    print(f"Total hands:    {num_hands}")
    print(f"Total chips:    {total:+.1f}")
    print(f"EV/hand:        {ev:+.3f} bb/hand")
    print(f"Std dev:        {std:.3f}")
    print(f"Win/Loss/Tie:   {wins}/{losses}/{ties}")
    
    print(f"Ev vs linear: {ev:.2f} bb/hand")



def main():
    num_hands = 5000
    asyncio.run(test_vs_linear(num_hands))


if __name__ == "__main__":
    main()
