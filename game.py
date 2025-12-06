import random
import itertools
import asyncio

'''

Wierd rules:
 - You can only bet pot, check, call, or fold
 - There are only 2 betting rounds per hand (1st player either checks or bets pot, second player checks, calls, folds, or bets pot)
'''



class PokerGame:
    def __init__(self, stacks, big_blind, dealer):
        self.stacks = stacks
        self.big_blind = big_blind
        self.dealer = dealer

    

    def log_cards(self, message, cards):
        # Display cards with suits
        suit_emojis = ['♠️', '♥️', '♦️', '♣️']
        
        def card_str(card):
            rank = (card - 1) % 13 + 1
            suit = (card - 1) // 13
            rank_map = {1: 'A', 10: 'T', 11: 'J', 12: 'Q', 13: 'K'}
            rank_str = rank_map.get(rank, str(rank))
            return f"{rank_str}{suit_emojis[suit]}"
        
        card_strings = [card_str(c) for c in cards]
        print(f"{message} {' '.join(card_strings)}")
    
    async def runBettingRound(self, hand, boards, player_stack, opponent_stack, pot, dealer):
        # return: (player_stack, opponent_stack, pot, folded_player)
        #folded_player, None if no fold, 1 if player1 folded, 2 if player2 folded

        current_bet = 0  # Amount needed to call
        player_invested = 0  # Amount player has put in this round
        opponent_invested = 0  # Amount opponent has put in this round
        
        # Determine who acts first (dealer acts second)    
        if dealer == 1:
            # Player 2 is dealer, so Player 1 acts first
            first_player = 1
        else:
            # Player 1 is dealer, so Player 2 acts first
            first_player = 2
        
        action_on = first_player
        last_aggressor = None  # Track who made the last raise/bet
        
        while True:
            if action_on == 1:
                # Player 1's turn
                decision = await self.playerOneDecision(
                    hand, boards, player_stack, opponent_stack, 
                    pot, dealer, current_bet, player_invested
                )
                
                if decision == 0:  # Check
                    if current_bet > 0:
                        # Can't check if there's a bet to call
                        raise ValueError("Cannot check when there's a bet")
                    # Action moves to opponent
                    action_on = 2
                    
                elif isinstance(decision, (int, float)) and decision > 0:  # Bet/Raise
                    bet_amount = decision
                    
                    # Ensure bet doesn't exceed stack
                    bet_amount = min(bet_amount, player_stack)
                    
                    # If there's already a bet, this is a raise
                    if current_bet > 0:
                        # Raise must be at least the current bet
                        additional = bet_amount - (current_bet - player_invested)
                        if additional <= 0:
                            raise ValueError("Raise must be higher than current bet")
                    
                    player_stack -= bet_amount
                    player_invested += bet_amount
                    pot += bet_amount
                    current_bet = player_invested  # New amount to call
                    last_aggressor = 1
                    action_on = 2
                    
                elif decision == 2:  # Call
                    call_amount = current_bet - player_invested
                    call_amount = min(call_amount, player_stack)
                    
                    player_stack -= call_amount
                    player_invested += call_amount
                    pot += call_amount
                    
                    # If both players have invested equally, round is over
                    if player_invested == opponent_invested:
                        break
                        
                    action_on = 2
                    
                elif decision == 3:  # Fold
                    return player_stack, opponent_stack, pot, 1  # Player 1 folded
                    
            else:  # action_on == 2
                # Player 2's turn
                decision = await self.playerTwoDecision(
                    hand, boards, player_stack, opponent_stack,
                    pot, dealer, current_bet, opponent_invested, player_invested
                )
                
                if decision == 0:  # Check
                    if current_bet > 0:
                        raise ValueError("Cannot check when there's a bet")
                    # If both checked, round is over
                    if last_aggressor is None:
                        break
                    action_on = 1
                    
                elif isinstance(decision, (int, float)) and decision > 0:  # Bet/Raise
                    bet_amount = decision
                    bet_amount = min(bet_amount, opponent_stack)
                    
                    if current_bet > 0:
                        additional = bet_amount - (current_bet - opponent_invested)
                        if additional <= 0:
                            raise ValueError("Raise must be higher than current bet")
                    
                    opponent_stack -= bet_amount
                    opponent_invested += bet_amount
                    pot += bet_amount
                    current_bet = opponent_invested
                    last_aggressor = 2
                    action_on = 1
                    
                elif decision == 2:  # Call
                    call_amount = current_bet - opponent_invested
                    call_amount = min(call_amount, opponent_stack)
                    
                    opponent_stack -= call_amount
                    opponent_invested += call_amount
                    pot += call_amount
                    
                    if player_invested == opponent_invested:
                        break
                        
                    action_on = 1
                    
                elif decision == 3:  # Fold
                    return player_stack, opponent_stack, pot, 2  # Player 2 folded
            
            # Check if either player is all-in
            if player_stack == 0 or opponent_stack == 0:
                # Match any remaining bet if needed
                if player_invested < opponent_invested and player_stack == 0:
                    # Player 1 is all-in but has less invested
                    excess = opponent_invested - player_invested
                    pot -= excess
                    opponent_stack += excess
                    opponent_invested = player_invested
                elif opponent_invested < player_invested and opponent_stack == 0:
                    # Player 2 is all-in but has less invested
                    excess = player_invested - opponent_invested
                    pot -= excess
                    player_stack += excess
                    player_invested = opponent_invested
                break
        
        return player_stack, opponent_stack, pot, None  # No fold


    async def playerOneDecision(self, hand, boards, player_stack, opponent_stack, pot, dealer, current_bet, player_invested):
        # return: 0 (check), positive number (bet/raise amount), 2 (call), or 3 (fold)
        amount_to_call = current_bet - player_invested
        
        # Allowed decisions based on game state
        if amount_to_call == 0:
            # No bet to call, can check or bet
            allowed_decisions = [0, 1]  # check, bet pot
        else:
            # There's a bet to call, can call, fold, or raise
            allowed_decisions = [2, 3, 1]  # call, fold, raise
        
        decision = await self.makeDecision(hand, boards, player_stack, opponent_stack, dealer, allowed_decisions)
        
        # No bet to call
        if amount_to_call == 0:
            if decision == 0:  # Check
                return 0
            elif decision == 1:  # Bet pot
                bet_size = min(pot, player_stack, opponent_stack)
                return bet_size
        else:
            # Bet to call
            if decision == 2:  # Call
                return 2
            elif decision == 3:  # Fold
                return 3
            elif decision == 1:  # Raise pot
                raise_to = min(pot + amount_to_call, player_stack)
                return raise_to
        
        return 0  # Default to check


    async def playerTwoDecision(self, hand, boards, player_stack, opponent_stack, pot, dealer, current_bet, opponent_invested, player_invested):
        # return: 0 (check), positive number (bet/raise amount), 2 (call), or 3 (fold)
        amount_to_call = current_bet - opponent_invested
        
        # Determine allowed decisions based on game state
        if amount_to_call == 0:
            # No bet to call, can check or bet
            allowed_decisions = [0, 1]  # check, bet pot
        else:
            # Bet to call, can call, fold, or raise
            allowed_decisions = [2, 3, 1]  # call, fold, raise
        
        decision = await self.makeDecision(hand, boards, player_stack, opponent_stack, dealer, allowed_decisions, player_invested)
        
        # No bet to call
        if amount_to_call == 0:
            if decision == 0:  # Check
                return 0
            elif decision == 1:  # Bet pot
                bet_size = min(pot, opponent_stack, player_stack)
                return bet_size
        else:
            # Bet to call
            if decision == 2:  # Call
                return 2
            elif decision == 3:  # Fold
                return 3
            elif decision == 1:  # Raise pot
                raise_to = min(pot + amount_to_call, opponent_stack)
                return raise_to
        
        return 0
        
    async def bet_round(self):
        # Handle a betting round check/bet/call
        if self.stacks[0] == 0 or self.stacks[1] == 0:
            return 0
        
        print(f"\nPlayer 1 stack: {self.stacks[0]}, Player 2 stack: {self.stacks[1]}")
        bet_input = input("Player 1, or fold (f), or call (c), or bet pot (p): ")
        
        if not bet_input.strip():
            print("Player 1 checks")
            return 0
        
        try:
            bet_value = int(bet_input)
            bet_value = min(bet_value, self.stacks[0], self.stacks[1])
            
            if bet_value > 0:
                self.stacks[0] -= bet_value
                self.stacks[1] -= bet_value
                print(f"Player 1 bets {bet_value}")
                print(f"Player 2 calls {bet_value}")
                return bet_value * 2
        except ValueError:
            print("Invalid bet, checking instead")
            return 0
        
        return 0

    def pay_out(self, winnings_player1, winnings_player2, pot):
        # pay out winners
        self.stacks[0] += int(winnings_player1 * pot)
        self.stacks[1] += int(winnings_player2 * pot)
        return self.stacks

    def deal_hands(self, cards):
        # Deal 4 cards to each player
        return [cards[:4], cards[4:8]]

    def deal_flops(self, cards):
        # Deal two flops
        return [cards[8:11], cards[11:14]]

    def deal_turns(self, cards, board1, board2):
        # Deal turn cards to both boards
        return [board1 + [cards[14]], board2 + [cards[15]]]

    def deal_rivers(self, cards, board1, board2):
        # Deal river cards to both boards
        return [board1 + [cards[16]], board2 + [cards[17]]]

    def omaha_hand_strength(self, hand, board):
        # Calculate best Omaha hand (2 from hand, 3 from board)
        # Returns, (hand_rank, best_five_cards)
        def card_rank(card):
            return ((card - 1) % 13) + 2  # 2-14, where 14 = Ace

        def card_suit(card):
            return (card - 1) // 13  # 0-3

        def hand_rank(five):
            # eval hand
            counts = {}
            suits = {}
            ranks = []
            
            for c in five:
                r = card_rank(c)
                s = card_suit(c)
                ranks.append(r)
                counts[r] = counts.get(r, 0) + 1
                suits[s] = suits.get(s, [])
                suits[s].append(r)
            
            ranks.sort(reverse=True)
            
            # Check for flush
            flush = None
            for suit, rlist in suits.items():
                if len(rlist) >= 5:
                    flush = sorted(rlist, reverse=True)
                    break
            
            straight = find_straight(ranks)
            
            # Straight flush checks
            if flush:
                straight_flush = find_straight(flush)
                if straight_flush:
                    if max(straight_flush) == 14:
                        return (9,) + tuple(straight_flush)  # Royal Flush
                    return (8,) + tuple(straight_flush)
            
            # Four of a kind
            if 4 in counts.values():
                quad = [r for r in counts if counts[r] == 4][0]
                kicker = max([r for r in counts if r != quad])
                return (7, quad, kicker)
            
            # Full house
            if sorted(counts.values(), reverse=True) == [3, 2]:
                trips = [r for r in counts if counts[r] == 3][0]
                pair = [r for r in counts if counts[r] == 2][0]
                return (6, trips, pair)
            
            # Flush
            if flush:
                return (5,) + tuple(flush[:5])
            
            # Straight
            if straight:
                return (4,) + tuple(straight)
            
            # Three of a kind
            if 3 in counts.values():
                trips = [r for r in counts if counts[r] == 3][0]
                kick = sorted([r for r in counts if r != trips], reverse=True)
                return (3, trips) + tuple(kick[:2])
            
            # Two pair
            pairs = [r for r in counts if counts[r] == 2]
            if len(pairs) >= 2:
                high, low = sorted(pairs, reverse=True)[:2]
                kicker = max([r for r in counts if r != high and r != low])
                return (2, high, low, kicker)
            
            # One pair
            if len(pairs) == 1:
                pair = pairs[0]
                kick = sorted([r for r in counts if r != pair], reverse=True)
                return (1, pair) + tuple(kick[:3])
            
            # High card
            return (0,) + tuple(ranks[:5])

        def find_straight(sorted_ranks):
            # Find the best straight
            uniq = sorted(set(sorted_ranks), reverse=True)
            for i in range(len(uniq) - 4):
                seq = uniq[i:i+5]
                if seq[0] - seq[4] == 4:
                    return seq
            # Wheel (A-2-3-4-5)
            if set([14, 5, 4, 3, 2]).issubset(uniq):
                return [5, 4, 3, 2, 14]
            return None

        best = None
        best_five = None
        for hand2 in itertools.combinations(hand, 2):
            for board3 in itertools.combinations(board, 3):
                five = list(hand2) + list(board3)
                rank = hand_rank(five)
                if best is None or rank > best:
                    best = rank
                    best_five = five
        return best, best_five

    def showdown(self, hands, boards, log_cards):
        # Determine winners on both boards
        # Player 1s best hands on both boards
        best_hand_b1_p1, cards_b1_p1 = self.omaha_hand_strength(hands[0], boards[0])
        best_hand_b2_p1, cards_b2_p1 = self.omaha_hand_strength(hands[0], boards[1])
        
        # Player 2s best hands on both boards
        best_hand_b1_p2, cards_b1_p2 = self.omaha_hand_strength(hands[1], boards[0])
        best_hand_b2_p2, cards_b2_p2 = self.omaha_hand_strength(hands[1], boards[1])
        
        winnings_player1 = 0
        winnings_player2 = 0
        print("Player 1's hand: ")
        log_cards("", hands[0])
        print("Player 2's hand: ")
        log_cards("", hands[1])
        print("Board 1: ")
        log_cards("", boards[0])
        print("Board 2: ")
        log_cards("", boards[1])
        
        # Board 1
        if best_hand_b1_p1 > best_hand_b1_p2:
            winnings_player1 += 0.5
            print("Player 1 wins Board 1 with hand: ")
            log_cards("", cards_b1_p1)
            print(" against ")
            log_cards("", cards_b1_p2)
        elif best_hand_b1_p1 < best_hand_b1_p2:
            winnings_player2 += 0.5
            print("Player 2 wins Board 1 with hand: ")
            log_cards("", cards_b1_p2)
            print(" against ")
            log_cards("", cards_b1_p1)
        else:
            winnings_player1 += 0.25
            winnings_player2 += 0.25
            print("Board 1 is a tie")
        
        # Board 2
        if best_hand_b2_p1 > best_hand_b2_p2:
            winnings_player1 += 0.5
            print("Player 1 wins Board 2 with hand: ")
            log_cards("", cards_b2_p1)
            print(" against ")
            log_cards("", cards_b2_p2)
        elif best_hand_b2_p1 < best_hand_b2_p2:
            winnings_player2 += 0.5
            print("Player 2 wins Board 2 with hand: ")
            log_cards("", cards_b2_p2)
            print(" against ")
            log_cards("", cards_b2_p1)
        else:
            winnings_player1 += 0.25
            winnings_player2 += 0.25
            print("Board 2 is a tie with hands: ")
            log_cards("", cards_b2_p1)
            print(" and ")
            log_cards("", cards_b2_p2)
        
        return winnings_player1, winnings_player2