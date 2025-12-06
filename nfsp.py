#!/usr/bin/env python3
"""

NFSP for Double Board Pot Limit Omaha

Hi TAs! This is our implementation of NFSP for Double Board Pot Limit Omaha.

It uses a simplified action space and reward function for faster learning.

For the game, it follows a mandatory 2 bb initial bet, then pot limit betting rounds
up to the turn and no betting on the river.

To train the model, we do pretraining with pure RL (DQN) against a rule-based linear player.
Following that, we do self-play training with NFSP mixed mode for a fewer number of iterations.

Action Space:
  When NOT facing a bet options are - 0 = check, 1 = bet 1/2 pot, 2 = bet pot
  When FACING a bet options are - 0 = call, 1 = raise to the pot size, 3 = fold
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from game import PokerGame

def get_legal_actions(bet_to_call, hero_stack, villain_stack, pot):
    """
    Returns list of legal action indices
    """
    if bet_to_call > 0:
        # 0 = call, 1 = raise pot, 3 = fold - no jamming allowed
        legal = [0, 3]
        
        if hero_stack > bet_to_call:
            pot_raise = bet_to_call + pot
            if hero_stack >= pot_raise:
                legal.append(1)  # raise pot size
            
        return sorted(legal)
    else:
        # options when there is no bet from da villain - 0=check, 1=bet½, 2=bet pot
        legal = [0]
        if hero_stack > 0:
            legal.extend([1, 2])  # Only bet 1/2 pot and bet pot, no jam allowed
        return sorted(legal)


def legal_to_mask(legal_actions, num_actions=4):
    """legal actions become a binary mask tensor"""
    mask = np.zeros(num_actions, dtype=np.float32)
    for a in legal_actions:
        mask[a] = 1.0
    return mask


def apply_action(action, bet_to_call, hero_stack, villain_stack, pot):
    """
    Convert semantic action to either bet_amount or is_fold.
    """
    if bet_to_call > 0:
        if action == 0:  # call
            return min(bet_to_call, hero_stack), False
        elif action == 1:  # raise pot size
            return min(bet_to_call + pot, hero_stack), False
        elif action == 3:  # fold
            return 0, True
    else:
        if action == 0:  # check
            return 0, False
        elif action == 1:  # bet 1/2 pot size
            return min(max(1, pot // 2), hero_stack, villain_stack), False
        elif action == 2:  # bet pot
            return min(pot, hero_stack, villain_stack), False
    return 0, False


def action_name(action, facing_bet):
    #for logging
    if facing_bet:
        return {0: 'call', 1: 'raise', 3: 'fold'}[action]
    return {0: 'check', 1: 'bet½', 2: 'bet'}[action]

def encode_state(hand, boards, pot, stacks, position, street, bet_to_call):
    #Encode state - will be 185 features
    features = []
    
    # Hand cards - 52
    hand_vec = np.zeros(52)
    for card in hand:
        hand_vec[card - 1] = 1
    features.extend(hand_vec)
    
    # Board cards - 104
    board_vec = np.zeros(104)
    if len(boards) > 0 and boards[0]:
        for card in boards[0]:
            board_vec[card - 1] = 1
        if len(boards) > 1 and boards[1]:
            for card in boards[1]:
                board_vec[52 + card - 1] = 1
    features.extend(board_vec)
    
    # Hand strength - 20 - added for a better holistic evaluation of the hand
    game = PokerGame([100, 100], 1, 0)
    for board_idx in range(2):
        if len(boards) > board_idx and boards[board_idx] and len(boards[board_idx]) >= 3:
            rank, _ = game.omaha_hand_strength(hand, boards[board_idx])
            vec = np.zeros(10)
            vec[int(rank[0])] = 1.0
            features.extend(vec)
        else:
            features.extend(np.zeros(10)) #doing if no board cards yet
    
    # Game state - 5 features
    bb = 1.0
    features.append(pot / (100 * bb))
    features.append(stacks[0] / (100 * bb))
    features.append(stacks[1] / (100 * bb))
    features.append(bet_to_call / (100 * bb))
    features.append(position)
    
    # Street - 3 features
    street_map = {'FLOP': 0, 'TURN': 1, 'RIVER': 2}
    street_vec = np.zeros(3)
    if street in street_map:
        street_vec[street_map[street]] = 1
    features.extend(street_vec)
    
    # Facing bet flag - 1 feature
    features.append(1.0 if bet_to_call > 0 else 0.0)
    
    return np.array(features, dtype=np.float32)


STATE_DIM = 185


class QNetwork(nn.Module):
    def __init__(self, state_dim = STATE_DIM, num_actions = 4, hidden_dim = 512):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.2) #added dropout to prevent overfitting - improved model a lot
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_dim, num_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=STATE_DIM, num_actions=4, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_dim, num_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)


class RLReplayBuffer:
    """RL buffer stores the following state, action, reward, next_state, done, next_legal_mask"""
    def __init__(self, capacity=200000): #increased buffer (can reduce in the future if needed)
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done, next_legal_mask):
        self.buffer.append((state, action, reward, next_state, done, next_legal_mask))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones, next_legal_masks = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
            torch.FloatTensor(np.array(next_legal_masks))
        )
    
    def __len__(self):
        return len(self.buffer)


class SLReplayBuffer:
    def __init__(self, capacity=500000): #never hits this limit in the current num training iterations but in case its large
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action):
        self.buffer.append((state, action))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions)
        )
    
    def __len__(self):
        return len(self.buffer)


class NFSPAgent:
    def __init__(self, state_dim=STATE_DIM, num_actions=4, lr_rl=0.001, lr_sl=0.001,
                 gamma=0.99, eta=0.1, epsilon=0.1, target_update_freq=500,
                 reward_clip=5.0, device='cpu'): #dont change these values - tweaked to be good
        self.num_actions = num_actions
        self.gamma = gamma
        self.eta = eta
        self.epsilon = epsilon
        self.reward_clip = reward_clip
        self.device = device
        
        self.q_network = QNetwork(state_dim, num_actions).to(device)
        self.q_target = QNetwork(state_dim, num_actions).to(device)
        self.q_target.load_state_dict(self.q_network.state_dict())
        # Adam with L2 regularization
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr_rl, weight_decay=1e-5)
        
        self.policy_network = PolicyNetwork(state_dim, num_actions).to(device)
        # Adam with L2 regularization
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr_sl, weight_decay=1e-5)
        
        self.rl_buffer = RLReplayBuffer()
        self.sl_buffer = SLReplayBuffer()
        
        self.update_count = 0
        self.target_update_freq = target_update_freq
        
    def select_action(self, state, legal_actions, mode='mixed'):
        """Select action. mode='rl' for pure RL, 'sl' for pure SL, 'mixed' for NFSP"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if mode == 'rl':
            use_rl = True
        elif mode == 'sl':
            use_rl = False
        else:  # mixed
            use_rl = random.random() < self.eta
        
        with torch.no_grad():
            if use_rl:
                #epsilon greedy on Qvalues
                if random.random() < self.epsilon:
                    action = random.choice(legal_actions)
                else:
                    q_values = self.q_network(state_tensor)[0]
                    masked_q = q_values.clone()
                    for a in range(self.num_actions):
                        if a not in legal_actions:
                            masked_q[a] = -float('inf')
                    action = masked_q.argmax().item()
                return action, use_rl
            else:
                # sample from policy network
                logits = self.policy_network(state_tensor)[0]
                masked_logits = logits.clone()
                for a in range(self.num_actions):
                    if a not in legal_actions:
                        masked_logits[a] = -float('inf')
                probs = F.softmax(masked_logits, dim=0)
                action = torch.multinomial(probs, 1).item()
                return action, use_rl
    
    def train_rl(self, batch_size=128):
        """Train Q network with proper legal action masking on targets"""
        if len(self.rl_buffer) < batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones, next_legal_masks = self.rl_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        next_legal_masks = next_legal_masks.to(self.device)
        
        # cur q values
        q_values = self.q_network(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # target q values with legal action masking
        with torch.no_grad():
            q_next_all = self.q_target(next_states)
            # Mask illegal actions - set to -inf so they're never selected
            illegal_mask = (next_legal_masks == 0)
            q_next_all[illegal_mask] = -float('inf')
            q_next = q_next_all.max(1)[0]
            # handle case where all actions are illegal - terminal state
            q_next = torch.where(torch.isinf(q_next), torch.zeros_like(q_next), q_next)
            target = rewards + self.gamma * (1 - dones) * q_next
        
        loss = F.mse_loss(q_sa, target)
        
        self.q_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.q_optimizer.step()
        
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def train_sl(self, batch_size=128):
        """Train policy network - supervised learning on RL actions"""
        if len(self.sl_buffer) < batch_size:
            return 0.0
        
        states, actions = self.sl_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        
        logits = self.policy_network(states)

        loss = F.cross_entropy(logits, actions)
        
        self.policy_optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 10.0)

        self.policy_optimizer.step()
        
        return loss.item()


class LinearPlayer:
    """Rule-based - bet/raise with 2 pair+, otherwise check/fold - good player in practice"""
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
        
        has_strong = best_rank >= 2  # two pair or better - good hands
        
        if bet_to_call > 0:
            if has_strong:
                if 1 in legal_actions and random.random() < 0.4:
                    return 1  # raise pot size
                return 0  # call
            return 3  # fold
        else:
            if has_strong:
                return 2 if random.random() < 0.6 else 1  # bet pot or half
            return 0  # check


class NFSPTrainer:
    # Phase 1 - Pure DQN pretrain vs linear strat
    # Phase 2 - NFSP self-play - mixed mode, SL training
    def __init__(self, pretrain_iterations=100000, selfplay_iterations=20000):
        self.pretrain_iterations = pretrain_iterations
        self.selfplay_iterations = selfplay_iterations
        # Lower Learning rate for more stable learning, higher exploration, dropout + L2 reg
        self.agent_p1 = NFSPAgent(lr_rl=0.0003, lr_sl=0.0005, epsilon=0.1)
        self.agent_p2 = NFSPAgent(lr_rl=0.0003, lr_sl=0.0005)
        self.linear = LinearPlayer()
        
        self.rl_losses = []
        self.sl_losses = []
        self.rewards = []
    
    async def play_hand_vs_linear(self, agent, agent_pos, stacks, bb, dealer):
        """
        Play agent vs linear. 
        pure RL mode, simple reward - chip delta only, no pot-weighting
        """
        game = PokerGame(stacks[:], bb, dealer)
        
        cards = list(range(1, 53))
        random.shuffle(cards)
        hands = game.deal_hands(cards)
        boards = game.deal_flops(cards)
        
        pot = 2 * bb
        game.stacks[0] -= bb
        game.stacks[1] -= bb
        
        # trajectory - list of state, action, legal_mask
        trajectory = []
        
        # flop and turn
        for street_idx, street_name in enumerate(['FLOP', 'TURN']):
            if street_idx == 1:
                boards = game.deal_turns(cards, boards[0], boards[1])
            
            first = 1 - dealer
            second = dealer
            current_bet = 0
            
            for action_num in range(4):
                actor = first if action_num % 2 == 0 else second
                hero_stack = game.stacks[actor]
                villain_stack = game.stacks[1 - actor]
                
                if hero_stack <= 0:
                    break
                
                legal = get_legal_actions(current_bet, hero_stack, villain_stack, pot)
                legal_mask = legal_to_mask(legal)
                
                if actor == agent_pos:
                    state = encode_state(
                        hands[actor], boards, pot,
                        [hero_stack, villain_stack],
                        actor, street_name, current_bet
                    )
                    # pure RL during pretrain
                    action, _ = agent.select_action(state, legal, mode='rl')
                    trajectory.append((state, action, legal_mask))
                else:
                    action = self.linear.get_action(hands[actor], boards, legal, current_bet)
                
                bet_amount, is_fold = apply_action(action, current_bet, hero_stack, villain_stack, pot)
                
                if is_fold:
                    winner = 1 - actor
                    game.stacks[winner] += pot
                    # chip delta reward is the best out of what we tried
                    reward = (game.stacks[agent_pos] - stacks[agent_pos]) / bb
                    self._finalize_trajectory(agent, trajectory, reward, legal_mask)
                    return reward
                
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
        
        # showdown
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
    
        reward = (game.stacks[agent_pos] - stacks[agent_pos]) / bb
        
        # Terminal state mask - all zeros, no next actions
        terminal_mask = np.zeros(4, dtype=np.float32)
        self._finalize_trajectory(agent, trajectory, reward, terminal_mask)
        return reward
    
    def _finalize_trajectory(self, agent, trajectory, reward, final_mask):
        """Add trajectory to RL buffer only - no SL during pretrain."""
        reward_clipped = np.clip(reward, -agent.reward_clip, agent.reward_clip)
        
        for i, (state, action, legal_mask) in enumerate(trajectory):
            if i + 1 < len(trajectory):
                next_state = trajectory[i + 1][0]
                next_legal = trajectory[i + 1][2]
                done = 0.0
                r = 0.0
            else:
                next_state = state
                next_legal = final_mask
                done = 1.0
                r = reward_clipped
            
            agent.rl_buffer.add(state, action, r, next_state, done, next_legal)
    
    async def play_hand_selfplay(self, stacks, bb, dealer):
        """Self-play with NFSP - mixed mode, adds to SL buffer"""
        game = PokerGame(stacks[:], bb, dealer)
        
        cards = list(range(1, 53))
        random.shuffle(cards)
        hands = game.deal_hands(cards)
        boards = game.deal_flops(cards)
        
        pot = 2 * bb
        game.stacks[0] -= bb
        game.stacks[1] -= bb
        
        traj_p1, traj_p2 = [], []
        
        # Flop and turn
        for street_idx, street_name in enumerate(['FLOP', 'TURN']):
            if street_idx == 1:
                boards = game.deal_turns(cards, boards[0], boards[1])
            
            first = 1 - dealer
            second = dealer
            current_bet = 0
            
            for action_num in range(4):
                actor = first if action_num % 2 == 0 else second
                hero_stack = game.stacks[actor]
                villain_stack = game.stacks[1 - actor]
                
                if hero_stack <= 0:
                    break
                
                legal = get_legal_actions(current_bet, hero_stack, villain_stack, pot)
                legal_mask = legal_to_mask(legal)
                agent = self.agent_p1 if actor == 0 else self.agent_p2
                
                state = encode_state(
                    hands[actor], boards, pot,
                    [hero_stack, villain_stack],
                    actor, street_name, current_bet
                )
                # NFSP mixed mode during self-play
                action, used_rl = agent.select_action(state, legal, mode='mixed')
                
                if actor == 0:
                    traj_p1.append((state, action, legal_mask, used_rl))
                else:
                    traj_p2.append((state, action, legal_mask, used_rl))
                
                bet_amount, is_fold = apply_action(action, current_bet, hero_stack, villain_stack, pot)
                
                if is_fold:
                    winner = 1 - actor
                    game.stacks[winner] += pot
                    r1 = (game.stacks[0] - stacks[0]) / bb
                    r2 = (game.stacks[1] - stacks[1]) / bb
                    terminal_mask = np.zeros(4, dtype=np.float32)
                    self._finalize_selfplay(traj_p1, traj_p2, r1, r2, terminal_mask)
                    return r1, r2
                
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
        
        r1 = (game.stacks[0] - stacks[0]) / bb
        r2 = (game.stacks[1] - stacks[1]) / bb
        
        terminal_mask = np.zeros(4, dtype=np.float32)
        self._finalize_selfplay(traj_p1, traj_p2, r1, r2, terminal_mask)
        return r1, r2
    
    def _finalize_selfplay(self, traj_p1, traj_p2, r1, r2, terminal_mask):
        """Add trajectories to both RL and SL buffers"""
        r1_clip = np.clip(r1, -self.agent_p1.reward_clip, self.agent_p1.reward_clip)
        r2_clip = np.clip(r2, -self.agent_p2.reward_clip, self.agent_p2.reward_clip)
        
        for traj, agent, r_clip in [(traj_p1, self.agent_p1, r1_clip), (traj_p2, self.agent_p2, r2_clip)]:
            for i, (state, action, legal_mask, used_rl) in enumerate(traj):
                if i + 1 < len(traj):
                    next_state = traj[i + 1][0]
                    next_legal = traj[i + 1][2]
                    done = 0.0
                    r = 0.0
                else:
                    next_state = state
                    next_legal = terminal_mask
                    done = 1.0
                    r = r_clip
                
                agent.rl_buffer.add(state, action, r, next_state, done, next_legal)
                
                # NFSP, only add to SL buffer when using RL policy
                if used_rl:
                    agent.sl_buffer.add(state, action)
    
    async def pretrain_vs_linear(self):
        """Phase 1 -pure DQN best response to linear."""
        print("phase 1 - DQN vs Linear")
        print(f"Iterations: {self.pretrain_iterations:,}")
        
        rewards = []
        
        for i in range(1, self.pretrain_iterations + 1):
            stacks = [100, 100]
            dealer = random.choice([0, 1])
            agent_pos = i % 2
            
            r = await self.play_hand_vs_linear(self.agent_p1, agent_pos, stacks, 1, dealer)
            rewards.append(r)
            
            # Train Q-network every 4 hands
            if i % 4 == 0:
                loss_rl = self.agent_p1.train_rl(batch_size=128)
                self.rl_losses.append(loss_rl)
            
            # epsilon - 0.15 -> 0.05 over training
            if i % 10000 == 0:
                new_eps = max(0.05, self.agent_p1.epsilon - 0.01)
                self.agent_p1.epsilon = new_eps
            
            if i % 1000 == 0:
                avg_rl = np.mean(self.rl_losses[-250:]) if self.rl_losses else 0
                avg_r = np.mean(rewards[-500:])
                print(f"Pretraining iterations {i:>6,}/{self.pretrain_iterations:,} | "
                      f"RL_loss: {avg_rl:.4f} | Reward: {avg_r:+.2f} | "
                      f"RL_buf: {len(self.agent_p1.rl_buffer):,}")
        
        final_r = np.mean(rewards[-1000:]) if len(rewards) >= 1000 else np.mean(rewards)
        print(f"\npretraining done - avg reward = {final_r:+.3f} bb/hand\n")
        
        return final_r
    
    async def train_selfplay(self):
        """Phase 2: NFSP self-play."""
        print("Phase 2 - NFSP self-play mixed strat")
        print(f"Iterations: {self.selfplay_iterations:,}")
        
        # copy pretrained weights to P2
        self.agent_p2.q_network.load_state_dict(self.agent_p1.q_network.state_dict())
        self.agent_p2.q_target.load_state_dict(self.agent_p1.q_target.state_dict())
        self.agent_p2.policy_network.load_state_dict(self.agent_p1.policy_network.state_dict())
        
        # reset epsilon for self play
        self.agent_p1.epsilon = 0.1
        self.agent_p2.epsilon = 0.1
        
        rewards_p1 = []
        
        for i in range(1, self.selfplay_iterations + 1):
            stacks = [100, 100]
            dealer = random.choice([0, 1])
            
            r1, r2 = await self.play_hand_selfplay(stacks, 1, dealer)
            rewards_p1.append(r1)
            
            if i % 4 == 0:
                self.agent_p1.train_rl(batch_size=128)
                sl_loss_1 = self.agent_p1.train_sl(batch_size=128)
                self.agent_p2.train_rl(batch_size=128)
                sl_loss_2 = self.agent_p2.train_sl(batch_size=128)
                self.sl_losses.append((sl_loss_1 + sl_loss_2) / 2)
            
            if i % 1000 == 0:
                avg_sl = np.mean(self.sl_losses[-250:]) if self.sl_losses else 0
                avg_r1 = np.mean(rewards_p1[-500:])
                print(f"Selfplay iterations {i:>6,}/{self.selfplay_iterations:,} | "
                      f"SL_loss: {avg_sl:.4f} | P1_reward: {avg_r1:+.2f} | "
                      f"SL_buf: {len(self.agent_p1.sl_buffer):,}")
        
        print(f"\nPhase 2, self play complete\n")
    
    async def train(self):
        """Full training pipeline"""
        if self.pretrain_iterations > 0:
            await self.pretrain_vs_linear()
        
        if self.selfplay_iterations > 0:
            await self.train_selfplay()
        
        torch.save(self.agent_p1.q_network.state_dict(), 'q_network_p1.pth')
        torch.save(self.agent_p1.policy_network.state_dict(), 'policy_network_p1.pth')
        
        print("training complete")


def main():
    import sys
    import asyncio
    
    if len(sys.argv) >= 3:
        pretrain = int(sys.argv[1])
        selfplay = int(sys.argv[2])
    elif len(sys.argv) == 2:
        pretrain = int(sys.argv[1])
        selfplay = 20000
    else:
        pretrain = 200000 #also optimal
        selfplay = 20000 #optimal number of iterations for self play
    
    
    trainer = NFSPTrainer(pretrain_iterations=pretrain, selfplay_iterations=selfplay)
    asyncio.run(trainer.train())


if __name__ == "__main__":
    main()
