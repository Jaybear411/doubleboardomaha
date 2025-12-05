#!/usr/bin/env python3
"""
NFSP for Double Board Omaha with RICH ACTION SPACE

Key fixes:
1. Pretraining = pure RL (DQN), NOT NFSP mixed mode
2. Q-targets mask illegal actions (no fantasy backups)
3. Simple rewards during pretrain (no pot-weighting)
4. SL training only happens in self-play phase

Action Space (context-dependent):
  When NOT facing a bet: 0=check, 1=bet½, 2=bet, 3=jam
  When FACING a bet: 0=call, 1=raise, 2=jam, 3=fold
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from game import PokerGame

# =============================================================================
# ACTION HANDLING
# =============================================================================

def get_legal_actions(bet_to_call, hero_stack, villain_stack, pot):
    """Returns list of legal action indices."""
    if bet_to_call > 0:
        # Facing bet: 0=call, 1=raise, 2=jam, 3=fold
        legal = [0, 3]  # Always can call or fold
        
        min_raise = bet_to_call * 2
        if hero_stack > bet_to_call and hero_stack >= min_raise:
            legal.append(1)  # raise
            legal.append(2)  # jam
        elif hero_stack > bet_to_call:
            legal.append(2)  # jam only
            
        return sorted(legal)
    else:
        # No bet: 0=check, 1=bet½, 2=bet, 3=jam
        legal = [0]
        if hero_stack > 0:
            legal.extend([1, 2, 3])
        return sorted(legal)


def legal_to_mask(legal_actions, num_actions=4):
    """Convert legal action list to binary mask tensor."""
    mask = np.zeros(num_actions, dtype=np.float32)
    for a in legal_actions:
        mask[a] = 1.0
    return mask


def apply_action(action, bet_to_call, hero_stack, villain_stack, pot):
    """Convert semantic action to (bet_amount, is_fold)."""
    if bet_to_call > 0:
        if action == 0:  # call
            return min(bet_to_call, hero_stack), False
        elif action == 1:  # raise pot
            return min(bet_to_call + pot, hero_stack), False
        elif action == 2:  # jam
            return hero_stack, False
        elif action == 3:  # fold
            return 0, True
    else:
        if action == 0:  # check
            return 0, False
        elif action == 1:  # bet ½ pot
            return min(max(1, pot // 2), hero_stack, villain_stack), False
        elif action == 2:  # bet pot
            return min(pot, hero_stack, villain_stack), False
        elif action == 3:  # jam
            return min(hero_stack, villain_stack), False
    return 0, False


def action_name(action, facing_bet):
    """Human-readable action name."""
    if facing_bet:
        return {0: 'call', 1: 'raise', 2: 'jam', 3: 'fold'}[action]
    return {0: 'check', 1: 'bet½', 2: 'bet', 3: 'jam'}[action]


# =============================================================================
# STATE ENCODING
# =============================================================================

def encode_state(hand, boards, pot, stacks, position, street, bet_to_call):
    """Encode state (185 features)."""
    features = []
    
    # Hand cards (52)
    hand_vec = np.zeros(52)
    for card in hand:
        hand_vec[card - 1] = 1
    features.extend(hand_vec)
    
    # Board cards (104)
    board_vec = np.zeros(104)
    if len(boards) > 0 and boards[0]:
        for card in boards[0]:
            board_vec[card - 1] = 1
        if len(boards) > 1 and boards[1]:
            for card in boards[1]:
                board_vec[52 + card - 1] = 1
    features.extend(board_vec)
    
    # Hand strength (20)
    game = PokerGame([100, 100], 1, 0)
    for board_idx in range(2):
        if len(boards) > board_idx and boards[board_idx] and len(boards[board_idx]) >= 3:
            rank, _ = game.omaha_hand_strength(hand, boards[board_idx])
            vec = np.zeros(10)
            vec[int(rank[0])] = 1.0
            features.extend(vec)
        else:
            features.extend(np.zeros(10))
    
    # Game state (5)
    bb = 1.0
    features.append(pot / (100 * bb))
    features.append(stacks[0] / (100 * bb))
    features.append(stacks[1] / (100 * bb))
    features.append(bet_to_call / (100 * bb))
    features.append(position)
    
    # Street (3)
    street_map = {'FLOP': 0, 'TURN': 1, 'RIVER': 2}
    street_vec = np.zeros(3)
    if street in street_map:
        street_vec[street_map[street]] = 1
    features.extend(street_vec)
    
    # Facing bet flag (1)
    features.append(1.0 if bet_to_call > 0 else 0.0)
    
    return np.array(features, dtype=np.float32)


STATE_DIM = 185


# =============================================================================
# NETWORKS
# =============================================================================

class QNetwork(nn.Module):
    def __init__(self, state_dim=STATE_DIM, num_actions=4, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=STATE_DIM, num_actions=4, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# =============================================================================
# REPLAY BUFFERS
# =============================================================================

class RLReplayBuffer:
    """RL buffer stores: (state, action, reward, next_state, done, next_legal_mask)"""
    def __init__(self, capacity=200000):
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
    def __init__(self, capacity=500000):
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


# =============================================================================
# NFSP AGENT
# =============================================================================

class NFSPAgent:
    def __init__(self, state_dim=STATE_DIM, num_actions=4, lr_rl=0.001, lr_sl=0.001,
                 gamma=0.99, eta=0.1, epsilon=0.1, target_update_freq=500,
                 reward_clip=5.0, device='cpu'):
        self.num_actions = num_actions
        self.gamma = gamma
        self.eta = eta
        self.epsilon = epsilon
        self.reward_clip = reward_clip
        self.device = device
        
        self.q_network = QNetwork(state_dim, num_actions).to(device)
        self.q_target = QNetwork(state_dim, num_actions).to(device)
        self.q_target.load_state_dict(self.q_network.state_dict())
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr_rl)
        
        self.policy_network = PolicyNetwork(state_dim, num_actions).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr_sl)
        
        self.rl_buffer = RLReplayBuffer()
        self.sl_buffer = SLReplayBuffer()
        
        self.update_count = 0
        self.target_update_freq = target_update_freq
        
    def select_action(self, state, legal_actions, mode='mixed'):
        """Select action. mode='rl' for pure RL, 'sl' for pure SL, 'mixed' for NFSP."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if mode == 'rl':
            use_rl = True
        elif mode == 'sl':
            use_rl = False
        else:  # mixed
            use_rl = random.random() < self.eta
        
        with torch.no_grad():
            if use_rl:
                # Epsilon-greedy on Q-values
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
                # Sample from policy network
                logits = self.policy_network(state_tensor)[0]
                masked_logits = logits.clone()
                for a in range(self.num_actions):
                    if a not in legal_actions:
                        masked_logits[a] = -float('inf')
                probs = F.softmax(masked_logits, dim=0)
                action = torch.multinomial(probs, 1).item()
                return action, use_rl
    
    def train_rl(self, batch_size=128):
        """Train Q-network with PROPER legal action masking on targets."""
        if len(self.rl_buffer) < batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones, next_legal_masks = self.rl_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        next_legal_masks = next_legal_masks.to(self.device)
        
        # Current Q-values
        q_values = self.q_network(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values WITH LEGAL ACTION MASKING
        with torch.no_grad():
            q_next_all = self.q_target(next_states)
            # Mask illegal actions: set to -inf so they're never selected
            illegal_mask = (next_legal_masks == 0)
            q_next_all[illegal_mask] = -float('inf')
            q_next = q_next_all.max(1)[0]
            # Handle case where all actions are illegal (terminal state)
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
        """Train policy network (supervised learning on RL actions)."""
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


# =============================================================================
# LINEAR PLAYER
# =============================================================================

class LinearPlayer:
    """Rule-based: bet/raise with 2 pair+, otherwise check/fold."""
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
        
        has_strong = best_rank >= 2
        has_monster = best_rank >= 5
        
        if bet_to_call > 0:
            if has_monster and 2 in legal_actions:
                return 2  # jam
            elif has_strong:
                if 1 in legal_actions and random.random() < 0.3:
                    return 1  # raise
                return 0  # call
            return 3  # fold
        else:
            if has_monster and 3 in legal_actions:
                return 3  # jam
            elif has_strong:
                return 2 if random.random() < 0.5 else 1  # bet pot or half
            return 0  # check


# =============================================================================
# TRAINER
# =============================================================================

class NFSPTrainer:
    """
    Phase 1: Pure DQN vs Linear (no NFSP mixing, no pot-weighting)
    Phase 2: NFSP self-play (mixed mode, SL training)
    """
    def __init__(self, pretrain_iterations=100000, selfplay_iterations=20000):
        self.pretrain_iterations = pretrain_iterations
        self.selfplay_iterations = selfplay_iterations
        self.agent_p1 = NFSPAgent(epsilon=0.15)  # Higher exploration for pretrain
        self.agent_p2 = NFSPAgent()
        self.linear = LinearPlayer()
        
        self.rl_losses = []
        self.sl_losses = []
        self.rewards = []
    
    async def play_hand_vs_linear(self, agent, agent_pos, stacks, bb, dealer):
        """
        Play agent vs linear. 
        PURE RL mode, SIMPLE reward (chip delta only, no pot-weighting).
        """
        game = PokerGame(stacks[:], bb, dealer)
        
        cards = list(range(1, 53))
        random.shuffle(cards)
        hands = game.deal_hands(cards)
        boards = game.deal_flops(cards)
        
        pot = 2 * bb
        game.stacks[0] -= bb
        game.stacks[1] -= bb
        
        # trajectory: list of (state, action, legal_mask)
        trajectory = []
        
        for street_idx, street_name in enumerate(['FLOP', 'TURN', 'RIVER']):
            if street_idx == 1:
                boards = game.deal_turns(cards, boards[0], boards[1])
            elif street_idx == 2:
                boards = game.deal_rivers(cards, boards[0], boards[1])
            
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
                    # PURE RL during pretrain
                    action, _ = agent.select_action(state, legal, mode='rl')
                    trajectory.append((state, action, legal_mask))
                else:
                    action = self.linear.get_action(hands[actor], boards, legal, current_bet)
                
                bet_amount, is_fold = apply_action(action, current_bet, hero_stack, villain_stack, pot)
                
                if is_fold:
                    winner = 1 - actor
                    game.stacks[winner] += pot
                    # SIMPLE REWARD: just chip delta
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
        
        # SIMPLE REWARD
        reward = (game.stacks[agent_pos] - stacks[agent_pos]) / bb
        
        # Terminal state mask (all zeros - no next actions)
        terminal_mask = np.zeros(4, dtype=np.float32)
        self._finalize_trajectory(agent, trajectory, reward, terminal_mask)
        return reward
    
    def _finalize_trajectory(self, agent, trajectory, reward, final_mask):
        """Add trajectory to RL buffer only (no SL during pretrain)."""
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
        """Self-play with NFSP (mixed mode, adds to SL buffer)."""
        game = PokerGame(stacks[:], bb, dealer)
        
        cards = list(range(1, 53))
        random.shuffle(cards)
        hands = game.deal_hands(cards)
        boards = game.deal_flops(cards)
        
        pot = 2 * bb
        game.stacks[0] -= bb
        game.stacks[1] -= bb
        
        traj_p1, traj_p2 = [], []
        
        for street_idx, street_name in enumerate(['FLOP', 'TURN', 'RIVER']):
            if street_idx == 1:
                boards = game.deal_turns(cards, boards[0], boards[1])
            elif street_idx == 2:
                boards = game.deal_rivers(cards, boards[0], boards[1])
            
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
        """Add trajectories to both RL and SL buffers."""
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
                
                # NFSP: only add to SL buffer when using RL policy
                if used_rl:
                    agent.sl_buffer.add(state, action)
    
    async def pretrain_vs_linear(self):
        """Phase 1: Pure DQN best response to linear."""
        print("=" * 70)
        print("PHASE 1: PURE RL (DQN) VS LINEAR")
        print("=" * 70)
        print(f"Iterations: {self.pretrain_iterations:,}")
        print("Mode: Pure RL (no NFSP mixing)")
        print("Reward: Simple chip delta (no pot-weighting)")
        print("Training: Q-network only (no SL)\n")
        
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
            
            # Anneal epsilon: 0.15 -> 0.05 over training
            if i % 10000 == 0:
                new_eps = max(0.05, self.agent_p1.epsilon - 0.01)
                self.agent_p1.epsilon = new_eps
            
            if i % 1000 == 0:
                avg_rl = np.mean(self.rl_losses[-250:]) if self.rl_losses else 0
                avg_r = np.mean(rewards[-500:])
                print(f"[Pretrain] {i:>6,}/{self.pretrain_iterations:,} | "
                      f"RL_loss: {avg_rl:.4f} | Reward: {avg_r:+.2f} | "
                      f"eps: {self.agent_p1.epsilon:.2f} | "
                      f"RL_buf: {len(self.agent_p1.rl_buffer):,}")
        
        final_r = np.mean(rewards[-1000:]) if len(rewards) >= 1000 else np.mean(rewards)
        print(f"\nPRETRAIN COMPLETE: Avg reward = {final_r:+.3f} bb/hand\n")
        
        return final_r
    
    async def train_selfplay(self):
        """Phase 2: NFSP self-play."""
        print("=" * 70)
        print("PHASE 2: NFSP SELF-PLAY")
        print("=" * 70)
        print(f"Iterations: {self.selfplay_iterations:,}")
        print("Mode: Mixed (NFSP)")
        print("Training: Q-network + Policy network\n")
        
        # Copy pretrained weights to P2
        self.agent_p2.q_network.load_state_dict(self.agent_p1.q_network.state_dict())
        self.agent_p2.q_target.load_state_dict(self.agent_p1.q_target.state_dict())
        self.agent_p2.policy_network.load_state_dict(self.agent_p1.policy_network.state_dict())
        
        # Reset epsilon for self-play
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
                print(f"[Selfplay] {i:>6,}/{self.selfplay_iterations:,} | "
                      f"SL_loss: {avg_sl:.4f} | P1_reward: {avg_r1:+.2f} | "
                      f"SL_buf: {len(self.agent_p1.sl_buffer):,}")
        
        print(f"\nSELF-PLAY COMPLETE\n")
    
    async def train(self):
        """Full training pipeline."""
        if self.pretrain_iterations > 0:
            await self.pretrain_vs_linear()
        
        if self.selfplay_iterations > 0:
            await self.train_selfplay()
        
        torch.save(self.agent_p1.q_network.state_dict(), 'q_network_p1.pth')
        torch.save(self.agent_p1.policy_network.state_dict(), 'policy_network_p1.pth')
        
        print("=" * 70)
        print("TRAINING COMPLETE")
        print("Saved: q_network_p1.pth, policy_network_p1.pth")
        print("=" * 70)


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
        pretrain = 100000
        selfplay = 20000
    
    print("\n" + "=" * 70)
    print("NFSP TRAINING - FIXED VERSION")
    print("=" * 70)
    print(f"  Phase 1 (Pure RL vs Linear): {pretrain:,} iterations")
    print(f"  Phase 2 (NFSP Self-play): {selfplay:,} iterations")
    print(f"\n  Key fixes:")
    print(f"    ✓ Pretrain = pure RL (no NFSP mixing)")
    print(f"    ✓ Q-targets mask illegal actions")
    print(f"    ✓ Simple rewards during pretrain")
    print(f"    ✓ SL training only in self-play")
    print("=" * 70 + "\n")
    
    trainer = NFSPTrainer(pretrain_iterations=pretrain, selfplay_iterations=selfplay)
    asyncio.run(trainer.train())


if __name__ == "__main__":
    main()
