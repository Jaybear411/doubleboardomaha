# Neural Fictitious Self-Play (NFSP) with Pretraining
# Phase 1: Pretrain against LinearPlayer (rule-based opponent)
# Phase 2: Self-play NFSP for continued improvement

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from game import PokerGame

# State representation
def encode_state(hand, boards, pot, stacks, position, street, bet_to_call):
    """Encode double board poker state into a feature vector"""
    features = []
    
    # Hand cards (4 cards), 52 features - one-hot encoding
    hand_vec = np.zeros(52)
    for card in hand:
        hand_vec[card - 1] = 1
    features.extend(hand_vec)
    
    # Board cards, 52 features per board, 2 boards = 104 features
    board_vec = np.zeros(52 * 2)
    if len(boards) > 0 and boards[0] is not None:
        for card in boards[0]:
            board_vec[card - 1] = 1
        if len(boards) > 1 and boards[1] is not None:
            for card in boards[1]:
                board_vec[52 + card - 1] = 1
    features.extend(board_vec)
    
    # Normalized pot and stack info, 5 features
    bb = 1.0  # big blind for normalization
    features.append(pot / (100 * bb))  # normalized pot
    features.append(stacks[0] / (100 * bb))  # normalized player stack
    features.append(stacks[1] / (100 * bb))  # normalized opponent stack
    features.append(bet_to_call / (100 * bb))  # normalized bet to call
    features.append(position)  # 0 or 1
    
    # Street encoding, 3 features (one-hot)
    street_map = {'FLOP': 0, 'TURN': 1, 'RIVER': 2}
    street_vec = np.zeros(3)
    if street in street_map:
        street_vec[street_map[street]] = 1
    features.extend(street_vec)
    
    return np.array(features, dtype=np.float32)


# Q Network for RL
class QNetwork(nn.Module):
    """Q-Network for estimating expected future rewards"""
    def __init__(self, state_dim=164, num_actions=4, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PolicyNetwork(nn.Module):
    """Policy Network for learning average strategy"""
    def __init__(self, state_dim=164, num_actions=4, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Return logits


# Replay Buffers
class RLReplayBuffer:
    """Store (s, a, r, s', done) transitions for RL training"""
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class SLReplayBuffer:
    """Store (s, a) pairs for supervised learning of average strategy"""
    def __init__(self, capacity=200000):
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


# NFSP Agent
class NFSPAgent:
    """Neural Fictitious Self-Play agent combining RL and SL"""
    def __init__(
        self,
        state_dim=164,
        num_actions=4,
        lr_rl=0.0001,  # Reduced from 0.0005 for stability
        lr_sl=0.0005,
        gamma=0.95,  # Reduced from 0.99 to shorten horizon
        eta=0.1,  # probability of using RL policy during training
        epsilon=0.1,  # exploration rate for RL policy
        target_update_freq=1000,
        reward_clip=5.0,  # Clip rewards to prevent large gradients
        device='cpu'
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.eta = eta
        self.epsilon = epsilon
        self.reward_clip = reward_clip
        self.device = device
        
        # RL networks (Q-learning)
        self.q_network = QNetwork(state_dim, num_actions).to(device)
        self.q_target = QNetwork(state_dim, num_actions).to(device)
        self.q_target.load_state_dict(self.q_network.state_dict())
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr_rl)
        
        # SL network (average strategy)
        self.policy_network = PolicyNetwork(state_dim, num_actions).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr_sl)
        
        # Replay buffers
        self.rl_buffer = RLReplayBuffer()
        self.sl_buffer = SLReplayBuffer()
        
        # Training stats
        self.update_count = 0
        self.target_update_freq = target_update_freq
        
    def select_action(self, state, legal_actions, mode='mixed'):
        """Select action using mixed strategy or specific policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Decide which policy to use
        if mode == 'mixed':
            use_rl = random.random() < self.eta
        elif mode == 'rl':
            use_rl = True
        elif mode == 'sl' or mode == 'average':
            use_rl = False
        else:
            use_rl = False  # Default to average strategy
        
        with torch.no_grad():
            if use_rl:
                # RL policy: epsilon-greedy on Q-values
                if random.random() < self.epsilon:
                    action = random.choice(legal_actions)
                else:
                    q_values = self.q_network(state_tensor)[0]
                    # Mask illegal actions
                    masked_q = q_values.clone()
                    for a in range(self.num_actions):
                        if a not in legal_actions:
                            masked_q[a] = -float('inf')
                    action = masked_q.argmax().item()
                return action, use_rl
            else:
                # SL policy: sample from learned average strategy
                logits = self.policy_network(state_tensor)[0]
                # Mask illegal actions
                masked_logits = logits.clone()
                for a in range(self.num_actions):
                    if a not in legal_actions:
                        masked_logits[a] = -float('inf')
                probs = F.softmax(masked_logits, dim=0)
                action = torch.multinomial(probs, 1).item()
                return action, use_rl
    
    def train_rl(self, batch_size=128):
        """Train RL network using Q-learning"""
        if len(self.rl_buffer) < batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.rl_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        q_values = self.q_network(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values using target network
        with torch.no_grad():
            q_next = self.q_target(next_states).max(1)[0]
            target = rewards + self.gamma * (1 - dones) * q_next
        
        # MSE loss
        loss = F.mse_loss(q_sa, target)
        
        self.q_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.q_optimizer.step()
        
        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def train_sl(self, batch_size=128):
        """Train SL network using cross-entropy loss"""
        if len(self.sl_buffer) < batch_size:
            return 0.0
        
        states, actions = self.sl_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        
        # Predict action probabilities
        logits = self.policy_network(states)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, actions)
        
        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 10.0)
        self.policy_optimizer.step()
        
        return loss.item()


# LinearPlayer for pretraining
class LinearPlayer:
    """Rule-based player: bet pot with 2 pair+, otherwise check/fold"""
    
    def __init__(self):
        self.game = PokerGame([100, 100], 1, 0)  # For hand evaluation
    
    def get_action(self, hand, boards, legal_actions):
        """
        Evaluate hand strength and decide action:
        - 2 pair or better (rank >= 2): bet pot (action 1) or call (action 2)
        - Worse than 2 pair: check (action 0) or fold (action 3)
        """
        # Evaluate hand on both boards
        best_rank = 0
        
        if boards and len(boards) > 0:
            # Check board 1
            if boards[0] and len(boards[0]) >= 3:
                rank1, _ = self.game.omaha_hand_strength(hand, boards[0])
                best_rank = max(best_rank, rank1[0])
            
            # Check board 2 if exists
            if len(boards) > 1 and boards[1] and len(boards[1]) >= 3:
                rank2, _ = self.game.omaha_hand_strength(hand, boards[1])
                best_rank = max(best_rank, rank2[0])
        
        # Hand rank: 0=high card, 1=pair, 2=two pair, 3=trips, 4=straight, etc.
        has_strong_hand = best_rank >= 2  # 2 pair or better
        
        # Decision based on hand strength and legal actions
        if has_strong_hand:
            # Strong hand: bet or call
            if 1 in legal_actions:  # Can bet pot
                return 1
            elif 2 in legal_actions:  # Can call
                return 2
            elif 0 in legal_actions:  # Can check
                return 0
            else:  # Must fold
                return 3
        else:
            # Weak hand: check or fold
            if 0 in legal_actions:  # Can check
                return 0
            else:  # Must fold
                return 3


# NFSP Training Loop with pretraining and self-play
class NFSPTrainer:
    """Trainer for NFSP agents with pretraining against LinearPlayer, then mixed training"""
    def __init__(self, pretrain_iterations=5000, selfplay_iterations=10000, linear_mix_rate=0.3):
        self.pretrain_iterations = pretrain_iterations
        self.selfplay_iterations = selfplay_iterations
        self.linear_mix_rate = linear_mix_rate  # % of games vs Linear during selfplay
        self.agent_p1 = NFSPAgent()
        self.agent_p2 = NFSPAgent()
        self.linear_player = LinearPlayer()
        
        # Stats
        self.rl_losses = []
        self.sl_losses = []
        self.rewards_p1 = []
        self.rewards_p2 = []
        self.pretrain_rewards = []
        self.vs_linear_rewards = []  # Track performance vs Linear throughout
    
    async def play_hand_vs_linear(self, agent, agent_position, stacks, big_blind, dealer):
        """Play one hand with NFSP agent vs LinearPlayer"""
        game = PokerGame(stacks[:], big_blind, dealer)
        
        # Deal cards
        cards = list(range(1, 53))
        random.shuffle(cards)
        hands = game.deal_hands(cards)
        boards = game.deal_flops(cards)
        
        # Initial pot (blinds already posted)
        pot = 2 * big_blind
        game.stacks[0] -= big_blind
        game.stacks[1] -= big_blind
        
        # Track trajectory for agent only
        trajectory = []  # [(state, action, reward, next_state, done, used_rl)]
        
        # Simplified betting: FLOP and TURN only (2 rounds)
        for street_idx, street_name in enumerate(['FLOP', 'TURN']):
            if street_idx == 1:
                boards = game.deal_turns(cards, boards[0], boards[1])
            
            first_player = 1 - dealer
            second_player = dealer
            
            # First player action
            if first_player == agent_position:
                state1 = encode_state(
                    hands[first_player], boards, pot,
                    [game.stacks[first_player], game.stacks[second_player]],
                    first_player, street_name, 0
                )
                action1, used_rl1 = agent.select_action(state1, [0, 1], mode='mixed')
                trajectory.append((state1, action1, 0, None, False, used_rl1))
            else:
                action1 = self.linear_player.get_action(hands[first_player], boards, [0, 1])
            
            bet_to_call = 0
            if action1 == 1:  # bet pot
                bet_amount = min(pot, game.stacks[first_player], game.stacks[second_player])
                game.stacks[first_player] -= bet_amount
                pot += bet_amount
                bet_to_call = bet_amount
            
            # Second player action
            legal2 = [2, 3] if bet_to_call > 0 else [0, 1]
            if second_player == agent_position:
                state2 = encode_state(
                    hands[second_player], boards, pot,
                    [game.stacks[second_player], game.stacks[first_player]],
                    second_player, street_name, bet_to_call
                )
                action2, used_rl2 = agent.select_action(state2, legal2, mode='mixed')
                trajectory.append((state2, action2, 0, None, False, used_rl2))
            else:
                action2 = self.linear_player.get_action(hands[second_player], boards, legal2)
            
            # Process second player's action
            if bet_to_call > 0:
                if action2 == 3:  # fold
                    game.stacks[first_player] += pot
                    reward = (game.stacks[agent_position] - stacks[agent_position]) / big_blind
                    self.finalize_agent_trajectory(agent, trajectory, reward)
                    return reward
                elif action2 == 2:  # call
                    call_amount = min(bet_to_call, game.stacks[second_player])
                    game.stacks[second_player] -= call_amount
                    pot += call_amount
            else:
                if action2 == 1:  # bet pot
                    bet_amount = min(pot, game.stacks[second_player], game.stacks[first_player])
                    game.stacks[second_player] -= bet_amount
                    pot += bet_amount
                    
                    # First player must respond
                    if first_player == agent_position:
                        state3 = encode_state(
                            hands[first_player], boards, pot,
                            [game.stacks[first_player], game.stacks[second_player]],
                            first_player, street_name, bet_amount
                        )
                        action3, used_rl3 = agent.select_action(state3, [2, 3], mode='mixed')
                        trajectory.append((state3, action3, 0, None, False, used_rl3))
                    else:
                        action3 = self.linear_player.get_action(hands[first_player], boards, [2, 3])
                    
                    if action3 == 3:  # fold
                        game.stacks[second_player] += pot
                        reward = (game.stacks[agent_position] - stacks[agent_position]) / big_blind
                        self.finalize_agent_trajectory(agent, trajectory, reward)
                        return reward
                    else:  # call
                        call_amount = min(bet_amount, game.stacks[first_player])
                        game.stacks[first_player] -= call_amount
                        pot += call_amount
        
        # Showdown
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
        
        reward = (game.stacks[agent_position] - stacks[agent_position]) / big_blind
        self.finalize_agent_trajectory(agent, trajectory, reward)
        return reward
    
    def finalize_agent_trajectory(self, agent, trajectory, reward):
        """Add single agent's trajectory to replay buffers"""
        reward_clipped = max(min(reward, agent.reward_clip), -agent.reward_clip)
        
        for i, (state, action, _, _, _, used_rl) in enumerate(trajectory):
            next_state = trajectory[i+1][0] if i+1 < len(trajectory) else state
            done = i == len(trajectory) - 1
            r = reward_clipped if done else 0.0
            
            # Add to RL buffer
            agent.rl_buffer.add(state, action, r, next_state, float(done))
            
            # Add to SL buffer when NOT using RL
            if not used_rl:
                agent.sl_buffer.add(state, action)
    
    async def play_hand_nfsp(self, stacks, big_blind, dealer):
        """Play one hand using NFSP agents"""
        game = PokerGame(stacks[:], big_blind, dealer)
        
        # Deal cards
        cards = list(range(1, 53))
        random.shuffle(cards)
        hands = game.deal_hands(cards)
        boards = game.deal_flops(cards)
        
        # Initial pot (blinds already posted)
        pot = 2 * big_blind
        game.stacks[0] -= big_blind
        game.stacks[1] -= big_blind
        
        # Track trajectories for each player
        trajectory_p1 = []  # [(state, action, reward, next_state, done, used_rl)]
        trajectory_p2 = []
        
        # Simplified betting: FLOP and TURN only (2 rounds)
        for street_idx, street_name in enumerate(['FLOP', 'TURN']):
            if street_idx == 1:
                # Deal turn cards
                boards = game.deal_turns(cards, boards[0], boards[1])
            
            # Determine acting order
            first_player = 1 - dealer
            second_player = dealer
            
            # First player action
            state1 = encode_state(
                hands[first_player], boards, pot,
                [game.stacks[first_player], game.stacks[second_player]],
                first_player, street_name, 0
            )
            agent1 = self.agent_p1 if first_player == 0 else self.agent_p2
            action1, used_rl1 = agent1.select_action(state1, [0, 1], mode='mixed')
            
            bet_to_call = 0
            if action1 == 1:  # bet pot
                bet_amount = min(pot, game.stacks[first_player], game.stacks[second_player])
                game.stacks[first_player] -= bet_amount
                pot += bet_amount
                bet_to_call = bet_amount
            
            # Second player action
            legal2 = [2, 3] if bet_to_call > 0 else [0, 1]
            state2 = encode_state(
                hands[second_player], boards, pot,
                [game.stacks[second_player], game.stacks[first_player]],
                second_player, street_name, bet_to_call
            )
            agent2 = self.agent_p1 if second_player == 0 else self.agent_p2
            action2, used_rl2 = agent2.select_action(state2, legal2, mode='mixed')
            
            # Store transitions for first player
            if first_player == 0:
                trajectory_p1.append((state1, action1, 0, None, False, used_rl1))
            else:
                trajectory_p2.append((state1, action1, 0, None, False, used_rl1))
            
            # Store transitions for second player
            if second_player == 0:
                trajectory_p1.append((state2, action2, 0, None, False, used_rl2))
            else:
                trajectory_p2.append((state2, action2, 0, None, False, used_rl2))
            
            # Process second player's action
            if bet_to_call > 0:
                if action2 == 3:  # fold
                    # First player wins
                    game.stacks[first_player] += pot
                    reward_p1 = (game.stacks[0] - stacks[0]) / big_blind
                    reward_p2 = (game.stacks[1] - stacks[1]) / big_blind
                    self.finalize_trajectories(trajectory_p1, trajectory_p2, reward_p1, reward_p2)
                    return reward_p1, reward_p2
                elif action2 == 2:  # call
                    call_amount = min(bet_to_call, game.stacks[second_player])
                    game.stacks[second_player] -= call_amount
                    pot += call_amount
            else:
                if action2 == 1:  # bet pot
                    bet_amount = min(pot, game.stacks[second_player], game.stacks[first_player])
                    game.stacks[second_player] -= bet_amount
                    pot += bet_amount
                    # First player must respond to bet
                    state3 = encode_state(
                        hands[first_player], boards, pot,
                        [game.stacks[first_player], game.stacks[second_player]],
                        first_player, street_name, bet_amount
                    )
                    action3, used_rl3 = agent1.select_action(state3, [2, 3], mode='mixed')
                    
                    if first_player == 0:
                        trajectory_p1.append((state3, action3, 0, None, False, used_rl3))
                    else:
                        trajectory_p2.append((state3, action3, 0, None, False, used_rl3))
                    
                    if action3 == 3:  # fold
                        game.stacks[second_player] += pot
                        reward_p1 = (game.stacks[0] - stacks[0]) / big_blind
                        reward_p2 = (game.stacks[1] - stacks[1]) / big_blind
                        self.finalize_trajectories(trajectory_p1, trajectory_p2, reward_p1, reward_p2)
                        return reward_p1, reward_p2
                    else:  # call
                        call_amount = min(bet_amount, game.stacks[first_player])
                        game.stacks[first_player] -= call_amount
                        pot += call_amount
        
        # Showdown - determine winners
        boards = game.deal_rivers(cards, boards[0], boards[1])
        best_b1_p1, _ = game.omaha_hand_strength(hands[0], boards[0])
        best_b2_p1, _ = game.omaha_hand_strength(hands[0], boards[1])
        best_b1_p2, _ = game.omaha_hand_strength(hands[1], boards[0])
        best_b2_p2, _ = game.omaha_hand_strength(hands[1], boards[1])
        
        # Calculate winnings for each board
        w1, w2 = 0, 0
        if best_b1_p1 > best_b1_p2:
            w1 += 0.5
        elif best_b1_p1 < best_b1_p2:
            w2 += 0.5
        else:
            w1 += 0.25
            w2 += 0.25
        
        if best_b2_p1 > best_b2_p2:
            w1 += 0.5
        elif best_b2_p1 < best_b2_p2:
            w2 += 0.5
        else:
            w1 += 0.25
            w2 += 0.25
        
        game.stacks[0] += int(w1 * pot)
        game.stacks[1] += int(w2 * pot)
        
        reward_p1 = (game.stacks[0] - stacks[0]) / big_blind
        reward_p2 = (game.stacks[1] - stacks[1]) / big_blind
        
        self.finalize_trajectories(trajectory_p1, trajectory_p2, reward_p1, reward_p2)
        return reward_p1, reward_p2
    
    def finalize_trajectories(self, traj_p1, traj_p2, reward_p1, reward_p2):
        """Add transitions to replay buffers with final rewards"""
        # Clip rewards for stability
        reward_p1_clipped = max(min(reward_p1, self.agent_p1.reward_clip), -self.agent_p1.reward_clip)
        reward_p2_clipped = max(min(reward_p2, self.agent_p2.reward_clip), -self.agent_p2.reward_clip)
        
        # Process player 1 trajectory
        for i, (state, action, _, _, _, used_rl) in enumerate(traj_p1):
            next_state = traj_p1[i+1][0] if i+1 < len(traj_p1) else state
            done = i == len(traj_p1) - 1
            r = reward_p1_clipped if done else 0.0
            
            # Add to RL buffer
            self.agent_p1.rl_buffer.add(state, action, r, next_state, float(done))
            
            # CRITICAL FIX: Add to SL buffer when NOT using RL (i.e., when using average strategy)
            if not used_rl:
                self.agent_p1.sl_buffer.add(state, action)
        
        # Process player 2 trajectory
        for i, (state, action, _, _, _, used_rl) in enumerate(traj_p2):
            next_state = traj_p2[i+1][0] if i+1 < len(traj_p2) else state
            done = i == len(traj_p2) - 1
            r = reward_p2_clipped if done else 0.0
            
            # Add to RL buffer
            self.agent_p2.rl_buffer.add(state, action, r, next_state, float(done))
            
            # CRITICAL FIX: Add to SL buffer when NOT using RL
            if not used_rl:
                self.agent_p2.sl_buffer.add(state, action)
    
    async def pretrain_vs_linear(self):
        """Phase 1: Pretrain both agents against LinearPlayer"""
        import asyncio
        
        print("="*70)
        print("PHASE 1: PRETRAINING VS LINEAR PLAYER")
        print("="*70)
        print(f"Iterations: {self.pretrain_iterations}")
        print(f"Linear strategy: Bet pot with 2 pair+, otherwise check/fold")
        print(f"RL learning rate: {self.agent_p1.q_optimizer.param_groups[0]['lr']:.6f}")
        print(f"SL learning rate: {self.agent_p1.policy_optimizer.param_groups[0]['lr']:.6f}")
        print(f"Gamma: {self.agent_p1.gamma}")
        print(f"Reward clip: ±{self.agent_p1.reward_clip}\n")
        
        pretrain_rewards_p1 = []
        pretrain_rewards_p2 = []
        
        for i in range(1, self.pretrain_iterations + 1):
            stacks = [100, 100]
            dealer = random.choice([0, 1])
            
            # Agent 1 plays vs linear (alternating positions)
            agent1_position = i % 2
            r1 = await self.play_hand_vs_linear(self.agent_p1, agent1_position, stacks[:], 1, dealer)
            pretrain_rewards_p1.append(r1)
            
            # Agent 2 plays vs linear (alternating positions)
            stacks = [100, 100]
            dealer = random.choice([0, 1])
            agent2_position = i % 2
            r2 = await self.play_hand_vs_linear(self.agent_p2, agent2_position, stacks[:], 1, dealer)
            pretrain_rewards_p2.append(r2)
            
            # Train networks every 4 hands
            if i % 4 == 0:
                loss_rl_p1 = self.agent_p1.train_rl(batch_size=128)
                loss_sl_p1 = self.agent_p1.train_sl(batch_size=128)
                loss_rl_p2 = self.agent_p2.train_rl(batch_size=128)
                loss_sl_p2 = self.agent_p2.train_sl(batch_size=128)
                
                self.rl_losses.append((loss_rl_p1 + loss_rl_p2) / 2)
                self.sl_losses.append((loss_sl_p1 + loss_sl_p2) / 2)
            
            # Print progress
            if i % 500 == 0:
                avg_rl_loss = np.mean(self.rl_losses[-100:]) if self.rl_losses else 0
                avg_sl_loss = np.mean(self.sl_losses[-100:]) if self.sl_losses else 0
                avg_reward_p1 = np.mean(pretrain_rewards_p1[-100:]) if pretrain_rewards_p1 else 0
                avg_reward_p2 = np.mean(pretrain_rewards_p2[-100:]) if pretrain_rewards_p2 else 0
                print(f"[Pretrain] {i}/{self.pretrain_iterations} | "
                      f"RL_loss: {avg_rl_loss:.4f} | SL_loss: {avg_sl_loss:.4f} | "
                      f"R_P1: {avg_reward_p1:+.2f} | R_P2: {avg_reward_p2:+.2f} | "
                      f"RL_buf: {len(self.agent_p1.rl_buffer)} | "
                      f"SL_buf: {len(self.agent_p1.sl_buffer)}")
        
        final_reward_p1 = np.mean(pretrain_rewards_p1[-500:]) if len(pretrain_rewards_p1) >= 500 else np.mean(pretrain_rewards_p1)
        final_reward_p2 = np.mean(pretrain_rewards_p2[-500:]) if len(pretrain_rewards_p2) >= 500 else np.mean(pretrain_rewards_p2)
        
        print("\n" + "-"*70)
        print("PRETRAINING COMPLETE")
        print(f"Agent 1 avg reward vs Linear: {final_reward_p1:+.3f} bb/hand")
        print(f"Agent 2 avg reward vs Linear: {final_reward_p2:+.3f} bb/hand")
        print(f"RL buffer size: {len(self.agent_p1.rl_buffer)}")
        print(f"SL buffer size: {len(self.agent_p1.sl_buffer)}")
        print("-"*70 + "\n")
    
    async def train_selfplay(self):
        """Phase 2: Mixed training (self-play + vs Linear) for continued improvement"""
        import asyncio
        
        print("="*70)
        print("PHASE 2: MIXED TRAINING (SELF-PLAY + VS LINEAR)")
        print("="*70)
        print(f"Iterations: {self.selfplay_iterations}")
        print(f"Linear opponent mix rate: {self.linear_mix_rate*100:.0f}%")
        print(f"Action mapping: 0=check, 1=bet_pot, 2=call, 3=fold\n")
        
        selfplay_count = 0
        vs_linear_count = 0
        
        for i in range(1, self.selfplay_iterations + 1):
            stacks = [100, 100]
            dealer = random.choice([0, 1])
            
            # Decide: self-play or vs Linear
            if random.random() < self.linear_mix_rate:
                # Play vs Linear (alternate which agent)
                agent = self.agent_p1 if i % 2 == 0 else self.agent_p2
                agent_position = i % 2
                r = await self.play_hand_vs_linear(agent, agent_position, stacks, 1, dealer)
                self.vs_linear_rewards.append(r)
                vs_linear_count += 1
            else:
                # Self-play
                r1, r2 = await self.play_hand_nfsp(stacks, 1, dealer)
                self.rewards_p1.append(r1)
                self.rewards_p2.append(r2)
                selfplay_count += 1
            
            # Train networks every 4 hands
            if i % 4 == 0:
                loss_rl_p1 = self.agent_p1.train_rl(batch_size=128)
                loss_sl_p1 = self.agent_p1.train_sl(batch_size=128)
                loss_rl_p2 = self.agent_p2.train_rl(batch_size=128)
                loss_sl_p2 = self.agent_p2.train_sl(batch_size=128)
                
                self.rl_losses.append((loss_rl_p1 + loss_rl_p2) / 2)
                self.sl_losses.append((loss_sl_p1 + loss_sl_p2) / 2)
            
            # Print progress
            if i % 500 == 0:
                avg_rl_loss = np.mean(self.rl_losses[-100:]) if self.rl_losses else 0
                avg_sl_loss = np.mean(self.sl_losses[-100:]) if self.sl_losses else 0
                avg_reward_p1 = np.mean(self.rewards_p1[-100:]) if self.rewards_p1 else 0
                avg_reward_p2 = np.mean(self.rewards_p2[-100:]) if self.rewards_p2 else 0
                avg_vs_linear = np.mean(self.vs_linear_rewards[-100:]) if self.vs_linear_rewards else 0
                print(f"[Mixed] {i}/{self.selfplay_iterations} | "
                      f"RL_loss: {avg_rl_loss:.4f} | SL_loss: {avg_sl_loss:.4f} | "
                      f"SP_R: {avg_reward_p1:+.2f} | vsLin: {avg_vs_linear:+.2f} | "
                      f"RL_buf: {len(self.agent_p1.rl_buffer)} | "
                      f"SL_buf: {len(self.agent_p1.sl_buffer)}")
        
        print("\n" + "="*70)
        print("MIXED TRAINING COMPLETE")
        print(f"Self-play games: {selfplay_count} ({selfplay_count/(selfplay_count+vs_linear_count)*100:.1f}%)")
        print(f"Vs Linear games: {vs_linear_count} ({vs_linear_count/(selfplay_count+vs_linear_count)*100:.1f}%)")
        print(f"Final RL buffer size: {len(self.agent_p1.rl_buffer)}")
        print(f"Final SL buffer size: {len(self.agent_p1.sl_buffer)}")
        
        if self.rewards_p1:
            final_window = min(500, len(self.rewards_p1))
            print(f"Final self-play reward (last {final_window}): {np.mean(self.rewards_p1[-final_window:]):.2f}")
        
        if self.vs_linear_rewards:
            final_window = min(500, len(self.vs_linear_rewards))
            print(f"Final vs Linear reward (last {final_window}): {np.mean(self.vs_linear_rewards[-final_window:]):+.3f} bb/hand")
            
            if np.mean(self.vs_linear_rewards[-final_window:]) > 0.1:
                print("✓ BEATING LINEAR PLAYER!")
            elif np.mean(self.vs_linear_rewards[-final_window:]) > -0.1:
                print("~ COMPETITIVE WITH LINEAR PLAYER")
            else:
                print("⚠ STILL LOSING TO LINEAR PLAYER - Consider more training")
        print("="*70)
        
        # Save models
        torch.save(self.agent_p1.q_network.state_dict(), 'q_network_p1.pth')
        torch.save(self.agent_p1.policy_network.state_dict(), 'policy_network_p1.pth')
        print("\nModels saved: q_network_p1.pth, policy_network_p1.pth")
    
    async def train(self):
        """Complete training: pretrain + self-play"""
        if self.pretrain_iterations > 0:
            await self.pretrain_vs_linear()
        
        if self.selfplay_iterations > 0:
            await self.train_selfplay()
        
        print("\n" + "="*70)
        print("ALL TRAINING COMPLETE!")
        print("="*70)


def main():
    import sys
    import asyncio
    
    # Parse command line arguments
    # Usage: python nfsp_pretrained.py [pretrain_iters] [mixed_iters] [linear_mix_rate]
    # Example: python nfsp_pretrained.py 10000 40000 0.5
    #   - 10k iterations purely vs Linear
    #   - 40k iterations mixed (50% self-play, 50% vs Linear)
    
    if len(sys.argv) >= 4:
        pretrain_iterations = int(sys.argv[1])
        selfplay_iterations = int(sys.argv[2])
        linear_mix_rate = float(sys.argv[3])
    elif len(sys.argv) >= 3:
        pretrain_iterations = int(sys.argv[1])
        selfplay_iterations = int(sys.argv[2])
        linear_mix_rate = 0.3  # Default 30% vs Linear during mixed phase
    elif len(sys.argv) == 2:
        # If only one argument, split 40% pretrain, 60% mixed
        total = int(sys.argv[1])
        pretrain_iterations = int(total * 0.4)
        selfplay_iterations = total - pretrain_iterations
        linear_mix_rate = 0.3
    else:
        # Default: More training, higher Linear mix
        pretrain_iterations = 20000  # Pure vs Linear
        selfplay_iterations = 30000  # Mixed
        linear_mix_rate = 0.5  # 50% vs Linear during mixed phase
    
    print(f"\nStarting NFSP training with LinearPlayer opponent:")
    print(f"  Phase 1 (Pure vs Linear): {pretrain_iterations} iterations")
    print(f"  Phase 2 (Mixed training): {selfplay_iterations} iterations")
    print(f"  Linear mix rate in Phase 2: {linear_mix_rate*100:.0f}%")
    print(f"  Total iterations: {pretrain_iterations + selfplay_iterations}")
    print(f"\nStrategy: Learn to beat Linear, then generalize with self-play\n")
    
    trainer = NFSPTrainer(
        pretrain_iterations=pretrain_iterations,
        selfplay_iterations=selfplay_iterations,
        linear_mix_rate=linear_mix_rate
    )
    asyncio.run(trainer.train())


if __name__ == "__main__":
    main()
