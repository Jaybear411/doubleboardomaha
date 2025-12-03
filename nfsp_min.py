# Neural Fictitious Self-Play (NFSP) for Double Board Omaha Poker
# Uses RL for best response and SL for average strategy

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


# NFSP Training Loop with self-play
class NFSPTrainer:
    """Trainer for NFSP agents using self-play"""
    def __init__(self, iterations=10000):
        self.iterations = iterations
        self.agent_p1 = NFSPAgent()
        self.agent_p2 = NFSPAgent()
        
        # Stats
        self.rl_losses = []
        self.sl_losses = []
        self.rewards_p1 = []
        self.rewards_p2 = []
    
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
    
    async def train(self):
        """Main training loop"""
        import asyncio
        
        print(f"Training NFSP for {self.iterations} iterations...")
        print(f"Action mapping: 0=check, 1=bet_pot, 2=call, 3=fold")
        print(f"RL learning rate: {self.agent_p1.q_optimizer.param_groups[0]['lr']:.6f}")
        print(f"SL learning rate: {self.agent_p1.policy_optimizer.param_groups[0]['lr']:.6f}")
        print(f"Gamma: {self.agent_p1.gamma}")
        print(f"Reward clip: ±{self.agent_p1.reward_clip}\n")
        
        for i in range(1, self.iterations + 1):
            # Play one hand
            stacks = [100, 100]
            dealer = random.choice([0, 1])
            r1, r2 = await self.play_hand_nfsp(stacks, 1, dealer)
            
            self.rewards_p1.append(r1)
            self.rewards_p2.append(r2)
            
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
                print(f"Iter {i}/{self.iterations} | "
                      f"RL_loss: {avg_rl_loss:.4f} | SL_loss: {avg_sl_loss:.4f} | "
                      f"Avg_R_P1: {avg_reward_p1:.2f} | Avg_R_P2: {avg_reward_p2:.2f} | "
                      f"RL_buf: {len(self.agent_p1.rl_buffer)} | "
                      f"SL_buf: {len(self.agent_p1.sl_buffer)}")
        
        print("\n" + "="*60)
        print("Training complete!")
        print(f"Final RL buffer size: {len(self.agent_p1.rl_buffer)}")
        print(f"Final SL buffer size: {len(self.agent_p1.sl_buffer)}")
        print(f"Final avg reward P1 (last 500): {np.mean(self.rewards_p1[-500:]):.2f}")
        print(f"Final avg reward P2 (last 500): {np.mean(self.rewards_p2[-500:]):.2f}")
        print("="*60)
        
        # Save models
        torch.save(self.agent_p1.q_network.state_dict(), 'q_network_p1.pth')
        torch.save(self.agent_p1.policy_network.state_dict(), 'policy_network_p1.pth')
        print("\nModels saved: q_network_p1.pth, policy_network_p1.pth")


def main():
    import sys
    import asyncio
    
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    
    trainer = NFSPTrainer(iterations=iterations)
    asyncio.run(trainer.train())


if __name__ == "__main__":
    main()
