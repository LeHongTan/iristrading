import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class TemporalEncoder(nn.Module):
    """GRU-based temporal encoder for sequence processing"""
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(TemporalEncoder, self).__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.gru(x)
        # Take the last output
        last_out = out[:, -1, :]
        return self.layer_norm(last_out)


class MultiSymbolActorCritic(nn.Module):
    """
    Actor-Critic for multi-symbol portfolio trading with:
    - Sequence input per symbol (L x F)
    - Multiple action heads per symbol: direction (categorical) + size (continuous)
    - Single critic head for value estimation
    """
    def __init__(
        self,
        num_symbols=5,
        sequence_length=256,
        features_per_timestep=20,
        hidden_dim=256,
        num_directions=3,  # HOLD, LONG, SHORT
    ):
        super(MultiSymbolActorCritic, self).__init__()
        
        self.num_symbols = num_symbols
        self.sequence_length = sequence_length
        self.features_per_timestep = features_per_timestep
        self.hidden_dim = hidden_dim
        self.num_directions = num_directions
        
        # Per-symbol temporal encoders
        self.symbol_encoders = nn.ModuleList([
            TemporalEncoder(features_per_timestep, hidden_dim // 2)
            for _ in range(num_symbols)
        ])
        
        # Portfolio-level feature processor
        self.portfolio_fc = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),  # 3 portfolio features
            nn.ReLU(),
        )
        
        # Combine all symbol encodings + portfolio features
        combined_dim = num_symbols * (hidden_dim // 2) + (hidden_dim // 4)
        
        self.shared = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Per-symbol action heads
        # Direction head (categorical): HOLD, LONG, SHORT
        self.direction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_directions),
            )
            for _ in range(num_symbols)
        ])
        
        # Size head (continuous): outputs parameters for Beta distribution
        # Beta distribution is bounded to [0, 1] which is perfect for size_fraction
        self.size_alpha_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus(),  # Ensure positive
            )
            for _ in range(num_symbols)
        ])
        
        self.size_beta_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus(),  # Ensure positive
            )
            for _ in range(num_symbols)
        ])
        
        # Critic head
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Smaller initialization for action heads
        for head in self.direction_heads:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=0.01)
    
    def forward(self, symbol_sequences, portfolio_features):
        """
        Args:
            symbol_sequences: list of tensors, each (batch, seq_len, features)
            portfolio_features: (batch, 3)
        
        Returns:
            direction_logits: list of (batch, num_directions) per symbol
            size_alphas: list of (batch, 1) per symbol
            size_betas: list of (batch, 1) per symbol
            value: (batch, 1)
        """
        # Encode each symbol sequence
        symbol_encodings = []
        for i, encoder in enumerate(self.symbol_encoders):
            encoding = encoder(symbol_sequences[i])
            symbol_encodings.append(encoding)
        
        # Encode portfolio features
        portfolio_encoding = self.portfolio_fc(portfolio_features)
        
        # Concatenate all encodings
        combined = torch.cat(symbol_encodings + [portfolio_encoding], dim=-1)
        
        # Shared features
        shared_features = self.shared(combined)
        
        # Per-symbol action heads
        direction_logits = []
        size_alphas = []
        size_betas = []
        
        for i in range(self.num_symbols):
            dir_logits = self.direction_heads[i](shared_features)
            direction_logits.append(dir_logits)
            
            alpha = self.size_alpha_heads[i](shared_features) + 1.0  # Ensure > 1 for stability
            beta = self.size_beta_heads[i](shared_features) + 1.0
            size_alphas.append(alpha)
            size_betas.append(beta)
        
        # Value head
        value = self.critic_head(shared_features)
        
        return direction_logits, size_alphas, size_betas, value
    
    def get_action(self, symbol_sequences, portfolio_features, deterministic=False):
        """
        Sample actions from the policy.
        
        Returns:
            directions: list of direction indices per symbol
            sizes: list of size fractions in [0, 1] per symbol
            log_probs: total log probability
            value: state value
        """
        direction_logits, size_alphas, size_betas, value = self.forward(
            symbol_sequences, portfolio_features
        )
        
        directions = []
        sizes = []
        log_probs = []
        eps = 1e-6  # Small epsilon for clamping to avoid log(0) in Beta distribution
        
        for i in range(self.num_symbols):
            # Direction (categorical)
            dir_probs = F.softmax(direction_logits[i], dim=-1)
            dir_dist = torch.distributions.Categorical(dir_probs)
            
            if deterministic:
                direction = dir_probs.argmax(dim=-1)
            else:
                direction = dir_dist.sample()
            
            dir_log_prob = dir_dist.log_prob(direction)
            
            # Size (Beta distribution)
            alpha = size_alphas[i].squeeze(-1)
            beta = size_betas[i].squeeze(-1)
            size_dist = torch.distributions.Beta(alpha, beta)
            
            if deterministic:
                # Use mean of Beta distribution
                size = alpha / (alpha + beta)
            else:
                size = size_dist.sample()
            
            # Clamp size to avoid log(0) or log(1-1) = log(0) in Beta log_prob
            # Use the same clamped value for both log_prob and execution to maintain policy gradient consistency
            size_clamped = size.clamp(eps, 1.0 - eps)
            size_log_prob = size_dist.log_prob(size_clamped)
            
            directions.append(direction)
            sizes.append(size_clamped)  # Store clamped value for consistency
            log_probs.append(dir_log_prob + size_log_prob)
        
        # Total log prob is sum across symbols
        total_log_prob = torch.stack(log_probs).sum(dim=0)
        
        return directions, sizes, total_log_prob, value


class PPOMemory:
    def __init__(self):
        self.symbol_sequences = []  # List of lists of symbol sequences
        self.portfolio_features = []
        self.actions_dir = []  # List of direction lists
        self.actions_size = []  # List of size lists
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def store(self, symbol_seqs, portfolio_feats, dirs, sizes, reward, value, log_prob, done):
        self.symbol_sequences.append(symbol_seqs)
        self.portfolio_features.append(portfolio_feats)
        self.actions_dir.append(dirs)
        self.actions_size.append(sizes)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        self.symbol_sequences.clear()
        self.portfolio_features.clear()
        self.actions_dir.clear()
        self.actions_size.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def get_batches(self, batch_size):
        n_states = len(self.symbol_sequences)
        indices = np.arange(n_states)
        np.random.shuffle(indices)
        
        batches = []
        for start in range(0, n_states, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            batches.append(batch_indices)
        
        return batches


class MultiSymbolPPOAgent:
    def __init__(
        self,
        num_symbols=5,
        sequence_length=256,
        features_per_timestep=20,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
    ):
        # Device selection
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("[AI Brain] Using Apple Silicon MPS acceleration")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("[AI Brain] Using CUDA acceleration")
        else:
            self.device = torch.device("cpu")
            print("[AI Brain] Using CPU")
        
        self.num_symbols = num_symbols
        self.sequence_length = sequence_length
        self.features_per_timestep = features_per_timestep
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        self.network = MultiSymbolActorCritic(
            num_symbols=num_symbols,
            sequence_length=sequence_length,
            features_per_timestep=features_per_timestep,
            hidden_dim=hidden_dim,
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.memory = PPOMemory()
        self.training_step = 0
        
        print(f"[AI Brain] Multi-Symbol PPO Agent initialized")
        print(f"  Symbols: {num_symbols}")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Features per timestep: {features_per_timestep}")
    
    def get_action_with_info(self, symbol_sequences, portfolio_features):
        """
        Get actions for all symbols.
        
        Args:
            symbol_sequences: list of numpy arrays, each (seq_len, features)
            portfolio_features: numpy array (3,)
        
        Returns:
            dict with:
                - directions: list of ints
                - sizes: list of floats in [0, 1]
                - log_prob: float
                - value: float
                - entropy: float (for monitoring)
        """
        try:
            # Convert to tensors
            symbol_tensors = []
            for seq in symbol_sequences:
                seq_array = np.array(seq, dtype=np.float32)
                # Ensure correct shape
                if len(seq_array.shape) == 1:
                    seq_array = seq_array.reshape(1, -1)
                # Pad or truncate to sequence_length
                if seq_array.shape[0] < self.sequence_length:
                    padding = np.zeros(
                        (self.sequence_length - seq_array.shape[0], seq_array.shape[1]),
                        dtype=np.float32
                    )
                    seq_array = np.vstack([padding, seq_array])
                elif seq_array.shape[0] > self.sequence_length:
                    seq_array = seq_array[-self.sequence_length:]
                
                tensor = torch.FloatTensor(seq_array).unsqueeze(0).to(self.device)
                symbol_tensors.append(tensor)
            
            portfolio_tensor = torch.FloatTensor(portfolio_features).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                directions, sizes, log_prob, value = self.network.get_action(
                    symbol_tensors, portfolio_tensor, deterministic=False
                )
                
                # Convert to Python types
                directions_list = [d.item() for d in directions]
                sizes_list = [s.item() for s in sizes]
                
                return {
                    'directions': directions_list,
                    'sizes': sizes_list,
                    'log_prob': log_prob.item(),
                    'value': value.item(),
                    'entropy': 0.0,  # Placeholder
                }
        
        except Exception as e:
            print(f"[AI Brain] Error in get_action_with_info: {e}")
            # Return safe defaults
            return {
                'directions': [0] * self.num_symbols,  # All HOLD
                'sizes': [0.0] * self.num_symbols,
                'log_prob': 0.0,
                'value': 0.0,
                'entropy': 0.0,
            }
    
    def store_transition(self, symbol_seqs, portfolio_feats, dirs, sizes, reward, value, log_prob, done):
        self.memory.store(symbol_seqs, portfolio_feats, dirs, sizes, reward, value, log_prob, done)
    
    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        
        values = values + [next_value]
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        
        return advantages, returns
    
    def train(self, next_value, epochs=4, batch_size=64):
        if len(self.memory.symbol_sequences) < batch_size:
            return {}
        
        advantages, returns = self.compute_gae(
            self.memory.rewards,
            self.memory.values,
            self.memory.dones,
            next_value
        )
        
        # Convert to tensors (simplified - in practice need to handle sequences properly)
        old_log_probs = torch.FloatTensor(self.memory.log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        policy_losses = []
        value_losses = []
        entropies = []
        total_losses = []
        eps = 1e-6  # Small epsilon for clamping
        
        for _ in range(epochs):
            batches = self.memory.get_batches(batch_size)
            
            for batch_indices in batches:
                # Prepare batch data
                batch_symbol_seqs = []
                for i in range(self.num_symbols):
                    symbol_batch = []
                    for idx in batch_indices:
                        seq = self.memory.symbol_sequences[idx][i]
                        symbol_batch.append(torch.FloatTensor(seq))
                    batch_symbol_seqs.append(torch.stack(symbol_batch).to(self.device))
                
                batch_portfolio = torch.FloatTensor(
                    [self.memory.portfolio_features[i] for i in batch_indices]
                ).to(self.device)
                
                batch_dirs = [[self.memory.actions_dir[i][j] for j in range(self.num_symbols)] 
                              for i in batch_indices]
                batch_sizes = [[self.memory.actions_size[i][j] for j in range(self.num_symbols)]
                               for i in batch_indices]
                
                # Forward pass
                direction_logits, size_alphas, size_betas, values = self.network(
                    batch_symbol_seqs, batch_portfolio
                )
                
                # Compute new log probs and entropy
                new_log_probs = []
                entropy_list = []
                for b in range(len(batch_indices)):
                    symbol_log_probs = []
                    symbol_entropies = []
                    for s in range(self.num_symbols):
                        # Direction
                        dir_probs = F.softmax(direction_logits[s][b:b+1], dim=-1)
                        dir_dist = torch.distributions.Categorical(dir_probs)
                        dir_lp = dir_dist.log_prob(torch.tensor([batch_dirs[b][s]]).to(self.device))
                        dir_entropy = dir_dist.entropy()
                        
                        # Size - clamp to avoid numerical issues
                        alpha = size_alphas[s][b].squeeze() 
                        beta = size_betas[s][b].squeeze()
                        size_dist = torch.distributions.Beta(alpha, beta)
                        size_value = torch.tensor([batch_sizes[b][s]]).to(self.device).clamp(eps, 1.0 - eps)
                        size_lp = size_dist.log_prob(size_value)
                        size_entropy = size_dist.entropy()
                        
                        symbol_log_probs.append(dir_lp + size_lp)
                        symbol_entropies.append(dir_entropy + size_entropy)
                    
                    new_log_probs.append(torch.stack(symbol_log_probs).sum())
                    entropy_list.append(torch.stack(symbol_entropies).sum())
                
                new_log_probs = torch.stack(new_log_probs)
                batch_entropy = torch.stack(entropy_list).mean()
                
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # PPO loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Total loss includes entropy bonus (negative entropy to encourage exploration)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * batch_entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(batch_entropy.item())
                total_losses.append(loss.item())
        
        self.memory.clear()
        self.training_step += 1
        
        return {
            'policy_loss': np.mean(policy_losses) if policy_losses else 0.0,
            'value_loss': np.mean(value_losses) if value_losses else 0.0,
            'entropy': np.mean(entropies) if entropies else 0.0,
            'total_loss': np.mean(total_losses) if total_losses else 0.0,
        }
    
    def save(self, path):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
        }, path)
        print(f"[AI Brain] Model saved to {path}")
    
    def load(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_step = checkpoint.get('training_step', 0)
            print(f"[AI Brain] Model loaded from {path}")
            return True
        return False


# Global agent instance
_agent = None

def initialize_agent(num_symbols=5, sequence_length=256, features_per_timestep=20, hidden_dim=256):
    global _agent
    if _agent is None:
        _agent = MultiSymbolPPOAgent(
            num_symbols=num_symbols,
            sequence_length=sequence_length,
            features_per_timestep=features_per_timestep,
            hidden_dim=hidden_dim,
        )
    return _agent

def get_action_with_info(symbol_sequences, portfolio_features):
    global _agent
    if _agent is None:
        _agent = initialize_agent()
    return _agent.get_action_with_info(symbol_sequences, portfolio_features)

def store_transition(symbol_seqs, portfolio_feats, dirs, sizes, reward, value, log_prob, done):
    global _agent
    if _agent is None:
        _agent = initialize_agent()
    _agent.store_transition(symbol_seqs, portfolio_feats, dirs, sizes, reward, value, log_prob, done)

def train_step(next_value):
    global _agent
    if _agent is None:
        _agent = initialize_agent()
    return _agent.train(next_value)

def save_model(path):
    global _agent
    if _agent is not None:
        _agent.save(path)

def load_model(path):
    global _agent
    if _agent is None:
        _agent = initialize_agent()
    return _agent.load(path)
