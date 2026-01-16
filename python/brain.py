import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class ActorCritic(nn.Module):
    def __init__(self, state_dim=20, hidden_dim=256, action_dim=3):
        super(ActorCritic, self).__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )
        
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim),
        )
        
        self.critic_head = nn. Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn. Linear):
                nn.init.orthogonal_(m. weight, gain=np.sqrt(2))
                nn.init.zeros_(m. bias)
        
        for m in self.actor_head. modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m. weight, gain=0.01)
    
    def forward(self, state):
        shared_features = self.shared(state)
        action_logits = self.actor_head(shared_features)
        value = self. critic_head(shared_features)
        return action_logits, value
    
    def get_action_probs(self, state):
        action_logits, _ = self.forward(state)
        return F.softmax(action_logits, dim=-1)
    
    def get_value(self, state):
        _, value = self.forward(state)
        return value


class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self. values = []
        self.log_probs = []
        self. dones = []
    
    def store(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs. append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        self.states. clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs. clear()
        self.dones.clear()
    
    def get_batches(self, batch_size):
        n_states = len(self. states)
        indices = np.arange(n_states)
        np.random.shuffle(indices)
        
        batches = []
        for start in range(0, n_states, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            batches.append(batch_indices)
        
        return batches


class PPOAgent:
    def __init__(self, state_dim=20, hidden_dim=256, action_dim=3,
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, 
                 clip_epsilon=0.2, entropy_coef=0.01, value_coef=0.5):
        
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("[AI Brain] Using Apple Silicon MPS acceleration")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("[AI Brain] Using CUDA acceleration")
        else:
            self.device = torch.device("cpu")
            print("[AI Brain] Using CPU")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self. gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self. entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        self.network = ActorCritic(state_dim, hidden_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network. parameters(), lr=lr)
        
        self.memory = PPOMemory()
        
        self.action_names = ["HOLD", "BUY", "SELL"]
        self.training_step = 0
        
        print(f"[AI Brain] PPO Agent initialized | State dim: {state_dim} | Actions: {action_dim}")
    
    def normalize_state(self, state_vector):
        if not isinstance(state_vector, (list, np.ndarray)):
            state_vector = list(state_vector)
        
        state_vector = list(state_vector)
        
        if len(state_vector) < self.state_dim:
            state_vector.extend([0.0] * (self.state_dim - len(state_vector)))
        elif len(state_vector) > self.state_dim:
            state_vector = state_vector[:self. state_dim]
        
        state_array = np.array(state_vector, dtype=np.float32)
        
        state_array = np.nan_to_num(state_array, nan=0.0, posinf=1.0, neginf=-1.0)
        state_array = np.clip(state_array, -10.0, 10.0)
        
        return state_array
    
    def get_action(self, state_vector):
        try:
            state = self.normalize_state(state_vector)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch. no_grad():
                action_probs = self.network. get_action_probs(state_tensor)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                
                action_idx = action.item()
                probs = action_probs.cpu().numpy()[0]
                
                return action_idx
                
        except Exception as e:
            print(f"[AI Brain] Error:  {e}")
            return 0
    
    def get_action_with_info(self, state_vector):
        try:
            state = self.normalize_state(state_vector)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action_logits, value = self.network(state_tensor)
                action_probs = F.softmax(action_logits, dim=-1)
                dist = torch.distributions. Categorical(action_probs)
                action = dist.sample()
                log_prob = dist. log_prob(action)
                
                return {
                    'action': action.item(),
                    'log_prob': log_prob.item(),
                    'value': value.item(),
                    'probs': action_probs.cpu().numpy()[0].tolist(),
                    'entropy': dist.entropy().item()
                }
                
        except Exception as e:
            print(f"[AI Brain] Error: {e}")
            return {'action': 0, 'log_prob': 0.0, 'value': 0.0, 'probs': [1.0, 0.0, 0.0], 'entropy': 0.0}
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        state = self.normalize_state(state)
        self.memory.store(state, action, reward, value, log_prob, done)
    
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
        if len(self.memory. states) < batch_size:
            return {}
        
        advantages, returns = self. compute_gae(
            self.memory.rewards,
            self.memory.values,
            self.memory. dones,
            next_value
        )
        
        states = torch.FloatTensor(np.array(self. memory.states)).to(self.device)
        actions = torch.LongTensor(self.memory. actions).to(self.device)
        old_log_probs = torch.FloatTensor(self. memory.log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for _ in range(epochs):
            batches = self.memory.get_batches(batch_size)
            
            for batch_indices in batches: 
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                action_logits, values = self.network(batch_states)
                action_probs = F.softmax(action_logits, dim=-1)
                dist = torch.distributions.Categorical(action_probs)
                
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F. mse_loss(values. squeeze(), batch_returns)
                
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils. clip_grad_norm_(self.network. parameters(), 0.5)
                self.optimizer.step()
                
                total_loss += loss. item()
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy. item())
        
        self.memory.clear()
        self.training_step += 1
        
        return {
            'total_loss':  total_loss,
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy':  np.mean(entropy_losses)
        }
    
    def save(self, path):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer. state_dict(),
            'training_step': self.training_step
        }, path)
        print(f"[AI Brain] Model saved to {path}")
    
    def load(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_step = checkpoint. get('training_step', 0)
            print(f"[AI Brain] Model loaded from {path}")
            return True
        return False


_agent = None

def initialize_agent(state_dim=20, hidden_dim=256, action_dim=3):
    global _agent
    if _agent is None: 
        _agent = PPOAgent(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim)
    return _agent

def get_action(state_vector):
    global _agent
    if _agent is None:
        _agent = initialize_agent()
    return _agent.get_action(state_vector)

def get_action_with_info(state_vector):
    global _agent
    if _agent is None:
        _agent = initialize_agent()
    return _agent.get_action_with_info(state_vector)

def store_transition(state, action, reward, value, log_prob, done):
    global _agent
    if _agent is None:
        _agent = initialize_agent()
    _agent.store_transition(state, action, reward, value, log_prob, done)

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


if __name__ == "__main__":
    print("Testing PPO Agent...")
    agent = initialize_agent()
    
    test_state = [0.1] * 20
    action = get_action(test_state)
    print(f"Test action: {action}")
    
    info = get_action_with_info(test_state)
    print(f"Action info: {info}")
    
    print("PPO Agent test passed!")