import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# ==== Mạng Temporal Encoder cho chuỗi nến từng symbol ====
class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1 if num_layers > 1 else 0.0)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        last_out = out[:, -1, :]
        return self.layer_norm(last_out)

# ==== Mạng Actor-Critic PPO đa symbol ====
class MultiSymbolActorCritic(nn.Module):
    def __init__(self, num_symbols=5, sequence_length=256, features_per_timestep=20, hidden_dim=256, num_directions=3):
        super().__init__()
        self.num_symbols = num_symbols
        self.sequence_length = sequence_length
        self.features_per_timestep = features_per_timestep
        self.hidden_dim = hidden_dim
        self.num_directions = num_directions

        # Encoder chuỗi nến cho từng symbol
        self.symbol_encoders = nn.ModuleList([
            TemporalEncoder(features_per_timestep, hidden_dim // 2) for _ in range(num_symbols)
        ])

        # Xử lý features Portfolio (ví dụ 3 đặc trưng)
        self.portfolio_fc = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.ReLU(),
        )

        # Gộp encoding của các symbol + portfolio
        combined_dim = num_symbols * (hidden_dim // 2) + (hidden_dim // 4)
        self.shared = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Đầu ra hành động Direction (Cho mỗi symbol)
        self.direction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_directions),
            ) for _ in range(num_symbols)
        ])

        # Đầu ra hối lượng (size) sử dụng Beta Distribution
        self.size_alpha_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus(),  # Để đầu ra luôn dương
            ) for _ in range(num_symbols)
        ])
        self.size_beta_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus(),
            ) for _ in range(num_symbols)
        ])

        # Đầu ra đánh giá giá trị (Critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, symbol_sequences, portfolio_features):
        # symbol_sequences: List[num_symbols][sequence_length][features_per_timestep]
        # portfolio_features: List[3]
        device = next(self.parameters()).device
        batch_size = 1
        encoded_symbols = []
        for i in range(self.num_symbols):
            x = torch.tensor(symbol_sequences[i], dtype=torch.float32, device=device).unsqueeze(0)
            out = self.symbol_encoders[i](x)
            encoded_symbols.append(out)
        enc_symbols = torch.cat(encoded_symbols, dim=-1)

        pf = torch.tensor(portfolio_features, dtype=torch.float32, device=device).unsqueeze(0)
        pf_enc = self.portfolio_fc(pf)

        features = torch.cat([enc_symbols, pf_enc], dim=-1)
        shared_out = self.shared(features)

        directions = []
        sizes = []
        log_probs = []
        for i in range(self.num_symbols):
            logits = self.direction_heads[i](shared_out)
            dir_dist = torch.distributions.Categorical(logits=logits)
            dir_action = dir_dist.sample()
            directions.append(dir_action.item())
            log_probs.append(dir_dist.log_prob(dir_action).item())

            alpha = self.size_alpha_heads[i](shared_out)
            beta = self.size_beta_heads[i](shared_out)
            size_dist = torch.distributions.Beta(alpha + 1e-2, beta + 1e-2)
            size = size_dist.sample()
            sizes.append(size.item())

        value = self.value_head(shared_out)
        return directions, sizes, log_probs, value.item()

# ==== Replay buffer cho PPO ====
class Memory:
    def __init__(self):
        self.data = []

    def store(self, state, action, reward, value, log_prob, done):
        self.data.append((state, action, reward, value, log_prob, done))

    def clear(self):
        self.data.clear()

    def all(self):
        return self.data

# ==== Agent & Toàn bộ API tương tác Rust ====
_agent = None
_optimizer = None
_buffer = None

def initialize_agent(num_symbols, sequence_length, features_per_timestep, hidden_dim, **kwargs):
    global _agent, _optimizer, _buffer
    _agent = MultiSymbolActorCritic(
        num_symbols=num_symbols,
        sequence_length=sequence_length,
        features_per_timestep=features_per_timestep,
        hidden_dim=hidden_dim
    )
    _agent.eval()
    _optimizer = torch.optim.Adam(_agent.parameters(), lr=3e-4)
    _buffer = Memory()

def get_action_with_info(symbol_sequences, portfolio_features):
    global _agent
    with torch.no_grad():
        directions, sizes, log_probs, value = _agent(symbol_sequences, portfolio_features)
        return {
            "directions": directions,
            "sizes": sizes,
            "log_prob": float(np.mean(log_probs)),
            "value": value
        }

def store_transition(symbol_sequences, portfolio_features, directions, sizes, reward, value, log_prob, done):
    global _buffer
    _buffer.store((symbol_sequences, portfolio_features), (directions, sizes), reward, value, log_prob, done)

def train_step(next_value):
    global _agent, _optimizer, _buffer
    # Dummy: Thực tế ở đây sẽ update PPO bằng replay buffer _buffer.all()
    # Ở bản đầy đủ, bạn phải triển khai PPO train thực sự
    if not _buffer.data:
        return {"policy_loss": 0.0, "value_loss": 0.0}
    _optimizer.zero_grad()
    # ... tính toán loss, update ...
    _buffer.clear()
    return {"policy_loss": 0.0, "value_loss": 0.0}

def save_model(path):
    global _agent
    torch.save(_agent.state_dict(), path)
    return True

def load_model(path):
    global _agent
    _agent.load_state_dict(torch.load(path, map_location="cpu"))
    return True