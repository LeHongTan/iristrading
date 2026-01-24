import os
import argparse
import gymnasium as gym
from trading_env import TradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"])
    parser.add_argument("--timeframes", nargs="+", default=["1h","5m","1m","4h"])
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--tb_log", type=str, default="logs_sb3")
    parser.add_argument("--save", type=str, default="ppo_ict_trading.zip")
    parser.add_argument("--eval_freq", type=int, default=40_000)
    args = parser.parse_args()
    print("Start train: ", args)

    env = TradingEnv(
        data_dir=args.data_dir,
        symbols=args.symbols,
        timeframes=args.timeframes,
        sequence_length=256,
        initial_balance=20.0,
    )
    eval_env = TradingEnv(
        data_dir=args.data_dir,
        symbols=args.symbols,
        timeframes=args.timeframes,
        sequence_length=256,
        initial_balance=20.0,
    )

    model = PPO(
        "MlpPolicy", env,
        verbose=1,
        tensorboard_log=args.tb_log,
        n_steps=2048, batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./logs_eval/",
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False
    )

    model.learn(total_timesteps=args.steps, callback=eval_callback)
    model.save(args.save)
    print(f"Training done. Model saved to {args.save}")

    # Visualize final equity curve
    obs, _ = eval_env.reset()
    done = False
    eq_list = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = eval_env.step(action)
        eq_list.append(info["equity"])
    import matplotlib.pyplot as plt
    plt.plot(eq_list)
    plt.title("Eval Equity Curve")
    plt.show()

if __name__ == "__main__":
    main()