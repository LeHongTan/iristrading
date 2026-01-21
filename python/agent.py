import sys
import json
import numpy as np

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
np.random.seed(SEED)

# TODO: Thay bằng nhập mô hình AI RL thực tế của bạn
def get_action(state):
    # Hiện tại trả random; sau này dùng model.predict(state)
    return [int(np.random.choice([-1, 0, 1])) for _ in state]  # buy/hold/sell cho mỗi symbol

if __name__ == "__main__":
    state = json.loads(sys.stdin.read())
    action = get_action(state)
    print(json.dumps(action))