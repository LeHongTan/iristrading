import sys
import json
import numpy as np

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
N_SYMBOLS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
np.random.seed(SEED)

def get_action(state):
    # Bạn muốn agent trade thật không chỉ hold thì random cả BUY+SELL luôn:
    return [int(x) for x in np.random.choice([-1, 1], N_SYMBOLS)]
    # Muốn nhiều HOLD hơn thì dùng [-1, 0, 1] thay vì trên

if __name__ == "__main__":
    state = json.loads(sys.stdin.read())
    action = get_action(state)
    print(json.dumps(action))