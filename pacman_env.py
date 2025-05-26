import numpy as np
from random import choice
from collections import deque, defaultdict

class PacmanEnv:
    def __init__(self):
        self.tiles = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
            0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
            0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0,
            0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
            0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0,
            0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]
        self.width = 20
        self.height = 20
        self.start_index = 272
        self.pacman = self.index_to_coord(self.start_index)

        # 👻 Add static ghost positions
        self.ghosts = [
            (-180, 160),
            (-180, -160),
            (100, 160),
            (100, -160),
        ]

        self.reset()

    def reset(self):
        self.pacman = self.index_to_coord(self.start_index)
        self.collected = set()
        self.visited = set()
        self.last_actions = deque(maxlen=10)
        self.state_visits = defaultdict(int)
        return self.get_state()

    def get_state(self):
        idx = self.coord_to_index(self.pacman)
        x = idx % self.width
        y = idx // self.width

        surroundings = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    tile_index = ny * self.width + nx
                    if 0 <= tile_index < len(self.tiles):
                        surroundings.append(self.tiles[tile_index])
                    else:
                        surroundings.append(0)  # Out-of-bounds fallback
                else:
                    surroundings.append(0)  # Wall by default
        return tuple(surroundings)

    def index_to_coord(self, index):
        x = (index % 20) * 20 - 200
        y = 180 - (index // 20) * 20
        return (x, y)

    def coord_to_index(self, pos):
        x, y = pos
        x_idx = int((x + 200) / 20)
        y_idx = int((180 - y) / 20)
        return y_idx * 20 + x_idx

    def valid(self, pos):
        idx = self.coord_to_index(pos)
        return 0 <= idx < len(self.tiles) and self.tiles[idx] == 1

    def step(self, action):
        x, y = self.pacman
        move = [(20, 0), (-20, 0), (0, 20), (0, -20)][action]
        new_pos = (x + move[0], y + move[1])

        if not self.valid(new_pos):
            return self.get_state(), -1, False

        self.pacman = new_pos
        idx = self.coord_to_index(self.pacman)

        reward = -0.01
        done = False

        # 🎁 Reward or penalize based on whether Pacman has visited this tile before
        if idx not in self.visited:
            self.visited.add(idx)
            reward += 1.0
        else:
            reward -= 0.1

        # 🍒 Reward for collecting a dot (tile == 1 and not collected yet)
        if idx not in self.collected and self.tiles[idx] == 1:
            remaining = self.tiles.count(1) - len(self.collected)
            reward += 10 + (1 / (remaining + 1)) * 30
            self.collected.add(idx)

        # 👹 Add a danger penalty based on ghost proximity
        for gx, gy in self.ghosts:
            dist = np.linalg.norm(np.array([gx, gy]) - np.array([x, y]))
            if dist < 40:
                reward -= (40 - dist) * 0.2  # scaled penalty

        # 🎯 Final goal reward: if all dots are collected, game is won
        if len(self.collected) >= self.tiles.count(1):
            reward += 100
            done = True
            print("🎉 Pacman cleared the maze!")

        return self.get_state(), reward, done

    def get_valid_actions(self):
        actions = []
        for i, (dx, dy) in enumerate([(20, 0), (-20, 0), (0, 20), (0, -20)]):
            new_pos = (self.pacman[0] + dx, self.pacman[1] + dy)
            if self.valid(new_pos):
                actions.append(i)
        return actions
