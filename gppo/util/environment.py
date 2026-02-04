from skimage.transform import resize
import numpy as np
import gymnasium as gym


class CatchEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.size = 21
        self.image_shape = (self.size,) * 2
        self.image = np.zeros(self.image_shape)
        self.state = []
        self.fps = 4
        self.output_shape = (self.size * self.fps,) * 2
        # Typically the 3rd dim indicates the nr. of color channels.
        # Instead, we use it to stack the previous frames.
        self.obs_shape = self.output_shape + (self.fps,)

        # gym.Env attributes
        self.reward_range = (0, 1)
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.obs_shape, dtype=np.uint8
        )

    def reset_random(self):
        self.image.fill(0)
        self.pos = self.np_random.integers(2, self.size - 2)
        self.vx = self.np_random.integers(5) - 2
        self.vy = 1
        self.ballx, self.bally = self.np_random.integers(self.size), 4
        self.image[self.bally, self.ballx] = 1
        self.image[-5, self.pos - 2 : self.pos + 3] = np.ones(5)
        return self.step(2)[0]

    def reset(self, **kwargs):
        super().reset(**kwargs)
        return self.reset_random(), {}

    def step(self, action):
        def left():
            if self.pos > 3:
                self.pos -= 2

        def right():
            if self.pos < 17:
                self.pos += 2

        def noop():
            pass

        {0: left, 1: right, 2: noop}[action]()

        self.image[self.bally, self.ballx] = 0
        self.ballx += self.vx
        self.bally += self.vy
        if self.ballx > self.size - 1:
            self.ballx -= 2 * (self.ballx - (self.size - 1))
            self.vx *= -1
        elif self.ballx < 0:
            self.ballx += 2 * (0 - self.ballx)
            self.vx *= -1
        self.image[self.bally, self.ballx] = 1

        self.image[-5].fill(0)
        self.image[-5, self.pos - 2 : self.pos + 3] = np.ones(5)

        terminal = self.bally == self.size - 1 - 4
        reward = int(self.pos - 2 <= self.ballx <= self.pos + 2) if terminal else 0

        [
            self.state.append(resize(self.image, self.output_shape) * 255)
            for _ in range(self.fps - len(self.state) + 1)
        ]
        self.state = self.state[-self.fps :]

        return (
            np.transpose(self.state, [1, 2, 0]),
            reward,
            terminal,  # terminated
            terminal,  # trucated
            {},  # empty info
        )

