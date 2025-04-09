from typing import Any, Optional
import numpy as np
import gymnasium as gym

from ...hppo.utility import seperate_action_from_numpy

def rvs(dim=3):
     random_state = np.random
     H = np.eye(dim)
     D = np.ones((dim,))
     for n in range(1, dim):
         x = random_state.normal(size=(dim-n+1,))
         D[n-1] = np.sign(x[0])
         x[0] -= D[n-1]*np.sqrt((x*x).sum())
         # Householder transformation
         Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
         mat = np.eye(dim)
         mat[n-1:, n-1:] = Hx
         H = np.dot(H, mat)
         # Fix the last sign such that the determinant is 1
     D[-1] = (-1)**(1-(dim % 2))*D.prod()
     # Equivalent to np.dot(np.diag(D), H) but faster, apparently
     H = (D*H.T).T
     return H

class ZeroControl(gym.Env):

    def __init__(self, size: int = 2, bound: float = 5, speed: float = 1, timeout: int = 20, tol: float = 1, is_param_action: bool = True, use_approach_reward: bool = True):

        self.size = size
        self.bound = bound
        self.timeout = timeout
        self.max_dis = float(np.linalg.norm(np.ones(size) * bound))
        self.max_move = speed
        self.tol = tol

        self.direct = rvs(size)

        self.observation_space = gym.spaces.Box(
            -bound, bound, (self.size,), np.float32
        )
        if is_param_action:
            self.action_space = gym.spaces.Tuple(
                (gym.spaces.Discrete(self.size + 1), gym.spaces.Box(-speed, speed, (1,), np.float32))
            )
        else:
            disp_bounds = np.hstack((np.ones(size + 1), speed * np.ones(1)))
            self.action_space = gym.spaces.Box(
                -disp_bounds, disp_bounds, disp_bounds.shape, np.float32
            )
        self.is_param_action = is_param_action
        self.cur_vec = np.zeros((self.size,))
        self.use_approach_reward = use_approach_reward

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.cur_vec = (np.random.random((self.size,)) * 2 - 1) * self.bound
        self.use_time = 0
        self.last_dist = float(np.linalg.norm(self.cur_vec))
        self.direct = rvs(self.size)

        return self.cur_vec, {}
    
    def step(self, action):
        self.use_time += 1

        info = {}
        is_success = False
        try_break = False

        if self.is_param_action:
            discrete_action, continue_action = seperate_action_from_numpy(action)
        else:
            discrete_action = np.argmax(action[:self.size + 1])
            continue_action = action[self.size + 1]

        if discrete_action == 0:
            try_break = True
            if float(np.linalg.norm(self.cur_vec)) < self.tol:
                is_success = True
            else:
                is_success = False
            
            info["is_success"] = is_success
        else:
            self.cur_vec += continue_action * self.direct[discrete_action - 1]

        self.cur_vec = np.clip(self.cur_vec, -self.bound, self.bound)
        cur_dis = float(np.linalg.norm(self.cur_vec))

        terminated = False
        truncated = try_break or self.use_time == self.timeout

        if try_break:
            if is_success:
                reward = 10
            else:
                reward = -10
        elif self.use_time == self.timeout:
            reward = -10
        else:
            if self.use_approach_reward:
                reward = (self.last_dist - cur_dis) / self.max_move - 1
            else:
                reward = (self.max_dis - cur_dis) / self.max_dis - 1
            # reward = (self.max_dis - float(np.linalg.norm(self.cur_vec))) / self.max_dis - 1

        self.last_dist = cur_dis
        return self.cur_vec, reward, terminated, truncated, info