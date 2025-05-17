   
    
class TrainingStats:
    def __init__(self, mean_freq: int = 1000):
        self.best_reward = -float('inf')
        self.ep_num = 0
        self.mean_freq = mean_freq

        self.ep_rewards = []
        self.mean_ep_rewards = []
        self.last_mean_ep_reward = None

        self.ep_lengths = []
        self.mean_ep_lengths = []
        self.last_mean_ep_length = None

        self.q_vals = []
        self.losses = []
        self.mean_loss = None

    def update_episode(self, reward, length):
        self.ep_num += 1
        self.ep_rewards.append(reward)
        self.ep_lengths.append(length)

        if reward > self.best_reward:
            self.best_reward = reward

        if self.ep_num % self.mean_freq == 0:
            self.last_mean_ep_reward = sum(self.ep_rewards[-self.mean_freq:]) / self.mean_freq
            self.mean_ep_rewards.append(self.last_mean_ep_reward)

            self.last_mean_ep_length = sum(self.ep_lengths[-self.mean_freq:]) / self.mean_freq
            self.mean_ep_lengths.append(self.last_mean_ep_length)

            self.mean_loss = sum(self.losses[-self.mean_freq:]) / self.mean_freq

    def add_loss(self, loss):
        self.losses.append(loss)

    def add_q_val(self, q_val):
        self.q_vals.append(q_val)

