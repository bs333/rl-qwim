import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

class PPO:
    def __init__(self, state_dim, action_dim, actor_lr, critic_lr, clip_ratio):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_ratio = clip_ratio

        self.actor = self.build_actor()
        self.critic = self.build_critic()
        
        self.actor_optimizer = Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = Adam(learning_rate=self.critic_lr)

