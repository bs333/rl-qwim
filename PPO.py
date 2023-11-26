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

    def build_actor(self):
        state_input = Input(shape=(self.state_dim,))
        advantages = Input(shape=(1,))
        old_prediction = Input(shape=(self.action_dim,))

        x = Dense(64, activation='relu')(state_input)
        x = Dense(64, activation='relu')(x)
        action_probs = Dense(self.action_dim, activation='softmax')(x)

        model = tf.keras.models.Model(inputs=[state_input, advantages, old_prediction], outputs=action_probs)
        return model


