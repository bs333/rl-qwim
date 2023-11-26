import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

class PPO:
    """Proximal Policy Optimization (PPO) agent with separate actor and critic networks.
    
    Attributes:
        state_dim (int): Dimensionality of the state space.
        action_dim (int): Dimensionality of the action space.
        actor_lr (float): Learning rate for the actor model.
        critic_lr (float): Learning rate for the critic model.
        clip_ratio (float): PPO clipping parameter.
        actor (Model): Actor network for policy approximation.
        critic (Model): Critic network for value approximation.
        actor_optimizer (Adam): Optimizer for the actor network.
        critic_optimizer (Adam): Optimizer for the critic network.
    """
    
    def __init__(self, state_dim: int, action_dim: int, actor_lr: float, critic_lr: float, clip_ratio: float):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_ratio = clip_ratio

        # Build the actor and critic networks.
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        
        # Define optimizers for both networks.
        self.actor_optimizer = Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = Adam(learning_rate=self.critic_lr)

    def build_actor(self) -> Model:
        """Builds the actor network.

        Returns:
            Model: A compiled actor model.
        """

        state_input = Input(shape=(self.state_dim,))
        advantages = Input(shape=(1,))
        old_prediction = Input(shape=(self.action_dim,))

        x = Dense(64, activation='relu')(state_input)
        x = Dense(64, activation='relu')(x)
        action_probs = Dense(self.action_dim, activation='softmax')(x)

        model = tf.keras.models.Model(inputs=[state_input, advantages, old_prediction], outputs=action_probs)
        return model

    def build_critic(self) -> Model:
        """Builds the critic network.

        Returns:
            Model: A compiled critic model.
        """

        state_input = Input(shape=(self.state_dim,))
        x = Dense(64, activation='relu')(state_input)
        x = Dense(64, activation='relu')(x)
        values = Dense(1)(x)

        model = tf.keras.models.Model(inputs=state_input, outputs=values)
        return model

    def policy_loss(self, advantages: tf.Tensor, old_prediction: tf.Tensor, action_probs: tf.Tensor) -> tf.Tensor:        
        """Defines the policy loss function to be used for training the actor.

        Args:
            advantages (tf.Tensor): The advantages for each action.
            old_prediction (tf.Tensor): The action probabilities from the old policy.
            action_probs (tf.Tensor): The action probabilities from the current policy.

        Returns:
            tf.Tensor: The computed policy loss.
        """
        pass

    def value_loss(self, rewards: tf.Tensor, values: tf.Tensor) -> tf.Tensor:
        """Defines the value loss function to be used for training the critic.

        Args:
            rewards (tf.Tensor): The rewards obtained from the environment.
            values (tf.Tensor): The predicted values from the critic network.

        Returns:
            tf.Tensor: The computed value loss.
        """
        pass

    def train(self, states, actions, rewards, next_states, dones):
        # Convert data into tensors
        # Compute advantages
        # Train actor and critic models using mini-batch updates
        pass



