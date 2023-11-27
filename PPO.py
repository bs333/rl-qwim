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
        """
        Builds the actor network which is responsible for the policy function.
    
        The actor network takes in the current state and outputs a probability distribution
        over the possible actions. It is trained to select actions that maximize expected rewards.

        Returns:
            Model: A compiled actor model that predicts action probabilities.
        """

        # Define the input layer for the state space, which forms the basis for the actor to make decisions.
        # This will be the input received from the environment.
        state_input = Input(shape=(self.state_dim,))

        # Define a second input for advantages, which will be used to scale the gradient during training.
        # The advantage function helps in determining the relative value of each action compared to a baseline.
        advantages = Input(shape=(1,))

        # Define a third input for the old policy's prediction. This is used for the ratio in PPO's clipped objective.
        # It ensures that the updated policy does not deviate too much from the old policy.
        old_prediction = Input(shape=(self.action_dim,))

        # Add a fully connected layer with 64 units and ReLU activation. It serves as the first hidden layer that processes the input state.
        x = Dense(64, activation='relu')(state_input)

        # Add another fully connected layer with 64 units and ReLU activation. This is the second hidden layer that further processes the information from the first layer.
        x = Dense(64, activation='relu')(x)

        # The output layer provides a probability distribution over all possible actions using the softmax activation.
        # This layer dictates the likelihood of taking each possible action in the given state.
        action_probs = Dense(self.action_dim, activation='softmax')(x)

        # Construct the actor model with state and advantage inputs and action probabilities as outputs.
        # This model will be trained with a custom loss function that utilizes the advantages and old predictions.
        model = tf.keras.models.Model(inputs=[state_input, advantages, old_prediction], outputs=action_probs)

        return model

    def build_critic(self) -> Model:
        """
        Builds the critic network which estimates the value function.
        
        The critic network assesses the value of being in a given state, which is critical for the actor's
        policy improvement. It helps in evaluating how good the state is by predicting the expected sum of rewards,
        also known as the state's value.

        Returns:
            Model: A compiled critic model that predicts state value.
        """

        # Define the input layer for the state space, which receives the state of the environment.
        # The critic uses this to evaluate the potential return from this state.
        state_input = Input(shape=(self.state_dim,), name='state_input')

        # The first hidden layer takes the state as input and processes it through 64 units with ReLU activation.
        # ReLU is used to add non-linearity to the model, allowing it to learn more complex patterns.
        x = Dense(64, activation='relu', name='critic_hidden_layer_1')(state_input)

        # The second hidden layer continues processing the data from the first hidden layer.
        # Another layer of 64 units with ReLU activation is used here for further transformation.
        x = Dense(64, activation='relu', name='critic_hidden_layer_2')(x)

        # The output layer provides a single value as output with no activation function.
        # This value represents the critic's estimate of the value function for the input state.
        values = Dense(1, name='state_value')(x)

        # Construct the critic model with state input and the value output.
        # The critic will be trained to minimize the difference between predicted values and actual returns,
        # thus learning to accurately estimate the value function.
        model = tf.keras.models.Model(inputs=state_input, outputs=values, name='critic_model')

        return model

    def policy_loss(advantages: tf.Tensor, old_probs: tf.Tensor, actions: tf.Tensor, new_probs: tf.Tensor, clip_ratio: float) -> tf.Tensor:
        """
        Computes the PPO policy loss.

        Args:
            advantages (tf.Tensor): The advantage estimates for the actions taken.
            old_probs (tf.Tensor): The action probabilities under the old policy.
            actions (tf.Tensor): The actions taken.
            new_probs (tf.Tensor): The action probabilities under the current policy.
            clip_ratio (float): The clipping parameter epsilon, used to define the bounds of the clipping.

        Returns:
            tf.Tensor: The computed policy loss.
        """

        # Convert actions to a one-hot encoding.
        actions_one_hot = tf.one_hot(actions, depth=new_probs.shape[-1])

        # Calculate the probability ratios.
        probs = tf.reduce_sum(new_probs * actions_one_hot, axis=-1)
        old_probs = tf.reduce_sum(old_probs * actions_one_hot, axis=-1)
        ratio = tf.exp(tf.math.log(probs) - tf.math.log(old_probs))

        # Evaluted the clipped function.
        clipped_ratio = tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        clipped_surrogate = clipped_ratio * advantages

        # Evaluate the unclipped function.
        unclipped_surrogate = ratio * advantages

        # Take the minimum of the clipped and unclipped surrogate functions to form the final objective function.
        loss = -tf.reduce_mean(tf.minimum(unclipped_surrogate, clipped_surrogate))

        return loss

    def value_loss(predicted_values: tf.Tensor, target_values: tf.Tensor) -> tf.Tensor:
        """
        Computes the value loss for the critic network.

        Args:
            predicted_values (tf.Tensor): The values predicted by the critic for the current states.
            target_values (tf.Tensor): The discounted sum of rewards (return) or the target value.

        Returns:
            tf.Tensor: The computed value loss.
        """
        pass

    def train(self, states: tf.Tensor, actions: tf.Tensor, rewards: tf.Tensor, next_states: tf.Tensor, dones: tf.Tensor):
        """Implements the training loop for the PPO agent.

        Args:
            states (tf.Tensor): Batch of states.
            actions (tf.Tensor): Batch of actions.
            rewards (tf.Tensor): Batch of rewards.
            next_states (tf.Tensor): Batch of next states.
            dones (tf.Tensor): Batch of done flags indicating the end of an episode.
        """
        pass



