import numpy as np
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

    def policy_loss(self, advantages: tf.Tensor, old_probs: tf.Tensor, actions: tf.Tensor, new_probs: tf.Tensor, clip_ratio: float) -> tf.Tensor:
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

    def value_loss(self, predicted_values: tf.Tensor, target_values: tf.Tensor) -> tf.Tensor:
        """
        Computes the value loss for the critic network.

        Args:
            predicted_values (tf.Tensor): The values predicted by the critic for the current states.
            target_values (tf.Tensor): The discounted sum of rewards (return) or the target value.

        Returns:
            tf.Tensor: The computed value loss.
        """
        # Use MSE as loss function for the critic.
        loss = tf.reduce_mean((predicted_values - target_values) ** 2)

        return loss

    def train(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray):
        """
        Implements the training loop for the PPO agent.

        Args:
            states (np.ndarray): Batch of states.
            actions (np.ndarray): Batch of actions taken.
            rewards (np.ndarray): Batch of rewards received.
            next_states (np.ndarray): Batch of next states.
            dones (np.ndarray): Batch of done flags (boolean) indicating the end of an episode.
        """

        # Convert numpy arrays to TensorFlow tensors.
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Calculate the value of current and next states.
        values = self.critic(states)
        next_values = self.critic(next_states)

        # Calculate discounted rewards and advantages.
        target_values = self.calculate_discounted_rewards(rewards, dones)
        advantages = self.calculate_advantages(rewards, values, next_values, dones)

        # Update the actor and critic networks.
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            # Calculate current policy probabilities.
            current_probs = self.actor([states, advantages, self.actor(states)])

            # Calculate policy loss.
            p_loss = self.policy_loss(advantages, self.actor(states), actions, current_probs, self.clip_ratio)

            # Recompute critic values and calculate value loss.
            values = self.critic(states)
            v_loss = self.value_loss(values, target_values)

        # Compute gradients and update actor network.
        actor_grads = actor_tape.gradient(p_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Compute gradients and update critic network.
        critic_grads = critic_tape.gradient(v_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

    def calculate_discounted_rewards(self, rewards: tf.Tensor, dones: tf.Tensor, gamma: float = 0.99) -> tf.Tensor:
        """
        Calculate discounted rewards, also known as the return.

        The return at each time step is the sum of future rewards discounted by the gamma factor.
        The rewards are processed in reverse order (from end of the episode to the beginning).

        Args:
            rewards (tf.Tensor): Tensor of rewards for each time step.
            dones (tf.Tensor): Tensor indicating whether an episode has ended at each time step.
            gamma (float): Discount factor, typically between 0 and 1.

        Returns:
            tf.Tensor: Tensor of discounted rewards for each time step.
        """
        # Initialize a tensor array to store the discounted rewards, which will be dynamically sized.
        discounted_rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        reward_sum = 0.0  # Variable to store the accumulated reward

        # Loop through the rewards array in reverse order to calculate discounted rewards.
        for t in reversed(range(len(rewards))):
            # If the episode ended, the reward_sum is reset to 0. Otherwise, it accumulates discounted rewards.
            reward_sum = rewards[t] + gamma * reward_sum * (1 - dones[t])

            # Write the calculated reward_sum to the corresponding position in the tensor array.
            discounted_rewards = discounted_rewards.write(t, reward_sum)

        # Convert the tensor array to a regular tensor before returning.
        return discounted_rewards.stack()

    def calculate_advantages(self, rewards: tf.Tensor, values: tf.Tensor, next_values: tf.Tensor, dones: tf.Tensor, gamma: float = 0.99, lambda_: float = 0.95) -> tf.Tensor:
        """
        Calculate advantages using Generalized Advantage Estimation (GAE).

        Advantages are calculated as the difference between the return and the value estimates of the critic.
        GAE helps in reducing the variance of the advantage estimates.

        Args:
            rewards (tf.Tensor): Tensor of rewards for each time step.
            values (tf.Tensor): Tensor of value estimates for the current states.
            next_values (tf.Tensor): Tensor of value estimates for the next states.
            dones (tf.Tensor): Tensor indicating whether an episode has ended at each time step.
            gamma (float): Discount factor for rewards.
            lambda_ (float): Smoothing parameter for GAE.

        Returns:
            tf.Tensor: Tensor of calculated advantages for each time step.
        """

        # Calculate the temporal difference error (delta) for each time step.
        deltas = rewards + gamma * next_values * (1 - dones) - values
        advantages = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        advantage_sum = 0.0  # Variable to store the accumulated advantage

        # Loop through the deltas array in reverse order to calculate the advantages.
        for t in reversed(range(len(rewards))):
            # The advantage at each time step is a discounted sum of future TD errors.
            advantage_sum = deltas[t] + gamma * lambda_ * advantage_sum * (1 - dones[t])

            # Write the calculated advantage to the corresponding position in the tensor array.
            advantages = advantages.write(t, advantage_sum)

        # Convert the tensor array to a regular tensor before returning.
        return advantages.stack()

    def select_action(self, state: np.ndarray) -> int:
        """
        Selects an action based on the current state using the actor model.

        Args:
            state (np.ndarray): The current state.

        Returns:
            int: The selected action.
        """
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action_probs = self.actor.predict(state)  # Use predict instead of direct call
        action = np.random.choice(self.action_dim, p=np.squeeze(action_probs))

        return action

    def save_models(self, actor_path: str, critic_path: str):
        """
        Saves the actor and critic models to the specified paths.
        """

        self.actor.save(actor_path)
        self.critic.save(critic_path)

    def load_models(self, actor_path: str, critic_path: str):
        """
        Loads the actor and critic models from the specified paths.
        """
        self.actor = tf.keras.models.load_model(actor_path)
        self.critic = tf.keras.models.load_model(critic_path)

# # Example usage outside the class:
# ppo_agent = PPO(state_dim, action_dim, actor_lr, critic_lr, clip_ratio)

# # Example training loop:
# for episode in range(total_episodes):
#     state = env.reset()
#     done = False
#     while not done:
#         action = ppo_agent.select_action(state)
#         next_state, reward, done, _ = env.step(action)
#         # Store state, action, reward, next_state, done
#         state = next_state

#         if ready_to_update:
#             ppo_agent.train(states, actions, rewards, next_states, dones)
#             # Reset or update your storage variables

#     if episode % save_interval == 0:
#         ppo_agent.save_models(actor_path, critic_path)

#     # Additional logging or monitoring as needed
