## Reinforcement Learning in Quantitative Wealth Investment Management (QWIM)

#### Overview of Reinforcement Learning (RL)

Reinforcement learning $\mathbb{RL}$ considers the problem of the automatic learning of optimal decisions over time. It differs from other types of learning (such as unsupervized or supervised) since the learning follows from feedback and experience (and not from some fixed training sample of data).

$\mathbb{RL}$ is learning what to do (how to map situations to actions), to maximize a numerical reward signal. The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them. Actions may affect not only the immediate reward but also the next situation and, through that, all subsequent rewards. These two characteristics (trial-and-error search and delayed reward) are the two most important distinguishing features of reinforcement learning.


Thus $\mathbb{RL}$ algorithms describe how an agent can learn an optimal action policy in a sequential decision process, through repeated experience, with primary purpose of finding an optimal policy (a mapping from the states of the world to the set of actions), in order to maximize cumulative reward (a long term strategy), although exploring might be sub-optimal on a short-term horizon.

In a given environment, the agent policy provides some intermediate and terminal rewards, while the agent learns sequentially. When an agent picks an action, she can not infer ex-post the rewards induced by other action choices. Agent’s actions have consequences, influencing not only rewards, but also future states of the world.

#### Project

The following project seeks to implement, compare, and contrast various reinforcement learning algorithms for quantitative wealth investment management in context to portfolio construction for a diversified portfolio with various asset classes.