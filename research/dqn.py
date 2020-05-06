
import sys
sys.path.append('./agents/')
sys.path.append('/home/peterjaq/project/optical-film-maker')

print(sys.path)

from common.DataLoader import MaterialLoader
from common.TransferMatrix import OpticalModeling
from common.Config import FilmConfig
from common.utils.FilmLoss import film_loss

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import q_policy, random_py_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.environments import tf_py_environment
from tf_agents.utils.common import *

#from common.FilmEnvironment import FilmEnvironment 

import tensorflow as tf 
import numpy as np

from common.FilmEnvironment import FilmEnvironment
filmEnv = FilmEnvironment(config_path='Zn.ini', random_init=True, debug=True)

filmEnv = tf_py_environment.TFPyEnvironment(filmEnv)

num_iterations = 20000 # @param {type:"integer"}

initial_collect_steps = 1000  # @param {type:"integer"} 
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}


fc_layer_params = (100,)

q_net = q_network.QNetwork(
    filmEnv.observation_spec(),
    filmEnv.action_spec(),
    fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    filmEnv.time_step_spec(),
    filmEnv.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()

print('Observation Spec:')
print(filmEnv.time_step_spec())
print('Action Spec:')
print(filmEnv.action_spec())

eval_policy = agent.policy
collect_policy = agent.collect_policy

random_policy = random_tf_policy.RandomTFPolicy(filmEnv.time_step_spec(),
                                                filmEnv.action_spec())


replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=filmEnv.batch_size,
    max_length=replay_buffer_max_length)

agent.train = function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

def collect_step(environment, policy, buffer):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
  for _ in range(steps):
    collect_step(env, policy, buffer)

def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=batch_size, 
    num_steps=2).prefetch(3)

iterator = iter(dataset)

avg_return = compute_avg_return(filmEnv, agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

  # Collect a few steps using collect_policy and save to the replay buffer.
  # for _ in range(collect_steps_per_iteration):
  #   collect_step(train_env, agent.collect_policy, replay_buffer)
  collect_data(filmEnv, agent.collect_policy, replay_buffer, 100)

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience).loss

  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss))

  # if step % eval_interval == 0:
  #   avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
  #   print('step = {0}: Average Return = {1}'.format(step, avg_return))
  #   returns.append(avg_return)                                                                   





