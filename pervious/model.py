import optical_model_env
import deepqnetwork


def run_maze():
    step = 0
    for episode in range(300):
        # initial observation
        observation, mean_abs = env.init_Device()

        print("The init Device is: %s abs:%f" % (observation, mean_abs))

        while True:

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done, mean_abs = env.run_simulate(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

            print('%d step, the final observation:%s, abs:%f, reward:%f' % (step, observation, mean_abs, reward))

    # end of game

    #env.destroy()


if __name__ == "__main__":
    # maze game
    env = optical_model_env.optical_film_env(config_path="conf/config.conf")

    RL = deepqnetwork.DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True
                      )
    run_maze()
    RL.plot_cost()
