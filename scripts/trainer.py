# marl/trainer.py
# This file contains the implementation of the MARL training loop.

from marl_tsc.environment.tsc_env import TrafficSignalControlEnv
from marl_tsc.advanced_techniques.techniques import apply_advanced_techniques
from marl_tsc.utils.logger_setup import setup_logger
from marl_tsc.utils.monitoring_params import init_monitoring
from marl_tsc.utils.other_utils import set_random_seeds
from marl_tsc.algorithms.algorithm_factory import create_algorithm
from marl_tsc.utils.config_loader import load_config


class Trainer:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(config)
        set_random_seeds(config['training']['seed'])
        self.setup_marl_model()

    def setup_marl_model(self):
        self.env = TrafficSignalControlEnv(self.config, logger=self.logger)

        if self.config['training']['register_env']:
            self.env = self.env.register()

        if self.config['training']['monitor']:
            self.monitor = init_monitoring(self.config)

        self.algorithm = create_algorithm(self.config, self.env, self.logger)
        print("Algorithm: ", self.algorithm)
        self.agents = self.algorithm.agents
        # apply_advanced_techniques(self.env, self.agents, self.config['advanced_techniques'])


    def train(self):
        num_episodes = self.config['training']['num_episodes']
        for episode in range(num_episodes):
            print("Episode: ", episode)
            observations = self.env.reset()
            print("Observations: ", observations)
            done = False
            episode_rewards = []
            step = 0
            while not done:
                trajectories, observations, done = self.algorithm.collect_rollouts(observations)
                # trajectories, observations, done = self.algorithm.collect_rollouts_random_actions(observations)
                # print("Trajectories: ", trajectories)
                # print("Observations: ", observations)
                print("Observations values: ", observations.values(), end="\n")
                # print("Done: ", done)
                if len(trajectories) == 0:
                    print(f"Episode {episode} Step {step}: No trajectories. Ending episode.")
                    break
                loss = self.algorithm.update(trajectories)
                episode_rewards.extend([sum(t['rewards'].values()) for t in trajectories])
                # print("Episode: ", episode, "Reward: ", sum(episode_rewards))
                step += 1
                if step%20 == 0:
                    print("Step: ", step)
                    print("Loss: ", loss)
                    print("Episode: ", episode, "Reward: ", sum(episode_rewards))

            total_reward = sum(episode_rewards)
            if episode % self.config['logging']['log_interval'] == 0:
                self.logger.info(f"Episode: {episode}, Total Reward: {total_reward}")

        # Save model after training
        self.algorithm.save_model(self.config['training']['model_path'])


    def evaluate(self, num_episodes=10):
        total_rewards = []
        for episode in range(num_episodes):
            observations = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                actions = {}
                for agent_id, agent in self.agents.items():
                    obs = observations[agent_id]
                    if agent.obs_rms:
                        obs = (obs - agent.obs_rms.mean) / (agent.obs_rms.var + 1e-8)
                    action, _ = agent.select_action(obs, evaluate=True)
                    actions[agent_id] = action

                observations, rewards, done, _ = self.env.step(actions)
                episode_reward += sum(rewards.values())

            total_rewards.append(episode_reward)

        average_reward = sum(total_rewards) / num_episodes
        self.logger.info(f"Evaluation over {num_episodes} episodes: Average Reward: {average_reward}")
        return average_reward



def main():
    config_path = 'marl_tsc/config/main_config.yaml'
    config = load_config(config_path)
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()


