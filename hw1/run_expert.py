#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import random

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None, :])
                # print(action)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0:
                    print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
                returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

        # print(expert_data['observations'].shape) # (2000, 111)
        # print(expert_data['actions'].shape) # (2000, 1, 8)

        # Adapted single layer neural network from mnist tutorial:
        # Split into training and validation data
        # create a partition vector
        print("partitioning")
        test_set_size = int(0.2 * len(expert_data["observations"]))
        partitions = [0] * len(expert_data["observations"])
        partitions[:test_set_size] = [1] * test_set_size
        random.shuffle(partitions)

        all_obs = expert_data["observations"]
        all_actions = expert_data["actions"].reshape(
                    len(expert_data["actions"]),
                    len(expert_data["actions"][0][0]))
        '''
        # partition our data into a test and train set
        train_obs, test_obs = tf.dynamic_partition(all_obs, partitions, 2)
        train_actions, test_actions = tf.dynamic_partition(all_actions,
                                                           partitions, 2)

        BATCH_SIZE = 100
        print("creating input queues")
        # create input queues
        train_input_queue = tf.train.slice_input_producer(
            [train_obs, train_actions],
            shuffle=False)
        test_input_queue = tf.train.slice_input_producer(
            [test_obs, test_actions],
            shuffle=False)

        # process path and string tensor into an image and a label
        train_ob = train_input_queue[0]
        train_action = train_input_queue[1]
        print("train_ob: {0}".format(train_ob))
        print("train_action: {0}".format(train_action))
        test_image = test_input_queue[0]
        test_action = test_input_queue[1]

        print("creating batches")
        batch_obs, batch_actions = tf.train.batch([train_ob, train_action],
                                                  batch_size=BATCH_SIZE)
        '''
        # TODO: Implement stochastic gradient descent
        x = tf.placeholder(tf.float32, [None, 111])

        h1 = 300
        h2 = 300
        
        W1 = tf.Variable(tf.truncated_normal([111, h1], stddev=0.1))
        b1 = tf.Variable(tf.constant(0.1, shape=[h1]))  # Initialize weights

        W2 = tf.Variable(tf.truncated_normal([h1, h2], stddev=0.1))
        b2 = tf.Variable(tf.constant(0.1, shape=[h2]))

        W3 = tf.Variable(tf.truncated_normal([h2, 8], stddev=0.1))
        b3 = tf.Variable(tf.constant(0.1, shape=[8]))

        hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
        hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2)
        y = tf.matmul(hidden2, W3) + b3
        y_ = tf.placeholder(tf.float32, [None, 8])

        cross_entropy = tf.reduce_mean(abs(y_ - y))

        train_step = tf.train.GradientDescentOptimizer(1.0) \
                             .minimize(cross_entropy)
        init = tf.global_variables_initializer()

        sess = tf.Session()
        sess.run(init)

        # initialize the queue threads to start to shovel data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(1000):
            sess.run(train_step, feed_dict={x: all_obs, y_: all_actions})

        correct_prediction = tf.reduce_sum(y_ - y, reduction_indices=[1])
        accuracy = tf.reduce_mean(correct_prediction)
        print(sess.run(accuracy,
                       feed_dict={x: all_obs,
                                  y_: all_actions}))

        # Test policy in environment
        print('testing model!')
        returns = []
        observations = []
        actions = []
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = sess.run(y, feed_dict={x: obs[None, :]})
            print(action)
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps % 100 == 0:
                print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
            returns.append(totalr)
        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        # stop our queue threads and properly close the session
        coord.request_stop()
        coord.join(threads)
        sess.close()

if __name__ == '__main__':
    main()
