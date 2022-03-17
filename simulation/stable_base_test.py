from sim_utils import *
import gym
print(tf.config.list_physical_devices('GPU'))

class DQNAgent():
    def __init__(self, env, epsilon_decay):
        self.replay_memory_size = 2000
        self.discount = 0.95
        self.min_replay_memory_size = 1000
        self.minibatch_size = 64
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.001
        self.action_space = env.action_space
        self.replay_memory = deque(maxlen=self.replay_memory_size)
        self.epsilon = 1
        self.model = self.create_model()

    def create_model(self):

        model = Sequential()
        model.add(Dense(512, activation='relu', input_dim=4))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(env.action_space.n, activation='linear'))

        model.compile(loss='mse', optimizer=adam_v2.Adam(learning_rate=0.001), metrics=['accuracy'])

        return model

    def update_replay_memory(self, transition):
        # current_state, action, reward, new_state, done
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def act(self, current_state):
        if np.random.random() > self.epsilon:
            action = np.argmax(self.get_qs(current_state))
        else:
            action = np.random.randint(0, self.action_space.n)

        # Decay Epsilon
        if len(self.replay_memory) > self.min_replay_memory_size:
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.min_epsilon, self.epsilon)
        return action

    def train(self, terminal_state):
        if len(self.replay_memory) < self.min_replay_memory_size:
            return

        minibatch = random.sample(self.replay_memory, self.minibatch_size)

        ## current states = [dict{img, vec}, dict{img, vec}, dict{img, vec}......]
        ## seperating dicts into arrays
        current_states = np.array([transition[0] for transition in minibatch])  # Add divide by max (scale results)
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])  # Add divide by max (scale results)
        future_qs_list = self.model.predict(new_current_states)

        x = []
        y = []  # array of new q values

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.discount * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            x.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(x), np.array(y), batch_size=self.minibatch_size, verbose=0, shuffle=False)

    def render(self):
        pass



env = gym.make('CartPole-v0')
env.reset()

reward_arr = []
episodes = 1000
agent = DQNAgent(env, epsilon_decay=0.999)
for episode in range(episodes):
    reward_tot = 0
    done = False
    current_state = env.reset()
    step = 1
    while not done:
        # env.render()
        action = agent.act(current_state)
        new_state, reward, done, info = env.step(action)
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        reward_tot += reward
        agent.train(done)
        current_state = new_state
        step += 1

    reward_arr.append(reward_tot)
    print('episode: {}, reward: {}, epsilon: {}, steps: {}'.
    format(episode, reward_tot, agent.epsilon, step, ))

x = np.linspace(1, episodes, episodes)
plt.scatter(x, reward_arr)
plt.xlabel('episode')
plt.ylabel('reward')
plt.show()
