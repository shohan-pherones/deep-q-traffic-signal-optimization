from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQNModel:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def predict(self, state):
        return self.model.predict(state)

    def fit(self, state, target, epochs=1, verbose=0):
        self.model.fit(state, target, epochs=epochs, verbose=verbose)