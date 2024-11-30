import numpy as np
import matplotlib.pyplot as plt
import pickle


class NeuralNetwork:
    def __init__(self, layers, activations=['sigmoid'], learning_rate=0.1, optimizer='adam', beta1=0.9, beta2=0.999,
                 epsilon=1e-8):
        if len(layers) < 2:
            raise ValueError("The network must have at least two layers (input and output).")
        if len(activations) not in [1, len(layers) - 1]:
            raise ValueError("Activations list must have either one or len(layers) - 1 elements.")

        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        self.activations = []
        self.activation_derivatives = []
        self.optimizer = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iterations = 0

        for i in range(len(layers) - 1):
            if len(activations) == 1:
                activation = activations[0]
            else:
                activation = activations[i]
            if activation == 'relu':
                limit = np.sqrt(2 / layers[i])
            else:
                limit = np.sqrt(6 / (layers[i] + layers[i + 1]))
            weight = np.random.uniform(-limit, limit, (layers[i], layers[i + 1]))
            self.weights.append(weight)
            self.biases.append(np.zeros((1, layers[i + 1])))

            if activation == 'sigmoid':
                self.activations.append(self.sigmoid)
                self.activation_derivatives.append(self.sigmoid_derivative)
            elif activation == 'relu':
                self.activations.append(self.relu)
                self.activation_derivatives.append(self.relu_derivative)
            elif activation == 'tanh':
                self.activations.append(self.tanh)
                self.activation_derivatives.append(self.tanh_derivative)
            else:
                raise ValueError(f"Unsupported activation function: {activation}")

        if self.optimizer == 'adam':
            self.m_weights = [np.zeros_like(w) for w in self.weights]
            self.v_weights = [np.zeros_like(w) for w in self.weights]
            self.m_biases = [np.zeros_like(b) for b in self.biases]
            self.v_biases = [np.zeros_like(b) for b in self.biases]

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(a):
        return a * (1 - a)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(a):
        return (a > 0).astype(float)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(a):
        return 1 - a ** 2

    def forward(self, X):
        activations = [X]
        input_layer = X
        for idx, (w, b, activation_func) in enumerate(zip(self.weights, self.biases, self.activations)):
            z = np.dot(input_layer, w) + b
            a = activation_func(z)
            activations.append(a)
            input_layer = a
        return activations

    def compute_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def backward(self, activations, y_true):
        weight_grads = [np.zeros_like(w) for w in self.weights]
        bias_grads = [np.zeros_like(b) for b in self.biases]

        y_pred = activations[-1]
        error = y_pred - y_true
        delta = error

        for i in reversed(range(len(self.weights))):
            a_prev = activations[i]
            weight_grads[i] = np.dot(a_prev.T, delta)
            bias_grads[i] = np.sum(delta, axis=0, keepdims=True)
            if i != 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivatives[i - 1](activations[i])
        return weight_grads, bias_grads

    def update_parameters(self, weight_grads, bias_grads):
        if self.optimizer == 'adam':
            self.iterations += 1
            for i in range(len(self.weights)):
                self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * weight_grads[i]
                self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * bias_grads[i]

                self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (weight_grads[i] ** 2)
                self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * (bias_grads[i] ** 2)

                m_weights_corr = self.m_weights[i] / (1 - self.beta1 ** self.iterations)
                m_biases_corr = self.m_biases[i] / (1 - self.beta1 ** self.iterations)

                v_weights_corr = self.v_weights[i] / (1 - self.beta2 ** self.iterations)
                v_biases_corr = self.v_biases[i] / (1 - self.beta2 ** self.iterations)

                self.weights[i] -= self.learning_rate * m_weights_corr / (np.sqrt(v_weights_corr) + self.epsilon)
                self.biases[i] -= self.learning_rate * m_biases_corr / (np.sqrt(v_biases_corr) + self.epsilon)
        else:
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * weight_grads[i]
                self.biases[i] -= self.learning_rate * bias_grads[i]

    def train(self, X, y, epochs=10000, print_loss_every=1000, early_stopping=False, patience=1000):
        loss_history = []
        best_loss = float('inf')
        patience_counter = 0
        for epoch in range(1, epochs + 1):
            activations = self.forward(X)
            y_pred = activations[-1]
            loss = self.compute_loss(y, y_pred)
            loss_history.append(loss)
            weight_grads, bias_grads = self.backward(activations, y)
            self.update_parameters(weight_grads, bias_grads)

            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
            if early_stopping and patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}. Loss did not improve for {patience} epochs.")
                break

            if epoch % print_loss_every == 0 or epoch == 1:
                print(f"Epoch {epoch}/{epochs} - Loss: {loss:.6f}")
        return loss_history

    def predict(self, X):
        activations = self.forward(X)
        return activations[-1]

    def evaluate(self, X, y, threshold=0.5):
        y_pred = self.predict(X)
        predictions = (y_pred > threshold).astype(int)
        accuracy = np.mean(predictions == y) * 100
        print(f"Accuracy: {accuracy:.2f}%")
        return accuracy

    def save_weights(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'weights': self.weights, 'biases': self.biases}, f)
        print(f"Weights and biases saved to {filepath}.")

    def load_weights(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.weights = data['weights']
            self.biases = data['biases']
        print(f"Weights and biases loaded from {filepath}.")


def generate_xor_data():
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    return X, y


def plot_loss(loss_history, loss_type='BCE Loss'):
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel(loss_type)
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    np.random.seed(42)

    X, y = generate_xor_data()

    layers = [2, 4, 1]
    activations = ['relu', 'sigmoid']
    nn = NeuralNetwork(layers=layers, activations=activations, learning_rate=0.1, optimizer='adam')

    epochs = 10000
    loss_history = nn.train(X, y, epochs=epochs, print_loss_every=1000, early_stopping=True, patience=1000)

    plot_loss(loss_history, loss_type='BCE Loss')

    nn.evaluate(X, y)

    print("Predictions on XOR inputs:")
    predictions = nn.predict(X)
    print(predictions)

    nn.save_weights('trained_model.pkl')


if __name__ == "__main__":
    main()
