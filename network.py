from copy import deepcopy
from os import error
import numpy as np

classes = ["alpha", "beta", "gamma", "delta", "epsilon"]

# Klasa za pojedinačni neuron
class Neuron:
    def __init__(self, inp_dim) -> None:
        self.weights = np.random.randn(inp_dim, 1)
        self.bias = np.random.rand()

    def output(self, x):
        y = np.matmul(self.weights.T, x) + self.bias
        return y.item()     # Samo nam treba broj, a ne array pošto je riječ o samo 1 broju.

    # def last_layer_error(self, sample, true_val):
    #     y = self.output(sample)
    #     self.error = y * (1 - y) * (true_val - y)
    #     return self.error

    # def hidden_layer_error(self, sample, ):
    #     y = self.output(sample)
    #     self.error = y * (1 - y) * np.sum(next_error * )
    #     return self.error
    
    # def update_weights(self, prev_y, l_r=0.001):
    #     self.weights += l_r * np.sum(self.error * prev_y)

# Klasa za skup neurona u sloju
class Layer:
    def __init__(self, inp_dim, num_neurons) -> None:
        self.neurons = [Neuron(inp_dim) for i in range(num_neurons)]
        self.num_neurons = num_neurons
        self.deltas = np.zeros((num_neurons, 1))
        
    def reset_deltas(self):
        self.deltas = np.zeros((self.num_neurons, 1))
        
    def forward(self, x):
        output = list()
        for n in self.neurons:
            output.append( n.output(x) )

        return np.array(output)
    
    def get_weights(self):
        weights = list()
        for neuron in self.neurons:
            weights_reshaped = np.reshape(neuron.weights, (neuron.weights.shape[0],))
            weights.append( weights_reshaped )

        return np.array(weights)
    
    def update_weights(self, update):
        
        for i, neuron in enumerate(self.neurons):
            # print(np.reshape(update[i], (update[i].shape[0], 1)) )
            # exit(1)
            # print(neuron.weights.shape)
            neuron.weights -= np.reshape(update[i], (update[i].shape[0], 1))
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

# Klasa za cijelu neuronsku mrežu
class Network:
    def __init__(self, inp_dim, arh) -> None:
        # self.weights = list()
        self.layers = list()
        arh += "5"  # 5 je za izlazni sloj koji nam vraća vektor
        arh = arh.split('s')
        self.arhitecture = [int(l) for l in arh]    # Arhitektura mreže da bi se težine mogle staviti na određene neurone kod treniranja.
        self.layers.append( Layer(inp_dim=int(inp_dim), num_neurons=int(arh[0])) )
        for lay_size in arh[1:]:    # Ide do izuzev ZADNJEG jer split vrati prazan element na kraju
            self.layers.append( Layer(inp_dim=self.layers[-1].num_neurons, num_neurons=int(lay_size)) )

    def forward_propagation(self, x):
        z = x.copy()
        for l in range(len(self.layers) - 1):   # Ide se do zadnjeg jer nakon zadnjeg nema sigmoide
            z = self.layers[l].forward(z)
            h = self.layers[l].sigmoid(z)

        y = self.layers[-1].forward(h)

        return y

    def train(self, X, Y, l_r=0.001, epochs=1000, alg=2):
                
        loss = list()
        reference_mse = self.mse_loss(X, Y)

        for epoch in range(epochs):
            errors = list()
            
            for x, t in zip(X, Y):
                all_layer_outputs = list()
                z = x.copy()
                for l in range(len(self.layers) - 1):
                    z = self.layers[l].forward(z)
                    h = self.layers[l].sigmoid(z)
                    all_layer_outputs.append(h)
                    
                all_layer_outputs.append(self.layers[-1].forward(h))
                
                # Output layer error
                y = all_layer_outputs[-1]
                err_k = y * (1 - y) * (t - y)
                err_k = np.reshape(err_k, (err_k.shape[0], 1))
                self.layers[-1].deltas += err_k

                for l in reversed(range(len(self.layers)-1)):
                    y_k = all_layer_outputs[l]
                    y_k = np.reshape(y_k, (1, y_k.shape[0]))
                    
                    err_k = y_k * (1 - y_k) * np.matmul(err_k.T, self.layers[l+1].get_weights())
                    err_k = np.reshape(err_k, (err_k.shape[1], 1))
                    self.layers[l].deltas += err_k
                    
                if alg == 2:    # Stohastic grad desc
                    all_layer_outputs.insert(0, x)
                    for l in range(1, len(self.layers)):
                        y_k_ = np.reshape(all_layer_outputs[l-1], (1, all_layer_outputs[l-1].shape[0]))
                        update = l_r * np.matmul(self.layers[l-1].deltas, y_k_)
                        self.layers[l-1].update_weights(update)
                        self.layers[l-1].reset_deltas()
                        
            if alg == 1:    # Goes through a whole epoch, then it updates the weights
                # for l in reversed(range(1, len(self.layers))):
                all_layer_outputs.insert(0, x)
                for l in range(1, len(self.layers)):
                    y_k_ = np.reshape(all_layer_outputs[l-1], (1, all_layer_outputs[l-1].shape[0]))
                    update = l_r * np.matmul(self.layers[l-1].deltas, y_k_)
                    self.layers[l-1].update_weights(update)
                    self.layers[l-1].reset_deltas()

            mse = self.mse_loss(X, Y)
            print(f"Epoch {epoch+1}   MSE: {mse}")

            if epoch % 50 == 0:
                loss.append(mse)
            if 5 * mse < reference_mse:
                print("Learning rate halved.")
                l_r *= 0.5
                reference_mse = mse

    def train_minibatch(self, X, Y, l_r=0.001, epochs=1000, alg=3):
                
        loss = list()
        reference_mse = self.mse_loss(X, Y)
        if alg == 3:
            X_ = deepcopy(X)
            Y_ = deepcopy(Y)

        for epoch in range(epochs):
            errors = list()
            
            # Pošto ja imam 60 uzoraka po svakom razredu (300 uk), moja minigrupa će imati 6*5=30 uzoraka
            if alg == 3:
                X_grande = deepcopy(X_)
                Y_grande = deepcopy(Y_)    # Ne želim mijenjati cijeli algoritam pa improviziram ovdje, isto tako i s X-om.
                last = 0          # Varijabla koja pamti gdje je gornji generator malih grupa stao.
                X_minibatch = list()
                Y_minibatch = list()

                X_minibatches = list()
                Y_minibatches = list()

                for j in range(0, 60, 6):
                    for i in range(6):
                        X_minibatch.append(X_grande[i + j])                # Alpha
                        X_minibatch.append(X_grande[i + 60  + j - 1])      # Beta
                        X_minibatch.append(X_grande[i + 120 + j - 1])      # Gamma
                        X_minibatch.append(X_grande[i + 180 + j - 1])      # Delta
                        X_minibatch.append(X_grande[i + 240 + j - 1])      # Epsilon

                        Y_minibatch.append(Y_grande[i + j])                # Alpha
                        Y_minibatch.append(Y_grande[i + 60  + j - 1])      # Beta
                        Y_minibatch.append(Y_grande[i + 120 + j - 1])      # Gamma
                        Y_minibatch.append(Y_grande[i + 180 + j - 1])      # Delta
                        Y_minibatch.append(Y_grande[i + 240 + j - 1])      # Epsilon
                    
                    X_minibatches.append(deepcopy(X_minibatch))
                    Y_minibatches.append(deepcopy(Y_minibatch))
                    X_minibatch.clear()
                    Y_minibatch.clear()
                    last += 1
                    
            for X, Y in zip(X_minibatches, Y_minibatches):
                for x, t in zip(X, Y):
                    all_layer_outputs = list()
                    z = x.copy()
                    for l in range(len(self.layers) - 1):
                        z = self.layers[l].forward(z)
                        h = self.layers[l].sigmoid(z)
                        all_layer_outputs.append(h)
                        
                    all_layer_outputs.append(self.layers[-1].forward(h))
                    
                    # Output layer error
                    y = all_layer_outputs[-1]
                    err_k = y * (1 - y) * (t - y)
                    err_k = np.reshape(err_k, (err_k.shape[0], 1))
                    self.layers[-1].deltas += err_k

                    for l in reversed(range(len(self.layers)-1)):
                        y_k = all_layer_outputs[l]
                        y_k = np.reshape(y_k, (1, y_k.shape[0]))
                        
                        err_k = y_k * (1 - y_k) * np.matmul(err_k.T, self.layers[l+1].get_weights())
                        err_k = np.reshape(err_k, (err_k.shape[1], 1))
                        self.layers[l].deltas += err_k
                        

                all_layer_outputs.insert(0, x)
                for l in range(1, len(self.layers)):
                    y_k_ = np.reshape(all_layer_outputs[l-1], (1, all_layer_outputs[l-1].shape[0]))
                    update = l_r * np.matmul(self.layers[l-1].deltas, y_k_)
                    self.layers[l-1].update_weights(update)
                    self.layers[l-1].reset_deltas()
                
            mse = self.mse_loss(X, Y)
            print(f"Epoch {epoch+1}   MSE: {mse}")

            if epoch % 50 == 0:
                loss.append(mse)
            if 5 * mse < reference_mse:
                print("Learning rate halved.")
                l_r *= 0.5
                reference_mse = mse
                
                
    def predict(self, x):
        return classes[np.argmax(self.forward_propagation(x))]
    
    def mse_loss(self, X, Y):
        mse = 0

        for x, y in zip(X, Y):
            mse += np.sum(( y - self.forward_propagation(x) ) ** 2)
        
        return mse / (2*len(Y))
      
if __name__ == "__main__":
    # Za skrivene slojeve unosi 20x20 pa neka onda mreža na kraju bude 2*Mx20x20x5 jer je ulazni i izlazni sloj predefiniran.
    from preprocessing import preprocess_symbol_vector
    from random import randint
    import pickle

    # vec = [(randint(0, 800), randint(0, 800)) for i in range(150)]
    # inp = preprocess_symbol_vector(vec) 
    # with open("test_vec_nn.pickle", "wb") as file:
    #     pickle.dump(inp, file)

    with open("extended_prepared_dataset.pickle", "rb") as file:
        ds = pickle.load(file)

    X = [el[0] for el in ds]
    Y = [el[1] for el in ds]
    M = X[0].shape[0]
    # print(X[0], Y[0])
    # print(M)

    arh = "20s20s"
    net = Network(M, arh)
    
    # net.train(X, Y)
    # net.train_minibatch(X, Y)