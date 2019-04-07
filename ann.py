import numpy.random as r
import matplotlib.pylab as plt
import numpy as np

class ANN:
    def __init__(self, layers, eta = 0.5):
        self.nn_structure = layers
        self.eta = eta
        self.W = {}
        self.b = {}
        self.avg_cost_func = []
        self.init_weights(layers)

    def init_weights(self, nn_structure):
        for l in range(1,len(nn_structure)):
            self.W[l] = r.random_sample((nn_structure[l], nn_structure[l-1]))
            self.b[l] = r.random_sample((nn_structure[l],))

    def set_weights(self, weights):
        self.W = weights

    def set_bias(self, bias):
        self.b = bias

    def f(self, x):
        return 1 / (1 + np.exp(-x))

    def f_deriv(self, x):
        return self.f(x) * (1 - self.f(x))

    def feed_forward(self, x):
        h = {1:x}
        z = {}
        for l in range(1,len(self.W)+1):
            if l == 1:
                node_in = x
            else:
                node_in = h[l]
            z[l+1] = self.W[l].dot(node_in) + self.b[l]
            h[l+1] =  self.f(z[l+1])
        return h, z

    def calculate_out_layer_delta(self, y,h_out,z_out):
        return -(y-h_out)*self.f_deriv(z_out)


    def calculate_hidden_delta(self, delta_plus_1,w_l,z_l):
        return np.dot(np.transpose(w_l),delta_plus_1) * self.f_deriv(z_l)

    def init_tri_values(self, nn_structure):
        tri_W = {}
        tri_b = {}
        for l in range(1,len(nn_structure)):
            tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
            tri_b[l] = np.zeros((nn_structure[l],))
        return tri_W, tri_b

    def fit(self, X, y, iter_num= 120000):
        cnt = 0
        m = len(y)
        while cnt < iter_num:
            tri_W, tri_b = self.init_tri_values(self.nn_structure)
            avg_cost = 0
            for i in range(len(y)):
                delta = {}
                h, z = self.feed_forward(X[i, :])
                for l in range(len(self.nn_structure), 0, -1):
                    if l == len(self.nn_structure):
                        delta[l] = self.calculate_out_layer_delta(y[i, :], h[l], z[l])
                        avg_cost += np.linalg.norm((y[i,:]-h[l]))
                    else:
                        if l > 1:
                            delta[l] = self.calculate_hidden_delta(delta[l+1], self.W[l], z[l])
                        tri_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(h[l][:,np.newaxis]))
                        tri_b[l] += delta[l+1]
            for l in range(len(self.nn_structure) - 1, 0, -1):
                self.W[l] += -self.eta * (1.0/m * tri_W[l])
                self.b[l] += -self.eta * (1.0/m * tri_b[l])
            avg_cost = 1.0/m * avg_cost
            self.avg_cost_func.append(avg_cost)
            cnt += 1

    def predict(self, X, n_layers):
        m = X.shape[0]
        y = np.zeros(x.shape)
        for i in range(m):
            h, z = self.feed_forward(X[i, :])
            y[i] = h[n_layers]
        return h[n_layers]

    def graph_avg_cost(self):
        plt.plot(self.avg_cost_func)
        plt.ylabel('Average J')
        plt.xlabel('Iteration number')
        plt.show()
 

w1 = np.array([[0.15, 0.2], [0.25, 0.3]])
w2 = np.array([[0.4, 0.45], [0.5, 0.55]])
b1 = np.array([0.35, 0.35])
b2 = np.array([0.6, 0.6])
w = {1:w1, 2:w2}
b = {1:b1, 2:b2}

x = np.array([[0.05, 0.1]])
y = np.array([[0.01, 0.99]])

model = ANN([2,2,2], 0.5)
model.set_weights(w)
model.set_bias(b)
print('model.predict')
print(model.predict(x,3))
print('model.fit')
model.fit(x,y)
print('model.predict')
y_predict = model.predict(x,3)
print('y predict: {}, {}'.format(y_predict[0], y_predict[1]))
model.graph_avg_cost