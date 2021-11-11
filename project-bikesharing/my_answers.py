## AVTsoof:
## References:
## https://towardsdatascience.com/https-medium-com-piotr-skalski92-deep-dive-into-deep-networks-math-17660bc376ba
## https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
## https://dustinstansbury.github.io/theclevermachine/derivation-backpropagation
        

import numpy as np

# activation functions
def activation_sigmoid(Z):
    return 1/(1 + np.exp(-Z))

def activation_sigmoid_backward(Z):
    return activation_sigmoid(Z) * (1 - activation_sigmoid(Z))

def activation_in_to_out(Z):
    return Z

def activation_in_to_out_backward(Z):
    return 1

# define nn
class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate


    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(X)
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass_single_layer(self, L_prev, W_curr, activation_function=activation_sigmoid):
        '''
            Arguments
            ---------
            L_prev: result from previous layer
            W_curr: weights of current layer
            activation_function: the function to apply to output

        '''
        return activation_function(np.dot(L_prev, W_curr))
        
        
    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        # Hidden layer
        hidden_inputs = X
        hidden_outputs = self.forward_pass_single_layer(hidden_inputs, self.weights_input_to_hidden, activation_sigmoid)

        # Output layer
        final_inputs = hidden_outputs
        final_outputs = self.forward_pass_single_layer(final_inputs, self.weights_hidden_to_output, activation_in_to_out)
        
        return final_outputs, hidden_outputs

    def backpropagation_single_layer(self, dY, W, Z, X, backward_activation_function=activation_sigmoid_backward):
        # X: input to layer
        # W: weights of layer
        # Z: output of layer (before activation)
        # Y: output of layer (after activation)
        
        # for next backprop
        dY_dZ = dY * backward_activation_funcltion(Z)
        dZ_dX = W
        dY_dX = np.dot(dZ_dX, dY_dZ.T)
        
        # for optimizer step
        dZ_dW = X
        dY_dW = np.dot(dY_dZ, dZ_dW.T)
        
        return dY_dX, dY_dW
        
    
    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''      
        
        # Output error
        # Output layer error is the difference between desired target and actual output.
        error = y - final_outputs
        
        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = None
        
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        output_error_term = None
        
        hidden_error_term = None
        
        # TODO: Add Weight step (input to hidden) and Weight step (hidden to output).
        # Weight step (input to hidden)
        dY = 1
        W = self.weights_input_to_hidden
        Z = np.dot(X, W)
        Y = activation_sigmoid(Z)
        dY, dW = self.backpropagation_single_layer(dY, W, Z, X, activation_sigmoid_backward)
        delta_weights_i_h += dW
        
        
        W = self.weights_hidden_to_output
        Z = np.dot(Y, W)
        dY, dW = self.backpropagation_single_layer(dY, W, Z, X, activation_in_to_out_backward)
        delta_weights_h_o += dW
        
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        
        # Update the weights with gradient descent step
        # update hidden-to-output weights with gradient descent step
        self.weights_hidden_to_output -= delta_weights_h_o * self.lr
        # update input-to-hidden weights with gradient descent step
        self.weights_input_to_hidden -= delta_weights_i_h * self.lr

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # Hidden layer
        hidden_inputs = features
        hidden_outputs = self.forward_pass_single_layer(hidden_inputs, self.weights_input_to_hidden, activation_sigmoid)

        # Output layer
        final_inputs = hidden_outputs
        final_outputs = self.forward_pass_single_layer(final_inputs, self.weights_hidden_to_output, activation_in_to_out)
        
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 100
learning_rate = 0.003
hidden_nodes = 2
output_nodes = 1
