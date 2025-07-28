import numpy
import matplotlib.pyplot
# scipy.special for the sigmoid function expit()
import scipy.special


# neural network class definition
class neuralNetwork: # initialise the neural network
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.innodes=inputnodes 
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        #learning rate
        self.lr=learningrate 
        #activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        self.inverse_activation_function = lambda x: scipy.special.logit(x)
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer # w11 w21
        # w12 w22 etc
        self.wih = (numpy.random.rand(self.hnodes, self.innodes) - 0.5) 
        self.who =(numpy.random.rand(self.onodes, self.hnodes) - 0.5)



    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # error is ( target - actual)
        output_errors = targets - final_outputs

        # hidden layer errors are the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        pass
    # query the neural network
    def query(self,inputs_list):
        #convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        #calculate signals into hidden layer
        hidden_inputs =numpy.dot(self.wih,inputs)
        #calculate the signals emerging from hidden layer
        hidden_outputs =self.activation_function(hidden_inputs)
        #calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #calculate the signals emergin from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs




# of hidden ,input and output nodes
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

# learning rate is 0.3
learning_rate = 0.1

#create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load the mnist training data CSV file into a list
training_data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network
# go through all records in the training data set

#number of times to train the neural network
epochs = 7
for epoch in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        # scale and shift the inputs
        inputs=(numpy.asarray(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01

        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)


image_array = numpy.asarray(all_values[1:], dtype=float).reshape((28, 28))
matplotlib.pyplot.imshow(image_array, cmap="Greys", interpolation='None')

# load the mnist test data CSV file into a list
test_data_file = open("mnist_dataset/mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()


#Test the neural network
#scorecard shows how well the network performs, initially empty
scorecard = []

#loop through all records in the training data set
for record in test_data_list:
    all_values = record.split(',')
    #correct label is the first value
    correct_label = int(all_values[0])
    #scale and shift the inputs
    inputs = (numpy.asarray(all_values[1:], dtype=float) /255.0 *0.99) + 0.01
    #query the neural network
    outputs = n.query(inputs)
    #index of highest values corresponds to the label
    label = numpy.argmax(outputs)
    #append correct or incorrect to scorecard
    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)


#calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
performance = scorecard_array.sum() / scorecard_array.size
print("Performance = ", performance)






