import numpy
import matplotlib.pyplot
# scipy.special for the sigmoid function expit()
import scipy.special
import glob
import imageio.v3


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
learning_rate = 0.2 

#create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load the mnist training data CSV file into a list
training_data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network
# go through all records in the training data set

#number of times to train the neural network
epochs = 6
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
    print("Epoch: ", epoch)

#my set of images
my_data_set = []

#load my images as a dataset
for image_file_name in glob.glob("my_dataset/image_?.png"):

    #set label (the character before .png)
    label = int(image_file_name[-5:-4])

    #load image data from png to array
    print("Loading image: ", image_file_name)
    img_array = imageio.v3.imread(image_file_name, mode= 'F')
    #reshape fro 28*28 to 784
    img_array =255.0 - img_array.reshape((784))

    #scale data to range between 0.01 and 1.0
    img_data = (img_array / 255.0 * 0.99) + 0.01
    print(numpy.min(img_data))
    print(numpy.max(img_data))

    record = numpy.append(label,img_data)
    my_data_set.append(record)
    

#test the neural network with my dataset

#record to test
item = 1
print("Dataset length: ", len(my_data_set))
#plot image
matplotlib.pyplot.imshow(my_data_set[item][1:].reshape((28, 28)), cmap="Greys", interpolation='None')
# matplotlib.pyplot.show()
#correct label
correct_label = int(my_data_set[item][0])

#the data are the remaining values
inputs = my_data_set[item][1:]

#query the neural network
outputs = n.query(inputs)
print("Outputs: ", outputs)

label = numpy.argmax(outputs)
print("Network's label: ", label)
      
#give correct or incorrect answer
if label == correct_label:
    print("Network's answer is correct")
else:
    print("Network's answer is incorrect")













