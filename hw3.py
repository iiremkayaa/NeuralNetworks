from preprocess import read_data, create_samples, split_data
import numpy as np
import os
import random
import math
from sympy import Symbol
# HYPERPARAMETERS
input_size = 50 #size of each word vector
output_size = 2 #number of classes
hidden_layer_size = 100
learning_rate = 0.1
number_of_epochs = 1000
path = "./data" #use relative path like this

def match_words(): #inex-kelime- 50lik vector
    wordsAndVectors = []
    with open(os.path.join(path, "vocab.txt"), "r") as input_file:
        wordIndex=0
        for line in input_file:
            wordsAndVectors.append([wordIndex,line.strip(),[]])
            wordIndex+=1
    with open(os.path.join(path, "wordVectors.txt"), "r") as input_file:
        wordIndex=0
        for line in input_file:
            wordsAndVectors[wordIndex][2]=line.strip().split()
            wordIndex+=1
    return wordsAndVectors
def createWeightForHidden():
    weightArray=[]
    for i in range(150):
        a=[]
        for j in range(hidden_layer_size):
            a.append(random.random())
        weightArray.append(a)
    return np.array(weightArray)
def createBiasForHidden():
    biasArray=[]
    for i in range(hidden_layer_size):
        biasArray.append(0)
    return np.array(biasArray)
def createWeightForOutput():
    weightArray=[]
    for i in range(hidden_layer_size):
        a=[]
        for j in range(output_size):            
            a.append(random.random())
        weightArray.append(a)
    return np.array(weightArray)
def createBiasForOutput():
    biasArray=[]
    for i in range(output_size):
        biasArray.append(0)
    return np.array(biasArray)
"""
//hidden için ve outptut ayrı ayrı bias ve weight
//hidden layerın weighti 150 ye 100 150ye hidden layer size yani 100
//hidden weigth: 150x 100
//hidden bias: 1x100 ilk sıfır
1 150 x 150 100 çarp +1 100
output weight: 100x2 random yine maxa böl
//output bias 1x 2 yani 1 e output size  ilk sıfır
"""
def nonlinearity(result):
    return np.tanh(result)


def activation_function(layer):
    return 1.0/(1.0 + np.exp(-layer))

def derivation_of_activation_function(signal):
    return activation_function(signal)*(1-activation_function(signal))

def loss_function(true_labels, probabilities):
    """sum_square_error = 0.0
    for i in range(len(true_labels)):
        sum_square_error += (true_labels[i] - probabilities[i])**2.0
    mean_square_error = 1.0 / len(true_labels) * sum_square_error
    return mean_square_error"""
    return (-( (true_labels * np.log(probabilities) ) + ( (1-true_labels) * np.log(1-probabilities)  ) )) 



# the derivation should be with respect to the output neurons
def derivation_of_loss_function(true_labels, probabilities):
    x = Symbol(probabilities)
    deriv = loss_function(true_labels, probabilities).diff(x)
    return deriv
    #return probabilities - true_labels.reshape(-1,1)


# softmax is used to turn activations into probability distribution
def softmax(layer):
    return np.exp(layer-layer.max())/np.exp(layer-layer.max()).sum()

def forward_pass(data):
    word_vector=embedding_layer(data)
    x_1=np.add(np.dot(word_vector,createWeightForHidden()),createBiasForHidden())
    hidden=activation_function(x_1)
    x_2=np.add(np.dot(hidden,createWeightForOutput()),createBiasForOutput())
    output=nonlinearity(x_2)
    return hidden,output
"""  
//emdeding layerda yap yukardakini forwardda çağır
//embdedding çağır 150 lik vektore çevir datayı
//variabke oluştur x1 matrix çarpımı yap embdeddingle hiddenin weightii çarp ona ata biasi ekle
//döneni activatine sok hiddena eşitle bu hidden alyer sonucu
//bi tane olutşru variabke hidden layerle output matrixinin çarp output biasiyle topla ona eşitle
//en son döneni nonlinearity fun a sok outputa eşitle
// return output ve hidden döndürdü"""

#should change the strings into word vectors. Should not be effected by the backpropagation
def embedding_layer(samples):
    word_vector=[]
    for words in match_words():
        for dataWord in samples:
            if dataWord == words[1]:
                for vector in words[2]:
                    word_vector.append(float(vector))
    if(len(word_vector) <150):
        for i in range(len(word_vector),150):
            word_vector.append(random.random())
    return np.array(word_vector)



# [hidden_layers] is not an argument, replace it with your desired hidden layers 
#def backward_pass(input_layer, [hidden_layers] , output_layer, loss): 
#    pass



def train(train_data, train_labels, valid_data, valid_labels):

    for epoch in range(number_of_epochs):
        index = 0

        #for each batch
        for data, labels in zip(train_data, train_labels):
            # Same thing about [hidden_layers] mentioned above is valid here also
            predictions, [hidden_layers] = forward_pass(data)
            print(predictions)
            #loss_signals = derivation_of_loss_function(labels, predictions)
            #backward_pass(data, [hidden_layers], predictions, loss_signals)
            #loss = loss_function(labels, predictions)

            #if index%20000 == 0: # at each 20000th sample, we run validation set to see our model's improvements
             #   accuracy, loss = test(valid_data, valid_labels)
              #  print("Epoch= "+str(epoch)+", Coverage= %"+ str(100*(index/len(train_data))) + ", Accuracy= "+ str(accuracy) + ", Loss= " + str(loss))

            #index += 1

    #return losses





def test(test_data, test_labels):

    avg_loss = 0
    predictions = []
    labels = []

    #for each batch
    for data, label in zip(test_data, test_labels):
        prediction, _, _ = forward_pass(data)
        predictions.append(prediction)
        labels.append(label)
        avg_loss += np.sum(loss_function(label, prediction))

    #turn predictions into one-hot encoded 
    one_hot_predictions = np.zeros(shape=(len(predictions), output_size))
    for i in range(len(predictions)):
        one_hot_predictions[i][np.argmax(predictions[i])] = 1

    predictions = one_hot_predictions
    accuracy_score = accuracy(labels, predictions)

    return accuracy_score,  avg_loss / len(test_data)




def accuracy(true_labels, predictions):
    true_pred = 0

    for i in range(len(predictions)):
        if np.argmax(predictions[i]) == np.argmax(true_labels[i]): # if 1 is in same index with ground truth
            true_pred += 1

    return true_pred / len(predictions)







 
if __name__ == "__main__":


	#PROCESS THE DATA
	words, labels = read_data(path)
	sentences = create_samples(words, labels)
	train_x, train_y, test_x, test_y = split_data(sentences)


	# creating one-hot vector notation of labels. (Labels are given numeric)
	# [0 1] is PERSON
	# [1 0] is not PERSON
	new_train_y = np.zeros(shape=(len(train_y), output_size))
	new_test_y = np.zeros(shape=(len(test_y), output_size))

	for i in range(len(train_y)):
		new_train_y[i][int(train_y[i])] = 1

	for i in range(len(test_y)):
		new_test_y[i][int(test_y[i])] = 1

	train_y = new_train_y
	test_y = new_test_y


	# Training and validation split. (%80-%20)
	valid_x = np.asarray(train_x[int(0.8*len(train_x)):-1])
	valid_y = np.asarray(train_y[int(0.8*len(train_y)):-1])
	train_x = np.asarray(train_x[0:int(0.8*len(train_x))])
	train_y = np.asarray(train_y[0:int(0.8*len(train_y))])

	train(train_x, train_y, valid_x, valid_y)
	print("Test Scores:")
	print(test(test_x, test_y))
