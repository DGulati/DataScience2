# Description: Homework 1 for CS 6360
import numpy as np
import sys

def read_data(file_name):
    document_id = []
    word_id = []
    count = []
    f = open(file_name, 'r')
    for line in f:
        line_list = line.split()
        document_id.append(int(line_list[0]))
        word_id.append(int(line_list[1]))
        count.append(int(line_list[2]))
    f.close()
    return [document_id, word_id, count]

def read_labels(file_name):
    labels = []
    f = open(file_name, 'r')
    for line in f:
        label.append(int(line))
    return labels

def sigmoid_function(x):
    z = 1/(1 + np.exp(x))
    return z

def Logistic_Regression_train(LR_input):
    #w_0 + \sum_{i=1}^{n} w_i * x_i
    #words are the variables
    #word count is the value of those variables
    #output is label
    words = LR_input[1]
    dim = len(words)+1
    #initialize the weights ( one extra for the intercept )
    weights = np.zeros(dim)
    #initialize the step size
    step_size = 0.0001
    for row in LR_input[0]:
        X_i = row[1]
        Y = row[2]

        #gradient = x_i * [Y - sigmoid(w_0 + \sum_{i=1}^{n} w_i * x_i)]

        X_weights = [0]
        X_weights.append(X_i)

        predict = sigmoid_function(np.sum(np.dot(X_weights,weights)))
        diff = Y - predict

        X_gradient = [1]
        X_gradient.append(X_i)

        gradient = diff * X_gradient
        
        weights = np.add(weights,step_size * gradient)

    return weights

    
def organize_doc_word_count(LR_unorganized_input):
    doc_word_count_dict = {}
    word_list = list(set(LR_unorganized_input[1]))
    for i in range(len(LR_unorganized_input[0])):
        doc_id = LR_unorganized_input[0][i]
        if doc_id not in doc_word_count_dict.keys():
            doc_word_count_dict[doc_id] = [0] * (len(word_list))
        doc_word_count_dict[doc_id][word_list.index(LR_unorganized_input[1][i])] = LR_unorganized_input[2][i]
    return (doc_word_count_dict, word_list)


def organize_doc_label(LR_unorganized_input):
    doc_label_dict = {}
    for i in range(len(LR_unorganized_input[0])):
        doc_id = LR_unorganized_input[0][i]
        doc_label_dict[doc_id] = LR_unorganized_input[3][i]
    return doc_label_dict


def organize_input_data(unorganized_data):
    doc_word_count_dict,word_list = organize_doc_word_count(unorganized_data)
    doc_label_dict = organize_doc_label(unorganized_data)
    LR_input_data = []
    for doc_id in doc_word_count_dict.keys():
        LR_input_data.append([doc_id, doc_word_count_dict[doc_id], doc_label_dict[doc_id]])
    return [LR_input_data, word_list]

def organize_test_data(LR_test,input_words):
    doc_word_count_dict = {}
    for i in range(len(LR_test[0])):
        doc_id = LR_test[0][i]
        if doc_id not in doc_word_count_dict.keys():
            doc_word_count_dict[doc_id] = [0] * (len(input_words))
        doc_word_count_dict[doc_id][input_words.index(LR_test[1][i])] = LR_test[2][i]
    LR_test_data = []
    for doc_id in doc_word_count_dict.keys():
        LR_test_data.append([doc_id,doc_word_count_dict[doc_id]])
    return LR_test_data

def Logistic_Regression_test(LR_test_input, LR_weights, LR_train_words):
    weights = LR_weights
    LR_output = {}

    for doc_id,X_i in LR_test_input:
        X_weights = [0]
        X_weights.append(X_i)

        predict = sigmoid_function(np.sum(np.dot(X_weights,weights)))

        LR_output[doc_id] = predict

    return LR_output




def main():   
    train_data_file_name = sys.argv[1]
    train_label_file_name = sys.argv[2]
    test_file_name = sys.argv[3]

    train_data = read_data(train_data_file_name)
    test_data = read_data(test_file_name)

    LR_data = [train_data[0], train_data[1], train_data[2], read_labels(train_label_file_name)]
    LR_input = organize_input_data(LR_data)
    LR_input_words = LR_input[1]

    LR_weights = Logistic_Regression_train(LR_input)

    LR_test = [test_data[0], test_data[1], test_data[2]]
    LR_test_input = organize_test_data(LR_test,LR_input_words)

    LR_output = Logistic_Regression_test(LR_test_input, LR_weights)

    for doc_id in test_data[0]:
        print(LR_output[doc_id])
    


if __name__ == '__main__':
    main()