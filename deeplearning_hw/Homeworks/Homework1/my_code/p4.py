import pickle
import numpy as np
from softmax import *
from cnn import *
from solver import *

def get_data(data_set):
	data = np.array(data_set[0])
	label = np.array(data_set[1])
	return data, label

f = open('mnist.pkl', 'rb')
train_set, valid_set, test_set = pickle.load(f,encoding='latin1')
train_data, train_label = get_data(train_set)
valid_data, valid_label = get_data(valid_set)
test_data, test_label = get_data(test_set)
f.close()
# print(valid_data.shape)
data = {
        'X_train': train_data[0:1000],# training data
        'y_train': train_label[0:1000],# training labels
        'X_val': valid_data[0:250],# validation data
        'y_val': valid_label[0:250]# validation labels
    }

model1 = ConvNet(input_dim=(1,28,28), filter_size = 7, num_classes=10, weight_scale=1e-3, reg=0.0,  addbatchdrop=False)
solver = Solver(model1, data,
              update_rule='sgd_momentum',
              optim_config={
                'learning_rate': 0.1,
                'momentum': 0.1
              },
              lr_decay=0.95,
              num_epochs=10, batch_size=20,
              print_every=20)	

# 4.3
# model = SoftmaxClassifier(input_dim=28*28, num_classes=10,
              # weight_scale=1e-3, reg=0.0)
# solver = Solver(model, data,
              # update_rule='sgd_momentum',
              # optim_config={
                # 'learning_rate': 1,
                # 'momentum': 0.1
              # },
              # lr_decay=0.95,
              # num_epochs=10, batch_size=20,
              # print_every=20)

# 4.4
# model = SoftmaxClassifier(input_dim=28*28, hidden_dim=15*15, num_classes=10,
              # weight_scale=1e-3, reg=0.0)
# solver = Solver(model, data,
              # update_rule='sgd_momentum',
              # optim_config={
                # 'learning_rate': 1,
                # 'momentum': 0.1
              # },
              # lr_decay=0.95,
              # num_epochs=10, batch_size=20,
              # print_every=20)
solver.train()
