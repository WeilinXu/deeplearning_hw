from solver import *
from logistic import *
from svm import *
import pickle

with open('data.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

data[1][:][data[1][:] == 0] = -1
data = {
    "X_train": data[0][:500],
    "y_train": data[1][:500],
    "X_val": data[0][500:750],
    "y_val": data[1][500:750],
    "X_test": data[0][750:],
    "y_test": data[1][750:]
}


input_len = len(data['X_train'][0])

# Problem 3.3
# model = SVM(input_dim=input_len, reg=0.0)
# solver = Solver(model, data,
                # update_rule='sgd',
                # optim_config={
                  # 'learning_rate': 0.1,
                # },
                # lr_decay=0.95,
                # num_epochs=10, batch_size=1,
                # print_every=20)

# Problem 3.4
model = SVM(input_dim=input_len, hidden_dim=20, reg=0.0)
solver = Solver(model, data,
                update_rule='sgd',
                optim_config={
                  'learning_rate': 0.001,
                },
                lr_decay=0.95,
                num_epochs=10, batch_size=1,
                print_every=20)

# Problem 2.3
# model = LogisticClassifier(input_dim=input_len, reg=0.00)
# solver = Solver(model, data,
                # update_rule='sgd',
                # optim_config={
                  # 'learning_rate': 0.15,
                # },
                # lr_decay=0.95,
                # num_epochs=10, batch_size=1,
                # print_every=20)

# Problem 2.4
# model = LogisticClassifier(input_dim=input_len, hidden_dim=10, reg=0.00)
# solver = Solver(model, data,
                # update_rule='sgd',
                # optim_config={
                  # 'learning_rate': 0.1,
                # },
                # lr_decay=0.95,
                # num_epochs=10, batch_size=1,
                # print_every=20)

solver.train()
