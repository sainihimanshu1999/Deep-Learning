#learning about loss and cost functions

import numpy as np

y_predicted = np.array([1,1,0,0,1])
y_true = np.array([0.30,0.7,1,0,0.5])

#making mean absolute error function
#running two parallel  for loops on this

def mae(y_true, y_predicted):
    total_error = 0
    for yt, yp in zip(y_true,y_predicted):
        total_error += abs(yt - yp)
    print(total_error)
    mae = total_error/len(y_predicted)
    print(mae)
    return mae

# mae(y_true,y_predicted)

#using numpy to do the same thing
# print(np.mean(np.abs(y_predicted-y_true)))

#Log function

epsilon = 1e-15

y_predicted_new = [max(i,epsilon) for i in y_predicted]
y_predicted_new = [min(i,1-epsilon) for i in y_predicted_new]
y_predicted_new = np.log(y_predicted_new)
x = np.log(y_predicted_new)
print(x)