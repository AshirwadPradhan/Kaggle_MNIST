import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


# importing the dataset
train_dataset = pd.read_csv('data/train.csv')
X_train = train_dataset.iloc[:, 1:].values
y_train = train_dataset.iloc[:, 0].values

#print(X_train.shape)
#print(y_train.shape)

# building the classifier
classifier = KNeighborsClassifier(metric='minkowski', p=2)
classifier.fit(X_train, y_train)

# importing the test data
test_dataset = pd.read_csv('data/test.csv')
X_test = test_dataset.iloc[:, :].values
#print(X_test.shape)

# predicting the test data
y_pred = classifier.predict(X_test)
print('done')

#print(y_pred)

# getting the submission file
with open('submission.csv', 'w') as f:
    f.write('ImageId,Label\n')
    for i, pred in enumerate(y_pred):
        f.write(str(i+1)+','+str(pred)+'\n')
