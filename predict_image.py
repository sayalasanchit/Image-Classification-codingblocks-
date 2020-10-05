import sys
import cv2
import joblib
import numpy as np

labels_dict={'cat': 0,
            'dog': 1,
            'horse': 2,
            'human': 3}
labels_list=['cat', 'dog', 'horse', 'human']
classes_no=len(labels_list)

# Loading model
svm_classifiers=joblib.load('model.pkl')

def binaryPredict(x, w, b):
    z=np.dot(x, w.T)+b
    if z>=0:
        return 1
    else:
        return -1

def predict(x):
    count=np.zeros((classes_no, ))
    for i in range(classes_no):
        for j in range(i+1, classes_no):
            w, b=svm_classifiers[i][j]
            z=binaryPredict(x, w, b)
            if z==-1:
                count[i]+=1
            elif z==1:
                count[j]+=1
    final_prediction=np.argmax(count)
    return final_prediction

img_path=sys.argv[1]
# try:
img=cv2.imread(img_path)
img=cv2.resize(img, (50, 50))
image_data=np.asarray(img, dtype='float32')/255.0
image_data=image_data.reshape((1, -1)) # Flattening
prediction=predict(image_data)
print(labels_list[prediction])

# except:
# 	print('Some error occured')
