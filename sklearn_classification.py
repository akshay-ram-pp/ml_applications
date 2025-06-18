################################################################
#    DATASET: IRIS DATASET                                     #
#        contains 150 samples of flowers across 3 species      #
#                                                              #
#    FEATURES:                                                 #
#        Sepal length, sepal width, petal width, petal length  #
#                                                              #
################################################################


####################################### STEPS ###################################
# 1. Load dataset
# 2. Split into training and test sets
# 3. Train a model(Random Forest)
# 4. Evaluate performance with classification report and confusion matrix
#################################################################################


from sklearn.datasets import load_iris # load iris dataset
from sklearn. model_selection import train_test_split # splits the dataset into training and testing sets
from sklearn.ensemble import RandomForestClassifier #a decision tree based classifier
from sklearn.metrics import classification_report, confusion_matrix  # evaluation metrics
# data vizualisation
import seaborn as sns
import matplotlib.pyplot as plt

## load dataset
iris = load_iris()
x,y = iris.data, iris.target
# iris.data contains features like sepal/petal measurements. shape(150,4)-  150 samples, each with 4 features
#iris.target contains the species of flower. array of integers (0,1,2), one for each class


## splits the dataset
x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
# splits the dataset for 80% training and 20% testing
# random_state=42 ensures reproducibility of split

## train the model
model = RandomForestClassifier(n_estimators=100) # an ensemble of 100 decision trees.
model.fit(x_train,y_train)  # trains the model on training data
# after the above step model learns the patterns to classify

## prediction
preds = model.predict(x_test) 
#  use the trained model to predict the labels for the test set


## model evaluation
print(classification_report(y_test,preds,target_names=iris.target_names))

## plot confusion matrix
sns.heatmap(confusion_matrix(y_test,preds), annot=True)
plt.title("IRIS Classification- Confusion Matrix")
plt.show()


## confusion matrix
# True classes on Y-axis
# Predicted classes on X-axis
# if model performs well most values will lie on the diagonal


#################### result - example ###############################
#                  precision   recall  f1-score   support

#       setosa       1.00      1.00      1.00        10
#   versicolor       1.00      1.00      1.00         9
#    virginica       1.00      1.00      1.00        11

#     accuracy                           1.00        30
#    macro avg       1.00      1.00      1.00        30
# weighted avg       1.00      1.00      1.00        30