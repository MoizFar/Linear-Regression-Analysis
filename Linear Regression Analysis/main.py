import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

breast_cancer = datasets.load_breast_cancer()

breast_cancer_X = breast_cancer.data

breast_cancer_X_train = breast_cancer_X[:-3]
breast_cancer_X_test = breast_cancer_X[-3:]

breast_cancer_y_train = breast_cancer.target[:-3]
breast_cancer_y_test = breast_cancer.target[-3:]

model = linear_model.LinearRegression()

model.fit(breast_cancer_X_train, breast_cancer_y_train)

breast_cancer_y_predicted = model.predict(breast_cancer_X_test)

print("Mean squared error is: ", mean_squared_error(breast_cancer_y_test, breast_cancer_y_predicted))

print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)

plt.scatter(breast_cancer_X_test, breast_cancer_y_test)
plt.plot(breast_cancer_X_test, breast_cancer_y_predicted)
plt.show()



