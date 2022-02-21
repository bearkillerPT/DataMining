import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from flask import Flask
from flask_restful import Resource, Api, reqparse
app = Flask(__name__)
api = Api(app)
dataset = pd.read_csv('../dataset/healthcare-dataset-stroke-data.csv')
df_dataset = pd.DataFrame(data=dataset)

#Dropping 'id' column
df_dataset.drop('id', axis=1, inplace=True)

#Substituting NaN (null) entries with the bmi means
df_dataset['bmi'].fillna(df_dataset['bmi'].mean(), inplace=True)

categoric_vars = ["gender","ever_married","work_type","Residence_type","smoking_status"]
#Encoding of non-numerical variables!
for categoric_var in categoric_vars:

    print(categoric_var + ": " + str({label: int(idx) for idx, label in enumerate(np.unique(df_dataset[categoric_var]))}))
    df_dataset[categoric_var].replace({label: int(idx) for idx, label in enumerate(np.unique(df_dataset[categoric_var]))}, inplace=True)

df_dataset.loc[df_dataset["stroke"] == 0] = df_dataset.loc[df_dataset["stroke"] == 0].sample(249)
df_dataset.dropna(inplace=True)
df_dataset.reset_index(drop=True, inplace=True)
x_train, x_test, y_train, y_test = train_test_split(df_dataset.drop(
    'stroke', axis=1), df_dataset['stroke'].astype('int'), test_size=0.2)

logistic_classifier = LogisticRegression(max_iter=1000).fit(x_train, y_train)
decision_tree_cls = tree.DecisionTreeClassifier().fit(x_train, y_train)
random_forest_cls = RandomForestClassifier().fit(x_train, y_train)
k_nearest_neighbors_cls = KNeighborsClassifier().fit(x_train, y_train)
svm_cls = clf = SVC(kernel='linear', probability=True).fit(x_train, y_train)
naive_bayes_cls = clf = GaussianNB().fit(x_train, y_train)

class logistic_class(Resource):
    def get(self):
        return {'hello': 'world'}
class decision_tree_class(Resource):
    def get(self):
        return {'hello': 'world'}
class random_forest_class(Resource):
    def get(self):
        return {'hello': 'world'}
class k_nearest_neighbors_class(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('key1', type=str)
        return parser.parse_args()
class svm_class(Resource):
    def get(self):
        return {'hello': 'world'}
class naive_bayes_class(Resource):
    def get(self):
        return {'hello': 'world'}

api.add_resource(logistic_class, '/logistic/')
api.add_resource(decision_tree_class, '/decision_tree/')
api.add_resource(random_forest_class, '/random_forest/')
api.add_resource(k_nearest_neighbors_class, '/k_nearest_neighbors/')
api.add_resource(svm_class, '/svm/')
api.add_resource(naive_bayes_class, '/naive_bayes/')

if __name__ == '__main__':
    app.run(debug=True)