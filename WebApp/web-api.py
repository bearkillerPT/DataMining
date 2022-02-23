from tokenize import Number
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from flask import Flask, jsonify, request
from flask_restful import reqparse
from flask_cors import CORS, cross_origin

app = Flask(__name__)
api = CORS(app, resources={r"*": {"origins": "*"}})
dataset = pd.read_csv('../dataset/healthcare-dataset-stroke-data.csv')
df_dataset = pd.DataFrame(data=dataset)

# Dropping 'id' column
df_dataset.drop('id', axis=1, inplace=True)

# Substituting NaN (null) entries with the bmi means
df_dataset['bmi'].fillna(df_dataset['bmi'].mean(), inplace=True)

categoric_vars = ["gender", "ever_married",
                  "work_type", "Residence_type", "smoking_status"]
# Encoding of non-numerical variables!
for categoric_var in categoric_vars:
    df_dataset[categoric_var].replace({label: int(idx) for idx, label in enumerate(
        np.unique(df_dataset[categoric_var]))}, inplace=True)

oversample = SMOTE()
over_x, over_y = oversample.fit_resample(df_dataset.drop(
    'stroke', axis=1), df_dataset['stroke'].astype('int'))
oversampled_data = over_x.join(over_y)


df_dataset.loc[df_dataset["stroke"] ==
               0] = df_dataset.loc[df_dataset["stroke"] == 0].sample(249)
df_dataset.dropna(inplace=True)
df_dataset.reset_index(drop=True, inplace=True)
x_train, x_test, y_train, y_test = train_test_split(df_dataset.drop(
    'stroke', axis=1), df_dataset['stroke'].astype('int'), test_size=0.2)
over_x_train, over_x_test, over_y_train, over_y_test = train_test_split(
    over_x, over_y, test_size=0.2)

under_logistic_cls = LogisticRegression(max_iter=1000).fit(x_train, y_train)
over_logistic_cls = LogisticRegression(
    max_iter=1000).fit(over_x_train, over_y_train)
under_decision_tree_cls = tree.DecisionTreeClassifier().fit(x_train, y_train)
over_decision_tree_cls = tree.DecisionTreeClassifier().fit(over_x_train,
                                                           over_y_train)
under_random_forest_cls = RandomForestClassifier().fit(x_train, y_train)
over_random_forest_cls = RandomForestClassifier().fit(over_x_train, over_y_train)
under_k_nearest_neighbors_cls = KNeighborsClassifier().fit(x_train, y_train)
over_k_nearest_neighbors_cls = KNeighborsClassifier().fit(over_x_train, over_y_train)
under_svm_cls = SVC(kernel='rbf', probability=True).fit(x_train, y_train)
over_svm_cls = SVC(kernel='rbf', probability=True).fit(
    over_x_train, over_y_train)
under_naive_bayes_cls = GaussianNB().fit(x_train, y_train)
over_naive_bayes_cls = GaussianNB().fit(over_x_train, over_y_train)


@app.route("/")
@cross_origin()
def get():
    parser = reqparse.RequestParser()
    parser.add_argument('age', type=int)
    parser.add_argument('gender', type=int)
    parser.add_argument('hypertension', type=int)
    parser.add_argument('heart_disease', type=int)
    parser.add_argument('ever_married', type=int)
    parser.add_argument('work_type', type=int)
    parser.add_argument('Residence_type', type=int)
    parser.add_argument('avg_glucose_level', type=int)
    parser.add_argument('bmi', type=int)
    parser.add_argument('smoking_status', type=int)
    args = parser.parse_args()
    args = pd.DataFrame(data=args, index=[0])

    return jsonify({
        "Logistic Regression": {
            "Under-Sampling":  int(under_logistic_cls.predict(args)[0]),
            "Over-Sampling":  int(over_logistic_cls.predict(args)[0])
        },
        "Decision Tree": {
            "Under-Sampling": int(under_decision_tree_cls.predict(args)[0]),
            "Over-Sampling": int(over_decision_tree_cls.predict(args)[0])
        },
        "Random Forest": {
            "Under-Sampling": int(under_random_forest_cls.predict(args)[0]),
            "Over-Sampling": int(over_random_forest_cls.predict(args)[0])
        },
        "K_Nearest Neighbors": {
            "Under-Sampling": int(under_k_nearest_neighbors_cls.predict(args)[0]),
            "Over-Sampling": int(over_k_nearest_neighbors_cls.predict(args)[0])
        },
        "SVM": {
            "Under-Sampling": int(under_svm_cls.predict(args)[0]),
            "Over-Sampling": int(over_svm_cls.predict(args)[0])
        },
        "Naive_Bayes": {
            "Under-Sampling": int(under_naive_bayes_cls.predict(args)[0]),
            "Over-Sampling": int(over_naive_bayes_cls.predict(args)[0])
        }
    })


if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=3003)
