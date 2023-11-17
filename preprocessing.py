import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    dataset = pd.read_csv('/Users/shrakhimzhonov/Machine Learning A-Z (Codes and Datasets)/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Python/Data.csv')
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values


    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(x[:, 1:3])
    x[:, 1:3] = imputer.transform(x[:, 1:3])



    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
    x = np.array(ct.fit_transform(x))

    le = LabelEncoder()
    y = le.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    sc = StandardScaler()

    x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
    x_test[:, 3:] = sc.transform(x_test[:, 3:])

    print(x_train)
    print(x_test)