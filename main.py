# IMPORTING THE LIBRARIES
import numpy as np #this library helps in mathematics operations in arrays
import matplotlib.pyplot as plt #this helps in plotting beautiful charts of our ML model
import pandas as pd #this helps in data analysis, data cleaning, data filling etc


# IMPORTING THE DATASET

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# TAKING CARE OF MISSING DATA`

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
#IMPUTER fit is used to apply the imputation strategy to the database
# imputer.fit(x[:, 1:3])
#The transform method of the Imputer class in scikit-learn is used to apply the
# imputation strategy learned during the fit step to the original dataset.
# This method replaces the missing values in the dataset with the imputed values.
x[:, 1:3] = imputer.fit_transform(x[:, 1:3])
# print(x)

# ENCODING CATEGORICAL DATA (To convert strings into numerical data types, so that ML model can make correlations)

#Encoding the independent variable

#IMPORTING columnTransfer from scikit learn library, what this does is
# This can be useful when you have a dataset with different types of data,
# such as both numerical and categorical data, and you want to
# apply different transformations to each type of data.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder #OneHotEnco# ding involves encoding each category as a binary vector, with the length of the vector equal to the number of categories and with a 1 in the position corresponding to the encoded category and 0s in all other positions. This is useful for input data that has categorical variables, since many machine learning algorithms expect numerical input data.
#Categorical data is data that can be divided into groups or categories. These categories do not have a numerical value and are often represented as strings. For example, the color of a car (e.g. red, blue, green)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
# print(x)

#Encoding the dependent variable

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder ()
y = le.fit_transform(y) #LabelEncoder is a class in scikit-learn that is used to encode categorical data as numerical data. It takes in a list of categories and assigns each category a unique integer value. For example, if the input to LabelEncoder is ['cat', 'dog', 'bird'], the output will be [0, 1, 2].
# print(y)

#SPLITTING THE DATA INTO TRAINING AND TEST SET

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

#FEATURE SCALING

# Standardisation is preferred over Normalization because it works all the times, normalisation works only in specific cases
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#only apply feature scaling to your numerical values, no need to apply them in your dummy variables, ie (categorical encoded variables)
#because if it is appllied, then we get nonsense range of scaled values (in this case b/w -3 to =3 coz standardisation)
#and we will lose the interpretability of the model, it won't be identifiable to which category they represent to
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
