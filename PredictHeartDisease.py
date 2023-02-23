import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

desired_width = 320
pd.set_option("display.width",desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option("display.max_columns",10)

df = pd.read_csv("/home/samet/Masaüstü/Datasets/framingham.csv")
print(df)

median = df.education.median()
medianCig = df.cigsPerDay.median()
medianBMI = df.BMI.median()
medianGlucose = df.glucose.median()

df.education = df.education.fillna(median)
df.cigsPerDay = df.cigsPerDay.fillna(medianCig)
df.BMI = df.BMI.fillna(medianBMI)
df.glucose = df.glucose.fillna(medianGlucose)



TenYear = df[df.TenYearCHD==1]
print(TenYear.shape)

pd.crosstab(df.age,df.TenYearCHD).plot(kind="bar")
plt.show()

print(df.groupby("TenYearCHD").mean())

subdf = df[["male","age","education","currentSmoker","cigsPerDay","sysBP","diaBP","glucose"]]


X = subdf

y = df.TenYearCHD

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X,y)

print(X_test)

print(model.predict(X_test))

print(model.predict([[0,19,4.0,1,10,120,80,79]]))

print(model.score(X_test,y_test))
