# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



# %%
df= pd.read_csv('Phase1.csv')
df.head()

# %%
df.info()

# %%
# Statistiques descriptives pour les variables quantitatives
df.describe()


# %%
df.isnull().sum()

# %%
df1=df.drop_duplicates()

# %%
df1.head()

# %%
df1.info()

# %%
df1.describe()

# %%
from sklearn.preprocessing import StandardScaler, LabelEncoder
encoder = LabelEncoder()
df['protocol_type'] = encoder.fit_transform(df['protocol_type'])
df['service'] = encoder.fit_transform(df['service'])
df['flag'] = encoder.fit_transform(df['flag'])
df['class'] = encoder.fit_transform(df['class'])

# %%
df.info()

# %%
df.head()

# %%
from sklearn.model_selection import train_test_split

X = df.iloc[:,:-1]
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y)

# %%
from sklearn.model_selection import  GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

grid_params={
    'n_neighbors':[3,5,7,9,11],
    'metric':['euclidean','manhattan']
}

gs=GridSearchCV(
    KNeighborsClassifier(),
    grid_params,
    cv=3 #nombre de folds dans la cross validation
)
gs_results=gs.fit(X_train, y_train)

# %%
gs_results.best_params_

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
knn= KNeighborsClassifier(n_neighbors=5,metric='manhattan')
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
precision_scorer = make_scorer(precision_score, pos_label=1)
recall_scorer = make_scorer(recall_score, pos_label=1)
f1_scorer = make_scorer(f1_score, pos_label=1)
precision = cross_val_score(df, X_scaled, y, cv=cv_strategy, scoring=precision_scorer)
recall = cross_val_score(df, X_scaled, y, cv=cv_strategy, scoring=recall_scorer)
f1 = cross_val_score(df, X_scaled, y, cv=cv_strategy, scoring=f1_scorer)

knn.fit(X_train, y_train)
y_pred=knn.predict(X_test) 

# %%

print('precision=',precision_score(y_test,y_pred))
print('recall=',recall_score(y_test,y_pred))

# %%
model = LogisticRegression()
scoring = {'f1': make_scorer(f1_score)}


