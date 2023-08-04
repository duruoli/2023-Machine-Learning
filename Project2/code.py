import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import re
import statistics
import statsmodels.api as sm
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform, jaccard
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer


df_c = pd.read_csv('./Reviews.csv')
#df_c['Date of Review']=pd.to_datetime(df_c['Date of Review'])
df_r = pd.read_csv('./Restaurants.csv')

##1. EDA

#1.1 missing check
#1.1.1 customer
na_percentages = df_c.isna().sum()/len(df_c)
na_df = pd.DataFrame({'Column': na_percentages.index, 'Missing_Percent': na_percentages.values})

# output missing value percentages to LaTeX table
table = na_df.to_latex(index=False, float_format='%.4f') #output latex

df_c1 = df_c.drop("Vegetarian?", axis=1)
#1.1.2 restaurants
# print table
print(table)

na_percentages = df_r.isna().sum()/len(df_r)
na_df = pd.DataFrame({'Column': na_percentages.index, 'Missing_Percent': na_percentages.values})

# output missing value percentages to LaTeX table
table = na_df.to_latex(index=False, float_format='%.4f')


# 1.2 Imbalance check
#customer
colnames_c = df_c1.columns
cates = ['Marital Status', 'Has Children?','Average Amount Spent', 'Preferred Mode of Transport', 'Northwestern Student?']
nums = ['Birth Year', 'Weight (lb)', 'Height (in)']

df_c['Marital Status']=df_c['Marital Status'].replace('SIngle', 'Single')
for var in cates:
    fig = plt.figure()
    sns.countplot(data=df_c, x=var)
    #sns.catplot(data=df_c, x=var, kind='count')
    plt.plot()


for var in nums:
    fig = plt.figure()
    sns.histplot(df_c, x=var)
    plt.show()
    
#restaurant
# cuisine_per = df_r['Cuisine'].value_counts(normalize=True)
plt.figure(figsize=(10, 6))
sns.countplot(data=df_r, x="Cuisine")
plt.xticks(rotation=60)
plt.show()

## 1.3 clustering (customers' demographic)
# cate to one-hot
demo_vars = ['Marital Status', 'Has Children?','Birth Year', 'Weight (lb)', 'Height (in)']
df_cdemo = df_c[['Reviewer Name']+demo_vars]
len(df_cdemo.dropna()) #=1340
df_cdemo1 = df_cdemo.dropna()

df_onehot = pd.get_dummies(df_cdemo1[['Marital Status', 'Has Children?']])
df = pd.concat([df_cdemo1[['Birth Year', 'Weight (lb)', 'Height (in)']], df_onehot], axis=1)
df_scaled = StandardScaler().fit_transform(df)
#clustering-kmeans
#choose best cluster_n
silhouette_score_coef = []
for k in range(2,11):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(df_scaled)
    s = silhouette_score(df_scaled, kmeans.labels_)
    silhouette_score_coef.append(s)
    
plt.plot(range(2,11), silhouette_score_coef)
plt.xticks(range(2,11))
plt.xlabel('number of clusters')
plt.ylabel('silhouette score')
plt.show()

# see cluster performance
kmeans = KMeans(n_clusters=5, random_state=0).fit(df_scaled)

scatter = plt.scatter(df['Weight (lb)'], df['Height (in)'], c=kmeans.labels_)
plt.xlabel('Weight (lb)')
plt.ylabel('Height (in)')
plt.legend(*scatter.legend_elements(), loc='upper right')
plt.show()

scatter = plt.scatter(df['Birth Year'], df['Height (in)'], c=kmeans.labels_)
plt.xlabel('Birth Year')
plt.ylabel('Height (in)')
plt.legend(*scatter.legend_elements(), loc='lower left')
plt.show()

scatter = plt.scatter(df['Birth Year'], df['Weight (lb)'], c=kmeans.labels_)
plt.xlabel('Birth Year')
plt.ylabel('Weight (lb)')
plt.legend(*scatter.legend_elements(), loc='upper left')
plt.show()

## 1.4 Trends in clusters
df_c2 = df_c.dropna(subset=['Marital Status', 'Has Children?','Birth Year', 'Weight (lb)', 'Height (in)'])
df_c2['cluster_label'] = kmeans.labels_
df_c2.groupby('cluster_label')['Rating'].mean().reset_index(name='average_review_score')

### 2. Popularity Matching
## 2.5 preparation-average score
df_rs = df_c.groupby('Restaurant Name')['Rating'].mean().reset_index(name='averageScore')
df_rs['reviewNumber'] = df_c.groupby('Restaurant Name')['Rating'].count().values


df_rs[df_rs['averageScore']==df_rs['averageScore'].max()]
df_rs['averageScore'].describe()
meanScore = df_rs['averageScore'].mean()

#histogram
sns.histplot(data=df_rs, x='averageScore')
plt.show()

## 2.6 preparation-average number
df_rs[df_rs['reviewNumber']==df_rs['reviewNumber'].max()]
medianNumber = df_rs['reviewNumber'].median()
meanNumber = df_rs['reviewNumber'].mean()

## 2.7 model-raw
df_R = pd.merge(df_rs, df_r, on='Restaurant Name', how='outer')
def popMatch(cuisine):
    idx = df_R.groupby('Cuisine')['averageScore'].idxmax()
    df_max = df_R.iloc[idx]
    choice = df_max[df_max['Cuisine']==cuisine]['Restaurant Name']
    return choice

## 2.8 shrinkage estimator
def shrinkFun(score, number):
    if number >= meanNumber:
        return score
    else:
        alpha = number/meanNumber
        return (1-alpha)*meanScore+alpha*score

df_rs['shrinkedAveScore'] = df_rs.apply(lambda row: shrinkFun(row['averageScore'],row['reviewNumber']), axis=1)

df_rs['scoreChange'] = df_rs['shrinkedAveScore']-df_rs['averageScore']
 
df_rs.iloc[df_rs['scoreChange'].idxmax()] 
df_rs.iloc[df_rs['scoreChange'].idxmin()] 

#see the distribution of change
sns.histplot(data=df_rs, x="scoreChange")
plt.show()

K=5
posK = df_rs[df_rs['scoreChange']>0].sort_values(by='scoreChange', ascending=False).iloc[:K]
negK = df_rs[df_rs['scoreChange']<0].sort_values(by='scoreChange', ascending=True).iloc[:K]

df_K = pd.concat([posK, negK], axis=0)
sns.histplot(data=df_K, x="scoreChange")
plt.title('Top 5 change in review scores')
plt.show()

### 3. Content Based Filtering
## 3.9, 3.10 Euclidean & cosine distance matrix
rCateCols = ['Cuisine', 'Open After 8pm?']
rNumCols = ['Latitude', 'Longitude', 'Average Cost']

rCate_Dum = pd.get_dummies(df_r[rCateCols])
df_rDum = pd.concat([df_r[rNumCols], rCate_Dum], axis=1)
#compute distance matrix
rNames = df_r['Restaurant Name']
dist_matrix_euc = pd.DataFrame(euclidean_distances(df_rDum), index=rNames, columns=rNames)
dist_matrix_cos = pd.DataFrame(cosine_distances(df_rDum), index=rNames, columns=rNames)


#coords = df_rDum.values
#dist_matrix_euc = squareform(pdist(coords, metric='euclidean')) #another way to calculate euclidean metrics
#pdist vector form, only consider >0, i.e., exclude diagonal 

## 3.11 build model
def content_based_recommender(user, distType):
    df = df_c[df_c['Reviewer Name']==user]
    rName = df.sort_values(by='Rating', ascending=False)['Restaurant Name'].values[0]
    output = pd.DataFrame(columns=['Top Recommend', 'Restaurant Name'])
    output['Top Recommend'] = range(1,6)
    
    if rName in rNames.values:
        if distType == "euclidean":
            v = dist_matrix_euc[rName]
            recs = v.sort_values()[1:6].index
            output['Restaurant Name'] = recs
        elif distType == 'cosine':
            v = dist_matrix_cos[rName]
            recs = v.sort_values()[1:6].index
            output['Restaurant Name'] = recs
        
        return output   
    else:
        return "Sorry! No recommendations."

content_based_recommender('Kim Hamilton', 'euclidean')

### 4. Distance matrix: NLP
## 4.12 create 'Augmented Description'
df_r['Augmented Description'] = df_r['Brief Description']+' '+ df_r['Cuisine']

## 4.13 Jaccard matrix
#split sentences into words without puntuations

def getWords(text):
    text = text.lower()
    return re.compile('\w+').findall(text)

df_r['Split Words']=df_r.apply(lambda row: getWords(row['Augmented Description']), axis=1)

#Jaccard distance = 1-Jaccard similarity
def jacDistance(x, y):
    intersection = x.intersection(y)
    union = x.union(y)
    similarity = float(len(intersection))/len(union)
    return 1-similarity

# jaccard distance matrix
n=len(df_r)
v = []
for i in range(0,n):
    for j in range(i+1,n):
        x = set(df_r['Split Words'][i])
        y = set(df_r['Split Words'][j])
        d = jacDistance(x,y)
        v.append(d)
resNames = df_r['Restaurant Name'].values
dist_matrix_jac = pd.DataFrame(squareform(np.array(v)), index=resNames, columns=resNames)

## 4.14 TF-IDF function

# def tfidfscore(word):
#     df = pd.DataFrame(0, columns=['TF','TF-IDF'])
#     idf = 1/df_r.apply(lambda row: word in row['Split Words'], axis=1).mean()
#     for i in range(0,len(df_r)):
#         d = df_r['Split Words'][i]
#         tf = d.count(word)/len(d)
#         df['TF-IDF'][i] = tf*math.log(idf)
#     return df

def tfidfscore(word0):
    word = word0.lower()
    idf = 1/df_r.apply(lambda row: word in row['Split Words'], axis=1).mean()
    tfs = df_r.apply(lambda row: row['Split Words'].count(word)/len(row['Split Words']), axis=1) 
    df = pd.DataFrame(tfs*math.log(idf), columns=[word0])
    df = pd.concat([df_r['Restaurant Name'], df], axis=1)
    return df

df = tfidfscore('cozy')
df.sort_values('cozy', ascending=False).iloc[0]

df = tfidfscore('Chinese')
df.sort_values('Chinese', ascending=False).iloc[0]

## 4.15, 4.16 TF-IDF matrix
# 100 most popular words
ds = []
for i in range(0, len(df_r)):
    ds.extend(df_r['Split Words'][i])

most_common = Counter(ds).most_common(100)
wordLists = [x[0] for x in most_common]

# matrix
df0 = pd.DataFrame(df_r['Restaurant Name'])
for word in wordLists:
    idf = 1/df_r.apply(lambda row: word in row['Split Words'], axis=1).mean()
    tfs = df_r.apply(lambda row: row['Split Words'].count(word)/len(row['Split Words']), axis=1) 
    df1 = pd.DataFrame(tfs*math.log(idf), columns=[word])
    df0 = pd.concat([df0, df1], axis=1)

coords = df0.iloc[:,1:]
tfidf_dist_matrix_euc = pd.DataFrame(euclidean_distances(coords), index=rNames, columns=rNames)
tfidf_dist_matrix_cos = pd.DataFrame(cosine_distances(coords), index=rNames, columns=rNames)

## 4.17 Word Embedding (high dimentions->lower dimentions)
# load pretrained model-BERT?
model = SentenceTransformer('all-MiniLM-L6-v2')

# get embedding feature-matrix (sentence 'condenses' to a vector)
df_embedding = df_r.apply(lambda row: pd.Series(model.encode(row['Augmented Description'])), axis=1)

bert_dist_matrix_euc = pd.DataFrame(euclidean_distances(df_embedding), index=rNames, columns=rNames)
bert_dist_matrix_cos = pd.DataFrame(cosine_distances(df_embedding), index=rNames, columns=rNames)

# euclidean_distances(): row-sample, column-feature

## 4.18 compare models: comparison of difference distance
# choose people with >=2 reviews
nameSize = df_c.groupby('Reviewer Name').size().reset_index(name="size")
testNames = nameSize[nameSize['size']>1]['Reviewer Name'].values
df_test = df_c[df_c['Reviewer Name'].isin(testNames)]

# all use euclidean matrix
jac = dist_matrix_jac
tfidf = tfidf_dist_matrix_euc
bert = bert_dist_matrix_euc
def compare_distance(distType, recN):

    dist = distType
    
    ratios = []
    for name in testNames:
        fav = df_c[df_c['Reviewer Name']== name].sort_values('Rating', ascending=False)['Restaurant Name'].iloc[0]
        real = set(df_c[df_c['Reviewer Name']== name]['Restaurant Name'])
        real.remove(fav)
        if len(real)!=0:
            rec = set(dist[fav].sort_values()[1:recN+1].index)
            intersection = real.intersection(rec)
            ratio = len(intersection)/len(real) 
            ratios.append(ratio)
    return statistics.mean(ratios)
    
compare_distance(distType=jac, recN=10) #0.23694328757896121
compare_distance(distType=tfidf, recN=10) #0.18085602569473538
compare_distance(distType=bert, recN=10) #0.30181341744909107

### 5. collaborative filtering: use information of other users
## 5.19 demographic/user matrix
df_c['cName-birth'] = df_c.apply(lambda x: x['Reviewer Name']+str(x['Birth Year']), axis=1) #use 'cName-birth' as the label for a uniqur person

cates = ['Marital Status', 'Has Children?','Average Amount Spent', 'Preferred Mode of Transport', 'Northwestern Student?']
nums = ['Birth Year', 'Weight (lb)', 'Height (in)']
df_cunique = df_c.dropna(subset=cates+nums).drop_duplicates(subset=['cName-birth'])
df=df_cunique

df_onehot = pd.get_dummies(df[cates])
user_matrix = pd.concat([df[nums], df_onehot], axis=1)
user_matrix_norm = StandardScaler().fit_transform(user_matrix)

cNames = df_cunique['cName-birth'].values
user_dist_euc= pd.DataFrame(euclidean_distances(user_matrix_norm), index=cNames, columns=cNames)
user_dist_cos = pd.DataFrame(euclidean_distances(user_matrix), index=cNames, columns=cNames)



## 5.20
def rec_collaborative_demo(user, K, distType):
    if distType == "euclidean":
        user_dist = user_dist_euc
    elif distType == "cosine":
        user_dist = user_dist_cos
    recUsers = user_dist[user].sort_values()[1:K+1].index
    recUserDists =  user_dist[user].sort_values()[1:K+1]
    idx = df_c[df_c['cName-birth'].isin(recUsers)].groupby('cName-birth')['Rating'].idxmax()
    recs = df_c.iloc[idx][['cName-birth', 'Restaurant Name']]
    recRes = set(recs['Restaurant Name'])
    print(f'Target user: {user} \nRecommended restaurants: \n{recRes}')
    return user, recUserDists, recs

[s for s in df_cunique['cName-birth'] if 'Sarah Hardy' in s]
targetUser, distances, recommendations = rec_collaborative_demo(user = 'Sarah Hardy1994.0', K=5, distType='euclidean')

## 5.21 score matrix
# >=4 reviews
#df_cr = df_c[df_c['Restaurant Name'].isin(rNames)]
# add a new label-distinguish different people with the same name
df_c2['cName-birth'] = df_c2.apply(lambda x: x['Reviewer Name']+str(x['Birth Year']), axis=1)


nameSize = df_c2.groupby('cName-birth').size().reset_index(name="size")
cVecNames = nameSize[nameSize['size']>=4]['cName-birth'].values # customers' names used to form score vectors
df_vec0 = df_c2[df_c2['cName-birth'].isin(cVecNames)]
rNames = list(set(df_vec0['Restaurant Name']).union(set(df_r['Restaurant Name'])))
#rNames = list(set(df_vec0['Restaurant Name']))

df_vec = pd.DataFrame(0, columns=rNames, index=cVecNames) #66-dimensional matrix(score vector->user coordinate)

df_vec0_grouped = df_vec0.groupby('cName-birth') 
for user, group0 in df_vec0_grouped:
    group = group0[['Restaurant Name', 'Rating']].groupby('Restaurant Name').mean()
    res = group.index
    df_vec.loc[user, res] = group['Rating'].values
    
df_clustered_score = df_c2[['Restaurant Name', 'Rating', 'cluster_label']].groupby(['cluster_label', 'Restaurant Name']).mean().reset_index() #df_c2 has kmeans cluster labels

clusterLabels = df_c2[df_c2['cName-birth'].isin(cVecNames)][['cName-birth', 'cluster_label']].drop_duplicates(subset=['cName-birth']).set_index('cName-birth')#if more than 1 labels, take the first one


df_vec_mut=df_vec
for user in cVecNames:
    row = df_vec.loc[user] #use index to select, df.loc[]
    clusterLabel = clusterLabels.loc[user]['cluster_label']
    zeroRes = row[row==0].index

    clustered_scores = df_clustered_score[df_clustered_score['cluster_label']==clusterLabel][['Restaurant Name', 'Rating']].set_index('Restaurant Name')
    
    fillRes = list(set(zeroRes).intersection(set(clustered_scores.index))) #there are only certain restaurants which have average scores from clustered data, i.e., some of the missing restaurants' scores couldn't be filled up
    if len(fillRes)>0:
        df_vec_mut.loc[user][fillRes] = clustered_scores.loc[fillRes]['Rating'].values

## 5.22 Score-vector method: recommendation model
cNames = df_vec_mut.index
cvec_dist_euc = pd.DataFrame(euclidean_distances(df_vec_mut), index=cNames, columns=cNames)
cvec_dist_cos = pd.DataFrame(cosine_distances(df_vec_mut), index=cNames, columns=cNames)

def rec_collaborative_score(user, K, distType):
    if distType == "euclidean":
        user_dist = cvec_dist_euc
    elif distType == "cosine":
        user_dist = cvec_dist_cos
    recUsers = user_dist[user].sort_values()[1:K+1].index
    recUserDists =  user_dist[user].sort_values()[1:K+1]
    idx = df_c[df_c['cName-birth'].isin(recUsers)].groupby('cName-birth')['Rating'].idxmax()
    recs = df_c.iloc[idx][['cName-birth', 'Restaurant Name']]
    recRes = set(recs['Restaurant Name'])
    print(f'Target user: {user} \nRecommended restaurants: \n{recRes}')
    return user, recUserDists, recs

rec_collaborative_score(user='Thomas Stoney2003.0', K=10, distType='cosine')

## 5.23 compare models: collaborative filtering
#simplify recommender function
def rec_demo(user, K, distType):
    if distType == "euclidean":
        user_dist = user_dist_euc
    elif distType == "cosine":
        user_dist = user_dist_cos
    recUsers = user_dist[user].sort_values()[1:K+1].index
    recUserDists =  user_dist[user].sort_values()[1:K+1]
    idx = df_c[df_c['cName-birth'].isin(recUsers)].groupby('cName-birth')['Rating'].idxmax()
    recs = df_c.iloc[idx][['cName-birth', 'Restaurant Name']]
    recRes = set(recs['Restaurant Name'])
    return recRes

def rec_score(user, K, distType):
    if distType == "euclidean":
        user_dist = cvec_dist_euc
    elif distType == "cosine":
        user_dist = cvec_dist_cos
    recUsers = user_dist[user].sort_values()[1:K+1].index
    recUserDists =  user_dist[user].sort_values()[1:K+1]
    idx = df_c[df_c['cName-birth'].isin(recUsers)].groupby('cName-birth')['Rating'].idxmax()
    recs = df_c.iloc[idx][['cName-birth', 'Restaurant Name']]
    recRes = set(recs['Restaurant Name'])
    return recRes

ratios = []
testNames = cNames
for name in testNames:
    fav = df_c[df_c['cName-birth']== name].sort_values('Rating', ascending=False)['Restaurant Name'].iloc[0]
    real = set(df_c[df_c['cName-birth']== name]['Restaurant Name'])
    real.remove(fav)
    if len(real)!=0:
        rec = rec_demo(user=name, K=10, distType="euclidean")
        intersection = real.intersection(rec)
        ratio = len(intersection)/len(real) 
        ratios.append(ratio)
statistics.mean(ratios)


ratios = []
testNames = cNames
for name in testNames:
    fav = df_c[df_c['cName-birth']== name].sort_values('Rating', ascending=False)['Restaurant Name'].iloc[0]
    real = set(df_c[df_c['cName-birth']== name]['Restaurant Name'])
    real.remove(fav)
    if len(real)!=0:
        rec = rec_score(user=name, K=10, distType="euclidean")
        intersection = real.intersection(rec)
        ratio = len(intersection)/len(real) 
        ratios.append(ratio)
statistics.mean(ratios)


### 6. Predictive Modeling
## 6.24 Model0: build
cates = ['Marital Status', 'Has Children?','Average Amount Spent', 'Preferred Mode of Transport', 'Northwestern Student?']
nums = ['Birth Year', 'Weight (lb)', 'Height (in)']

df_pred = pd.merge(df_c, df_r[['Restaurant Name', 'Cuisine']], on='Restaurant Name', how='outer').dropna(subset=['Rating', 'Cuisine']+cates+nums) #1328

X0 = df_pred[['Cuisine']+cates+nums]
y0 = df_pred['Rating']
X0 = pd.get_dummies(X0, columns=['Cuisine']+cates)

model0 = LinearRegression().fit(X0, y0)
predicted_score = model0.predict(X0)

## 6.25 Model0: evaluation
# overall test
X=X0
y=y0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model_std = LinearRegression()
model_std.fit(X_train, y_train)

y_pred_std = model_std.predict(X_test)

mse_std = mean_squared_error(y_test, y_pred_std)
print('Mean squared error:', mse_std)

# one-review test
i=2
x3_test = np.array(X_test.iloc[i]).reshape(1, -1)
y3_test = y_test.iloc[i]

y3_pred = model_std.predict(x3_test)

print('Predicted score:', y3_pred, 'Real score:', y3_test)

## 6.26 Shrinkage: Model_lasso
# build the best lasso model (find best alpha by CV)
# find the best alpha value found by cv
lasso_cv = LassoCV(cv=5, random_state=1).fit(X_train, y_train)
print("Best alpha:", lasso_cv.alpha_)
a=lasso_cv.alpha_

#fit the lasso model
model_lasso = Lasso(alpha=a, max_iter=1000).fit(X_train, y_train)
y_pred_lasso = model_lasso.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

print("Standard Linear Regression MSE:", mse_std)
print("Lasso Regression MSE:", mse_lasso)

coefCompare = pd.DataFrame(0, index=X_train.columns, columns=['Standard', 'Lasso'])
coefCompare['Standard'] = model_std.coef_
coefCompare['Lasso'] = model_lasso.coef_
coefCompare
for i, feature in enumerate(X_train.columns):
    print("{}: {}".format(feature, model_lasso.coef_[i]))

coefCompare.sort_values('Lasso', ascending=False)
coefCompare[coefCompare['Lasso']==0].index


## 6.27 Model1_0: Use embedded 'Review Text' (BERT)
model = SentenceTransformer('all-MiniLM-L6-v2')

# get embedding feature-matrix (sentence 'condenses' to a vector)
reviewTextVec = df_c.dropna(subset=['Review Text']).apply(lambda row: pd.Series(model.encode(row['Review Text'])), axis=1)

X = reviewTextVec
y = df_c.dropna(subset=['Review Text'])['Rating']
model1_0 = LinearRegression().fit(X, y)
predicted_score = model1_0.predict(X)

mse_embed = mean_squared_error(y, predicted_score)
# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model1_0 = LinearRegression()
model1_0.fit(X_train, y_train)

y_pred1_0 = model1_0.predict(X_test)

mse1_0 = mean_squared_error(y_test, y_pred1_0)
print('Mean squared error:', mse1_0)


## 6.28 Model1: integrated X (demographic + embedded reviews)
df_pred1 = pd.concat([df_c.dropna(subset=['Review Text']), reviewTextVec], axis=1) #894
df_pred1 = pd.merge(df_pred1, df_r[['Restaurant Name', 'Cuisine']], on='Restaurant Name', how='outer').dropna(subset=['Rating', 'Cuisine']+cates+nums) #859

X1 = df_pred1[['Cuisine']+cates+nums+list(reviewTextVec.columns)]
y1 = df_pred1['Rating']
X1 = pd.get_dummies(X1, columns=['Cuisine']+cates)


model1 = LinearRegression().fit(X1, y1)
predicted_score = model1.predict(X1)

#compare models: use another api(R2, AIC, BIC)
m0 = sm.OLS(y0, sm.add_constant(X0)).fit()
m0.summary()
m1 = sm.OLS(y1, sm.add_constant(X1)).fit()
m1.summary()

# MSE
#m0
X_train, X_test, y_train, y_test = train_test_split(X0, y0, test_size=0.2, random_state=1)
m0 = LinearRegression()
m0.fit(X_train, y_train)
y_pred0 = m0.predict(X_test)
mse0 = mean_squared_error(y_test, y_pred0)
print('Mean squared error:', mse0)
#m1
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=1)
m1 = LinearRegression()
m1.fit(X_train, y_train)
y_pred1 = m1.predict(X_test)
mse1 = mean_squared_error(y_test, y_pred1)
print('Mean squared error:', mse1)

## 6.29: Specific prediction (Cuisine: coffee)
df_cf = pd.merge(df_c, df_r[['Restaurant Name', 'Cuisine']], on='Restaurant Name', how='outer')
df_cf=df_cf[df_cf['Cuisine']=='Coffee'].dropna(subset=['Rating']+nums+cates)

X3 = df_cf[cates+nums]
y3 = df_cf['Rating']
X3 = pd.get_dummies(X3, columns=cates)


model_cf = LinearRegression().fit(X3, y3)
predicted_score = model_cf.predict(X3)

m3=sm.OLS(y3, sm.add_constant(X3)).fit()
m3.summary() #not significant
pd.concat([m3.params, m3.pvalues], axis=1)
## 6.30 explore coefficients of coffee model
# use coefficients as feature importance
sfm = SelectFromModel(model_cf, prefit=True)
significant_features = sfm.get_support(indices=True)
X3_1 = X3.iloc[:,significant_features]
#X3_1 = sfm.transform(X3)

m3_1=sm.OLS(y3, sm.add_constant(X3_1)).fit()
m3_1.summary() #not significant
print(pd.concat([m3_1.params, m3_1.pvalues], axis=1).to_latex())


# Lasso-Coffee
lasso_cv = LassoCV(cv=5, random_state=1).fit(sm.add_constant(X3), y3)
print("Best alpha:", lasso_cv.alpha_)
a=lasso_cv.alpha_


model = sm.OLS(y3, sm.add_constant(X3))
results_fu = model.fit()
results_fr = model.fit_regularized(alpha=a, L1_wt=1)
final = sm.regression.linear_model.OLSResults(model, results_fr.params, model.normalized_cov_params)


X4 = df_cf[['Has Children?', 'Northwestern Student?']+nums]
y4 = df_cf['Rating']
X4 = pd.get_dummies(X4, columns=['Has Children?', 'Northwestern Student?'])
m4=sm.OLS(y4, sm.add_constant(X4)).fit()
m4.summary()
