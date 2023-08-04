import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

df = pd.read_csv('CreditCardFraud.csv')

## 1. FORMAT: entire-missing & duplicated

#missing
missing_columns = df.isna().all() #all(): whether values are all True; nested function
print(missing_columns)
df1 = df.dropna(axis=1, how="all")

#duplicated
# df1_tr = df1.transpose()
# duplicated_columns=df1_tr.duplicated() 
# print(duplicated_columns)
df1['accountNumber'].equals(df1['customerId']) #TRUE
df1['acqCountry'].equals(df1['merchantCountryCode']) #FALSE
df1.loc[df1['acqCountry'] != df1['merchantCountryCode']].index[0]
df1['cardCVV'].equals(df1['enteredCVV']) #FALSE
df1['accountOpenDate'].equals(df1['dateOfLastAddressChange']) #FALSE
df1['expirationDateKeyInMatch'].equals(df1['isFraud']) #FALSE

## 2. VALUE: outliers detect and handle
df1.select_dtypes(include='number')
df1_num = df1[['creditLimit', 'availableMoney', 'transactionAmount', 'currentBalance']]
#2.1 detect boxplot
df1_num.boxplot()
plt.show()
sns.boxplot(x="variable", y="value", data=pd.melt(df1_num))
plt.show()
#2.2 detect:Z-score
def detect_outliers_zscore(data, threshold=3):
    z_scores = np.abs((data - data.mean()) / data.std())
    return z_scores > threshold

outliers_zscore=df1_num.apply(detect_outliers_zscore)
#summarise number of outliers 
outliers_zscore.sum()
#2.3 detect: iqr
def detect_outliers_iqr(data, m=1.5):
    q1, q3 = data.quantile([0.25, 0.75])
    iqr = q3-q1
    t1, t3 = q1-iqr*m, q3+iqr*m
    return (data<t1)|(data>t3)
outliers_iqr = df1_num.apply(detect_outliers_iqr)
outliers_iqr.sum()

## 3. FROMAT: missing value
df1.isna().sum()/len(df1)*100 #percentage of missing
df1_clean = df1.dropna()
(df1.shape[0] - df1_clean.shape[0])/df1.shape[0]

## 4. FORMAT & VALUE: time variables
# select time variables
df1_time = df1[['transactionDateTime', 'currentExpDate', 'accountOpenDate',
       'dateOfLastAddressChange', 'isFraud']]
df1_time.head(10)
#4.1 potential issues
# missing
df1_time.isna().sum()
# inconsistent format
df1_time['transactionDateTime']=pd.to_datetime(df1_time['transactionDateTime'])
df1_time['transactionDate']=df1_time['transactionDateTime'].dt.date
df1_time['transactionTime']=df1_time['transactionDateTime'].dt.time
df1_time['currentExpDate_new']=pd.to_datetime(df1_time['currentExpDate'])

#4.2 extract fearures 
df1_time['transactionTimeRange'] = pd.cut(df1_time['transactionDateTime'].dt.hour, bins=[0,6,12,18,24], labels=['midnight', 'morning', 'afternoon', 'evening']) #range(0, 24, 6)=[0,6,12,18]  end point won't be included in edge vector => range(0,25,6)=[0,6,12,18,24]
df1_time['transactionTimeRange'].value_counts()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
df0=df1_time[df1_time['isFraud']].groupby('transactionTimeRange').size().reset_index(name='count')
sns.barplot(y='count', x='transactionTimeRange', data=df0, ax=axs[0])
df0=df1_time[df1_time['isFraud']==False].groupby('transactionTimeRange').size().reset_index(name='count')
sns.barplot(y='count', x='transactionTimeRange', data=df0, ax=axs[1])
axs[0].set_title('Fraud~time range')
axs[1].set_title('Non-fraud~time range')
plt.show()
## 5. VALUE: special variables
# convert to string type, since they shouldn't be regarded as integer
df1[['cardCVV','enteredCVV','cardLast4Digits']] = df1[['cardCVV','enteredCVV','cardLast4Digits']].astype(str)
df1[['cardCVV','enteredCVV','cardLast4Digits']].dtypes

# 5.1 compare cvv & fraud
df1['compareCVV'] = df1['cardCVV'] == df1['enteredCVV']
ct = pd.crosstab(df1['compareCVV'], df1['isFraud']) #crosstab, contigency table
ct.loc['Total'] = ct.sum()
ct['Total'] = ct.sum(axis=1)
ct['Percentage'] = ct.iloc[:,1]/ct['Total']
print(ct)
sns.countplot(x="compareCVV", hue="isFraud", data=df1)
# only consider unmatched CVV 
# "raw" way
fraud_counts = df1[df1['compareCVV']==False].groupby('isFraud').size()
plt.bar(fraud_counts.index, fraud_counts.values)
plt.title('cardCVV not match enteredCVV')
plt.show()
# "delicate" way
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sns.countplot(x="isFraud", data=df1[df1['compareCVV']==False], ax=axs[0])
sns.countplot(x="isFraud", data=df1[df1['compareCVV']], ax=axs[1])
axs[0].set_title('CVV-unmatched')
axs[1].set_title('CVV-matched')
plt.show()
# 5.2 cardLast4Digits-isFraud
# which types of 4digits are more likely related to fraud transcations
df1_fraud = df1[df1['isFraud']]
counts = df1_fraud.groupby('cardLast4Digits').size()
counts.sort_values(ascending=False)

counts_all = df1.groupby('cardLast4Digits').size()
counts_all.sort_values(ascending=False)

irreg_cardLast4Digits = df1[df1['cardLast4Digits'].str.len()!=4].groupby('cardLast4Digits').size()#there are many irregular cardLast4Digits, i.e., less than 4 digits

# examing whether "less than 4 digits" is a feature related to fraud
df1['is4Digits'] = df1['cardLast4Digits'].str.len()==4
ct2 = pd.crosstab(df1['is4Digits'], df1['isFraud']) #crosstab, contigency table
ct2.loc['Total'] = ct2.sum()
ct2['Total'] = ct2.sum(axis=1)
ct2['Percentage'] = ct2.iloc[:,1]/ct2['Total']
print(ct2)
sns.countplot(x="is4Digits", hue="isFraud", data=df1)

## 7. VALUE: explore-transactionAmount
# 7.1 overall
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.histplot(df1, x='transactionAmount', kde=True, ax=ax[0])
#sns.kdeplot(df1['transactionAmount'], ax=ax)
sns.boxplot(y='transactionAmount', data=df1, ax=ax[1])
ax[0].set_title('Density plot of transactionAmount')
ax[1].set_title('Boxplot of transactionAmount')
plt.show()

df1['transactionAmount'].describe()
# 7.2 details
#count zeros
len(df1[df1['transactionAmount']==0])/len(df1) #2.83% zeros in all values
#explore "tail"
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sns.histplot(df1[df1['transactionAmount']>500], x='transactionAmount', ax=axs[0])
sns.histplot(df1[df1['transactionAmount']>1200], x='transactionAmount',ax=axs[1])
# Add titles
axs[0].set_title('transactionAmount > 500')
axs[1].set_title('transactionAmount > 1200')
plt.show()

## 8. VALUE: categorical vars vs isFraud
#create categorical df
df1_cate = df1[['merchantCountryCode',
 'posEntryMode',
 'posConditionCode',
 'merchantCategoryCode','transactionType', 'isFraud']]

#calculate fraud rate
# 8.1 drop na
cate_vars = ['merchantCountryCode', 'posEntryMode', 'posConditionCode', 'merchantCategoryCode','transactionType']

for var in cate_vars:
    df0 = df1_cate.groupby(var)['isFraud'].mean().reset_index()
    sns.barplot(x=df0.iloc[:,0], y=df0.iloc[:,1],data=df0)
    plt.title(f'Fraud Rate for {var}')
    if var == 'merchantCategoryCode':
        plt.xticks(rotation=45)
    plt.show()

# 8.2 fill in NA
df1_cate.fillna('None', inplace=True)
for var in cate_vars:
    df0 = df1_cate.groupby(var)['isFraud'].mean().reset_index()
    if var == 'merchantCategoryCode':
        plt.figure(figsize=(9, 5))
        sns.barplot(x=df0.iloc[:,0], y=df0.iloc[:,1],data=df0)
        plt.xticks(rotation=45)
    else:
        sns.barplot(x=df0.iloc[:,0], y=df0.iloc[:,1],data=df0)
    plt.title(f'Fraud Rate for {var}')
    
    plt.show()

## 9. VALUE: 2 categorical vars, nested, vs isFraud
#drop na
df1_cate = df1[['merchantCountryCode',
 'posEntryMode',
 'posConditionCode',
 'merchantCategoryCode','transactionType', 'isFraud']]
df0 = df1_cate.groupby(['merchantCategoryCode','transactionType'])['isFraud'].mean().reset_index()
sns.catplot(x='merchantCategoryCode', y='isFraud', hue='transactionType', kind='bar', data=df0, palette='Set2', alpha=0.8, height=6, aspect=1.5)
plt.xticks(rotation=45)
plt.title('IsFraud~Merchant,Transaction(drop na)')
plt.show()

#fill in na with 'none'
df1_cate.fillna('None', inplace=True)
df0 = df1_cate.groupby(['merchantCategoryCode','transactionType'])['isFraud'].mean().reset_index()
sns.catplot(x='merchantCategoryCode', y='isFraud', hue='transactionType', kind='bar', data=df0, palette='Set2', alpha=0.8, height=6, aspect=1.5)
plt.xticks(rotation=45)
plt.title('IsFraud~Merchant,Transaction(with na)')
plt.show()

## 9. Numerical vars vs isFraud
num_vars = ['creditLimit', 'availableMoney', 'transactionAmount', 'currentBalance']
# ERROR! This will build overall kde for numerical variable, then separate it into two groups, True and False, which is not regard each group as an individual kde.
# for var in num_vars:
#     sns.kdeplot(x=var, hue="isFraud", data=df1)
#     plt.title(f'Conditional Density Plot: {var} on isFraud')
#     plt.show()

for var in num_vars:
    # Create a FacetGrid with isFraud as the hue
    g = sns.FacetGrid(df1, hue='isFraud', height=5)
    # Map a kde plot for numerical var column
    g.map(sns.kdeplot, var, shade=True)
    g.add_legend()
    plt.title(f'Conditional Density Plot: {var} on isFraud')
    plt.show()
    
    
## 11. VALUE: Define multi-swipe 
# 11.1 define and build dataframe
def multi_swipe_transactions(df, max_minute_diff=5, max_amount_diff=0.01):
    df = df.dropna(subset=['transactionDateTime', 'transactionAmount'])
    df['transactionDateTime']=pd.to_datetime(df['transactionDateTime'])
    df['index'] = range(1, len(df)+1)
    
    max_time_diff = datetime.timedelta(minutes=max_minute_diff)
    
    df_grouped = df.groupby(['accountNumber', 'merchantName'])
    
    target_df = pd.DataFrame(columns=['id', 'amountName', 'merchantName', 'transactionDateTime', 'transactionAmount', 'isFraud'])
    
    for key, group in df_grouped:
        for i in range(1, len(group)):
            curr=group.iloc[i,:]
            prev=group.iloc[i-1,:]

            time_diff=curr['transactionDateTime']-prev['transactionDateTime']
            amount_diff=abs(curr['transactionAmount']-prev['transactionAmount'])
            
            if (time_diff<=max_time_diff) and (amount_diff<=max_amount_diff):
                target_df = target_df.append({'id': curr['index'], 'amountName':key[0], 'merchantName': key[1], 'transactionDateTime': curr['transactionDateTime'], 'transactionAmount': curr['transactionAmount'], 'isFraud': curr['isFraud']}, ignore_index=True)
    return target_df

df1_multi_swipe = multi_swipe_transactions(df1)
df1_multi_swipe.to_csv('multi_swipe_transactions.csv', index=False)

# 11.2 calculation: percentage
len(df1_multi_swipe)/len(df1)
df1_multi_swipe['transactionAmount'].sum()/df1['transactionAmount'].sum()

df2 = df1.dropna(subset=['transactionDateTime', 'transactionAmount'])
df2['id'] = range(1, len(df2)+1)
df2_multi_all = df2[df2['id'].isin(df1_multi_swipe['id'])]

# 11.3 explore fraud vs non-fraud
# 11.3.1 vs categorical
for var in cate_vars:
    df0 = df2_multi_all.groupby(var)['isFraud'].mean().reset_index()
    sns.barplot(x=df0.iloc[:,0], y=df0.iloc[:,1],data=df0)
    plt.title(f'Fraud Rate for {var}')
    if var == 'merchantCategoryCode':
        plt.xticks(rotation=45)
    plt.show()
    
for var in cate_vars:
    df0 = df2_multi_all.groupby(var)['isFraud'].mean().reset_index()
    if var == 'merchantCategoryCode':
        plt.figure(figsize=(9, 5))
        sns.barplot(x=df0.iloc[:,0], y=df0.iloc[:,1],data=df0)
        plt.xticks(rotation=45)
    else:
        sns.barplot(x=df0.iloc[:,0], y=df0.iloc[:,1],data=df0)
    plt.title(f'Fraud Rate for {var}')
    
    plt.show()
    
df0 = df2_multi_all.groupby(['merchantCategoryCode','transactionType'])['isFraud'].mean().reset_index()
sns.catplot(x='merchantCategoryCode', y='isFraud', hue='transactionType', kind='bar', data=df0, palette='Set2', alpha=0.8, height=6, aspect=1.5)
plt.xticks(rotation=45)
plt.title('IsFraud~Merchant,Transaction(drop na)')
plt.show()
   
df2_multi_all.fillna('None', inplace=True)
df0 = df2_multi_all.groupby(['merchantCategoryCode','transactionType'])['isFraud'].mean().reset_index()
sns.catplot(x='merchantCategoryCode', y='isFraud', hue='transactionType', kind='bar', data=df0, palette='Set2', alpha=0.8, height=6, aspect=1.5)
plt.xticks(rotation=45)
plt.title('IsFraud~Merchant,Transaction(with na)')
plt.show()
# 11.3.2 vs continuous
for var in num_vars:
    # Create a FacetGrid with isFraud as the hue
    g = sns.FacetGrid(df2_multi_all, hue='isFraud', height=5)
    # Map a kde plot for numerical var column
    g.map(sns.kdeplot, var, shade=True)
    g.add_legend()
    plt.title(f'Conditional Density Plot: {var} on isFraud')
    plt.show()
# 11.3.3 average transaction amount
df1_multi_swipe[df1_multi_swipe['isFraud']==False]['transactionAmount'].mean()

# 11.3.4 time range (midnight, morning, afternoon, evening)
df1_multi_swipe = pd.read_csv('multi_swipe_transactions.csv')
df1_multi_swipe['transactionDateTime']=pd.to_datetime(df1_multi_swipe['transactionDateTime'])
df1_multi_swipe['transactionTime']=df1_multi_swipe['transactionDateTime'].dt.time
df1_multi_swipe['transactionTimeRange'] = pd.cut(df1_multi_swipe['transactionDateTime'].dt.hour, bins=[0,6,12,18,24], labels=['midnight', 'morning', 'afternoon', 'evening'])
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
df0=df1_multi_swipe[df1_multi_swipe['isFraud']].groupby('transactionTimeRange').size().reset_index(name='count')
sns.barplot(y='count', x='transactionTimeRange', data=df0, ax=axs[0])
df0=df1_multi_swipe[df1_multi_swipe['isFraud']==False].groupby('transactionTimeRange').size().reset_index(name='count')
sns.barplot(y='count', x='transactionTimeRange', data=df0, ax=axs[1])
axs[0].set_title('Multi-swipe Fraud~time range')
axs[1].set_title('Multi-swipe~time range')
plt.show()

## 12. VALUE: "imbalance" of isFraud
# 12.1 detect imbalance
len(df1[df1['isFraud']])/len(df1)
len(df1[df1['isFraud']==False])/len(df1)
# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

