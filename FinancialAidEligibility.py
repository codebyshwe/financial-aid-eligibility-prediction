#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_csv(r"C:\Users\JOSH\Downloads\Student.csv")
df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.describe()


# In[8]:


df.s_no.nunique()


# In[9]:


df.gender.value_counts()


# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(4,5))
sns.set_style("whitegrid")
ax = sns.countplot(x='UG_choice', data=df, palette='pastel')
plt.title('Status Count', fontsize=15)

for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.show()


# In[11]:


plt.figure(figsize=(4,5))
ax = sns.countplot(x='gender', data=df, palette='viridis')
plt.title('Gender Count', fontsize=15)

for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.show()


# In[12]:


plt.figure(figsize=(8,6))
sns.histplot(df['Annual_Income'], bins=20, kde=True, color='skyblue')
plt.title('Annual_Fees Distribution', fontsize=15)
plt.xlabel('Annual_Income', fontsize=12)
plt.ylabel('No_Of_Dependents', fontsize=12)
plt.show()


# In[17]:


plt.figure(figsize=(6,4))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap', fontsize=15)
plt.show()


# In[18]:


sns.pairplot(df, palette='viridis')
plt.show()


# In[35]:


plt.figure(figsize=(6,4))
df['HSC_GROUP'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'yellow','pink'])
plt.title('HSC GROUP', fontsize=15)
plt.show()


# In[26]:


plt.figure(figsize=(20,8))
sns.set_style("whitegrid")

colors = ["#2F9599", "#FEC601", "#EC2049"]

plt.subplot(1,3,1)
sns.boxplot(data=df, x='Annual_Income', color=colors[0], width=0.5)
plt.title('Annual ncome', fontsize=14)

plt.subplot(1,3,2)
sns.boxplot(data=df, x='gpa', color=colors[1], width=0.5)
plt.title('gpa', fontsize=14)

plt.subplot(1,3,3)
sns.boxplot(data=df, x='Annual_Fees', color=colors[2], width=0.5)
plt.title('Annual Fees', fontsize=14)

plt.suptitle('Outliers Detection', fontsize=20, y=0.95)


# In[27]:


print("Before Removing the outliers", df.shape) 
df = df[df['Annual_Income']<400000] 
print("After Removing the outliers", df.shape)


# In[28]:


print("Before Removing the outliers", df.shape) 
df = df[df['gpa']<3] 
print("After Removing the outliers", df.shape)


# In[29]:


print("Before Removing the outliers", df.shape) 
df = df[df['Annual_Fees']<500000]
print("After Removing the outliers", df.shape)


# In[30]:


fig, axes = plt.subplots(1, 3, figsize=(20, 7))

colors = ["#2F9599", "#FEC601", "#EC2049"]

sns.histplot(df['Annual_Income'], kde=True, color=colors[0], ax=axes[0])
axes[0].set_title('Applicant Income Distribution', fontsize=14)
axes[0].set_xlabel('Applicant Income', fontsize=12)
axes[0].set_ylabel('gpa', fontsize=12)

sns.histplot(df['UG_choice'], kde=True, color=colors[1], ax=axes[1])
axes[1].set_title('Coapplicant Income Distribution', fontsize=14)
axes[1].set_xlabel('Coapplicant Income', fontsize=12)
axes[1].set_ylabel('gpa', fontsize=12)

sns.histplot(df['Annual_Fees'], kde=True, color=colors[2], ax=axes[2])
axes[2].set_title('Loan Amount Distribution', fontsize=14)
axes[2].set_xlabel('Loan Amount', fontsize=12)
axes[2].set_ylabel('gpa', fontsize=12)

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('Distributions of Annual Income, Annual Fees and Ug choice', fontsize=20)
plt.tight_layout(pad=3.0)

plt.show()


# In[31]:


df['Annual_Income'] = np.log(df['Annual_Income'])
df['gpa'] = np.log1p(df['gpa'])

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

colors = ["#66c2a5", "#fc8d62", "#8da0cb"]

sns.histplot(df['Annual_Income'], kde=True, color=colors[0], ax=axes[0])
axes[0].set_title('Annual Income Distribution\n(After Log Transformation)', fontsize=14)
axes[0].set_xlabel('Annual Income (Log Transformed)', fontsize=12)
axes[0].set_ylabel('UG_choice', fontsize=12)

sns.histplot(df['gpa'], kde=True, color=colors[1], ax=axes[1])
axes[1].set_title('gpa Distribution\n(After Log Transformation)', fontsize=14)
axes[1].set_xlabel('gpa (Log Transformed)', fontsize=12)
axes[1].set_ylabel('UG_choice', fontsize=12)

sns.histplot(df['Annual_Fees'], kde=True, color=colors[2], ax=axes[2])
axes[2].set_title('AnnuaLFees Distribution', fontsize=14)
axes[2].set_xlabel('Annual_Fees', fontsize=12)
axes[2].set_ylabel('gpa', fontsize=12)

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('Distributions of Log Transformed Applicant Income, Coapplicant Income, and Original Loan Amount', fontsize=20)
plt.tight_layout(pad=3.0)

plt.show()


# In[32]:


categorical_col = df.select_dtypes(include='object').columns
cat = categorical_col[1:-1]

colors = ["#EC2049", "#FEC601"]

for column in cat:
    plt.figure(figsize=(8,6))
    ax = sns.countplot(x=column, hue="Status", data=df, palette=colors)
    ax.set_xlabel(column, fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    plt.title('Distribution of Loan Status by '+ column, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Displaying the count on top of the bars
    for p in ax.patches:
        ax.annotate(format(p.get_height()), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha = 'center', 
                     va = 'center', 
                     xytext = (0, 10), 
                     textcoords = 'offset points',
                     fontsize = 12)
    plt.show()


# In[20]:


df.select_dtypes('object').head()


# In[19]:


df


# In[15]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

df = df.drop(columns='Ref_No', axis=1)

# handling categorical values
cat_cols = ['gender', 'HSC_GROUP', 'UG_choice', 'Status']

# use label encoding to convert categorical values to numeric
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# handle numerical values
num_cols = ['No_Of_Dependents']  # add other numerical columns to this list

# use standard scaler to scale numeric values
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


# In[16]:


print(df.columns)


# In[14]:


df


# In[21]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True) 
plt.show()


# In[53]:


plt.figure(figsize=(8,6))
sns.countplot(x='gpa', hue='Status', data=df, palette='pastel')
plt.title('gpa vsStatus', fontsize=15)
plt.show()


# In[14]:


df['No_Of_Dependents'] = df['No_Of_Dependents'].replace('4', '3')
df['No_Of_Dependents'] = df['No_Of_Dependents'].astype(int)


# In[16]:


df['No_Of_Dependents'].value_counts()


# In[18]:


x = df.drop(['Status'], axis = 1)


# In[19]:


y = df['Status']


# In[20]:


from imblearn.over_sampling import SMOTE


# In[21]:


x_resample, y_resample = SMOTE().fit_resample(x, y)


# In[22]:


print(x_resample.shape)
print(y_resample.shape)


# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2)


# In[25]:


from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

def evaluate_model(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)
    
    y_pred = model.predict(X_test)
    
    train_accuracy = model.score(X_train, Y_train)
    test_accuracy = model.score(X_test, Y_test)
    
    print(f"Model: {str(model)}")
    print(f"\nModel Accuracy: {accuracy_score(Y_test, y_pred)}")
    print(f"\nTraining Accuracy: {train_accuracy} \nTesting Accuracy: {test_accuracy}")
    print('--------------------------------------')
    
    return model

knn = evaluate_model(KNeighborsClassifier(), X_train, Y_train, X_test, Y_test)
svc = evaluate_model(SVC(), X_train, Y_train, X_test, Y_test)
dt = evaluate_model(DecisionTreeClassifier(), X_train, Y_train, X_test, Y_test)
lr = evaluate_model(LogisticRegression(), X_train, Y_train, X_test, Y_test)
gnb = evaluate_model(GaussianNB(), X_train, Y_train, X_test, Y_test)
rfc = evaluate_model(RandomForestClassifier(), X_train, Y_train, X_test, Y_test)


# In[32]:


from sklearn.model_selection import GridSearchCV
# Set the parameters by cross-validation
tuned_parameters = [{'n_estimators': [50, 100, 200, 500], 
                     'max_depth' : [5, 10, 15, 20, None],
                     'min_samples_split': [2, 4, 10],
                     'min_samples_leaf': [1, 2, 4]}]

rfc = RandomForestClassifier(random_state=42)

clf = GridSearchCV(rfc, tuned_parameters, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

clf.fit(X_train, Y_train)

print("Best parameters set found on development set:")
print(clf.best_params_)

print("Detailed classification report:")
y_true, y_pred = Y_test, clf.predict(X_test)
print((y_true, y_pred))


# In[27]:


best_params = {'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 50}
rfc_best = RandomForestClassifier(**best_params, random_state=42)


# In[28]:


rfc_best.fit(X_train, Y_train)


# In[29]:


y_pred_rfc = rfc_best.predict(X_test)


# In[38]:


print("Confusion Matrix:")
print(Y_test, y_pred_rfc)


# In[39]:


accuracy = accuracy_score(Y_test, y_pred_rfc)
print("Model Accuracy:", accuracy)


# In[40]:


sns.heatmap((Y_test, y_pred_rfc), annot=True, cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:




