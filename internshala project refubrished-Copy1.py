#!/usr/bin/env python
# coding: utf-8

# # ` Problem Statement ` 
# 
# > Indented block
# 
# **Data Science Problem Statement**
# 
# Your client is an Insurance company and they need your help in building a model to predict
# whether the policyholder (customer) will pay next premium on time or not.
# 
# **Evaluation Metric**
# 
# Our evaluation will based on AUC-ROC score
# 
# 
# 

# ### **Understanding the Data set**
# 
# Now, in order to predict whether the customer would pay the next premium or not, you have
# information about past premium payment history for the policyholders along with their
# demographics (age, monthly income, area type) and sourcing channel etc.
# 
# 
# There are two data sets : **`train.csv`**  including the target feature.
# 
# `test.csv` which is the test data set without the target feature.

# ## Importing necessary libraries
# 
# The following code is written in Python 3.x. Libraries provide pre-written functionality to perform necessary tasks.

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Data loading and Cleaning
# 
# 
# 
# 
# *  In this task, we'll load the dataframe in pandas, drop the unnecessary columns and display the top five rows of the dataset.

# In[5]:


# Loading the dataframe

df = pd.read_csv('train_file.csv')

print(df.shape)

df.head()


# # Variable identification#

# In[6]:


# CHECK THE DATATYPES OF ALL COLUMNS:
    
print(df.dtypes)


# # classifying our given data as :
# * categorical variables
# * numerical variables

# In[7]:


# IDENTIFYING CATEGORICAL FEATURES
# Finding the total missing values and arranging them in ascending order
total = df.isnull().sum()

# Converting the missing values in percentage
percent = (df.isnull().sum()/df.isnull().count())
print(percent)
df.head()


# In[8]:



# IDENTIFYING NUMERICAL FEATURES

numeric_data = df.select_dtypes(include=np.number) # select_dtypes selects data with numeric features
numeric_col = numeric_data.columns   # we will store the numeric features in a variable

print("Numeric Features:")
print(numeric_data.head())


# # Check Missing Data 
# 
# One of the main steps in data preprocessing is handling missing data. Missing data means absence of observations in columns that can be caused while procuring the data, lack of information, incomplete results etc. Feeding missing data to your machine learning model could lead to wrong prediction or classification. Hence it is necessary to identify missing values and treat them.
# 
# - In the code below, we calculate the total missing values and the percentage of missing values in every feature of the dataset.
# - The code ideally returns a dataframe consisting of the feature names as index and two columns having the count and percentage of missing values in that feature.

# In[9]:


# To identify the number of missing values in every feature

# Finding the total missing values and arranging them in ascending order
total = df.isnull().sum()

# Converting the missing values in percentage
print(percent)
print(total)


# # Univariate Analysis

# In[10]:


df.describe()


# In[11]:


#plotting a histogram of no_of_premiums_paids
df["no_of_premiums_paid"].plot.hist()


# In[12]:


#plotting a box plot for no of premium paids
df['Count_3-6_months_late'].plot.hist()


# In[13]:


#plotting a histogram for perc premium paid by cash
df["perc_premium_paid_by_cash_credit"].plot.hist()


# In[14]:


#plotting a box plot for perc premium paid by cash or credit
df["perc_premium_paid_by_cash_credit"].plot.box()


# In[15]:


# plotting a histogram for age in days
df["age_in_days"].plot.hist()


# In[16]:


# plotting a box plot for age in days
df["age_in_days"].plot.box()


# # Univariate analysis of categorical column 
# 
# Univariate analysis means analysis of a single variable. It’s mainly describes the characteristics of the variable.
# 
# If the variable is categorical we can use either a bar chart or a pie chart to find the distribution of the classes in the variable.
# * The code plots the frequency of all the values in the categorical variables.

# In[17]:


# Selecting the categorical columns

#creating frequency for area_type 
df["residence_area_type"].value_counts()

#creating percentages from frequencies
df["residence_area_type"].value_counts() / len(df["residence_area_type"])


# In[18]:


#creating a bar plot
df["residence_area_type"].value_counts().plot.bar()


# In[19]:


#creating percentages of area_type
(df["residence_area_type"].value_counts() / len(df["residence_area_type"])).plot.bar()


# Observations :
# From the above visuals, we can make the following observations:
# * nearly about 0.4 % customes belongs to Rural area of residency
# * nearly about 0.6 % customers belongs to Urban area of residency
#    ~ maximum of the customers comes from Urban area of residency and thats a positive sign for the company

# # Bivariate Analysis
# 
# Bivariate analysis involves checking the relationship between two variables simultaneously. In the code below, we plot every categorical feature against the target by plotting a barchart.

# In[20]:


df.corr()


# In[21]:


df.plot.scatter("perc_premium_paid_by_cash_credit","age_in_days")


# In[22]:


df.plot.scatter("target","Count_3-6_months_late")


# In[23]:


df.plot.scatter("target","Count_more_than_12_months_late")


# In[24]:


df.plot.scatter("target","no_of_premiums_paid")


# In[25]:


df.plot.scatter("age_in_days","no_of_premiums_paid")


# In[26]:


df.plot.scatter("perc_premium_paid_by_cash_credit","Count_6-12_months_late")


# ## observations :
#  
#  * As the number of days increases the chances of paying the premium also decreases
#  * The maximum no of premium are paid between  age in days from 1200 - 1500 
#  * The number of premium paid is directly proportional to our target variable
#  * As the count of the days increases the chances of fulfulling our target variable also decreases

# # categorical - continuous variables

# In[27]:


df.groupby("residence_area_type")["Income"].mean().plot.bar()


# In[28]:


# performing t-test
from scipy.stats import ttest_ind
urbans = df[df["residence_area_type"]=="Urban"]
rurals = df[df["residence_area_type"]=="Rural"]
ttest_ind(urbans["Income"],rurals["Income"])


# In[29]:


df.groupby("residence_area_type")["Count_more_than_12_months_late"].mean().plot.bar()


# In[30]:


# performing t-test

#urbans = df[df["residence_area_type"]=="Urban"]
#rurals = df[df["residence_area_type"]=="Rural"]
#ttest_ind(urbans["Count_more_than_12_months_late"],rurals["Count_more_than_12_months_late"])


# ## Observations :
#  
# * The mean income of the proples living in urban areas are slightly high as compared to rural areas
# * Also the nnumber of peoples paying premium late by more than 12 months are slightly higher in rural areas as compared to urban areas

# ##  Categorical-categorical Analysis
# 

# In[31]:


pd.crosstab(df["residence_area_type"],df["target"])


# In[32]:


pd.crosstab(df["residence_area_type"],df["sourcing_channel"])


# In[33]:



#from scipy.stats import chi2_contigency

#chi2_contigency(pd.crosstab(df["residence_area_type"],df["sourcing_channel"]))


# # Missing Value Treatment

# In[34]:


df.isnull().sum()


# In[35]:



# dropping rows where all the entries are missing
df.dropna(how="all")

# dropping columns where all the entries are missing
df.dropna(axis=1,how="all")

# filling the missing values of column(Count_6-12_months_late) with median
median =df["Count_3-6_months_late"].median()
df["Count_3-6_months_late"].fillna(median,inplace = True)

# filling the missing values of column(Count_6-12_months_late) with median
median =df["Count_6-12_months_late"].median()
df["Count_6-12_months_late"].fillna(median,inplace = True)

# filling the missing values of column(Count_more_than_12_months_late ) with median
median =df["Count_more_than_12_months_late"].median()
df["Count_more_than_12_months_late"].fillna(median,inplace = True)

# filling the missing values of column(application_underwriting_score  ) with median
median =df["application_underwriting_score"].median()
df["application_underwriting_score"].fillna(median,inplace = True)


# In[36]:


df.isnull().sum()


# So we can see that there is no presence of missing values in our data set. Now it has been cleaned with respective tools.

# # Outliers Treatment
# 
# ## Treating outliers in the continuous columns
# 
# * Outliers can be treated in a variety of ways. It depends on the skewness of the feature.
# * To reduce right skewness, we use roots or logarithms or reciprocals (roots are weakest). This is the most common problem in practice.
# * To reduce left skewness, we take squares or cubes or higher powers.

# In[37]:


# detecting outliers in all continuous variables using box plot method
#df["no_of_premiums_paid"].plot.box()
# removing outliers from the dataset
df[df["no_of_premiums_paid"]<23]
df=df[df["no_of_premiums_paid"]<23]
df["no_of_premiums_paid"].plot.box()



# In[38]:


#removing outliers in column(age_in_days)
df.loc[df["age_in_days"]>34000,"age_in_days"]=np.median(df["age_in_days"])

df["age_in_days"].plot.box()


# # Variable Transformation
# 
# ## Variable Transformation is a process by which :-
# 
# * We replace a variable with some function of that variable
# * We change the distribution or relationship of a variable

# In[39]:


# changing income to its logarithimic form
np.log(df["Income"]).plot.box()


# In[40]:


# changing 'Count_3-6_months_late' to its square form
np.sqrt(df["Count_3-6_months_late"],).plot.hist()


# #  Model Building
# 

# In[57]:


df["target"].value_counts()
#df=pd.get_dummies(df)


# In[53]:


sf=pd.read_csv('test_file.csv')
sf=pd.get_dummies(sf)


# In[3]:


#making data set 
train = df[0:63844]
test = df[65845:79855]

main=sf
x_train=train.drop("target", axis=1)
y_train=train["target"]

x_test=test.drop("target",axis=1)
true_p=test["target"]


y_test=main


# In[2]:


#importing libraries
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)


# In[1]:


pred=logreg.predict(y_test)


# In[46]:


pred


# In[47]:


logreg.score(x_test,true_p)


# In[48]:


logreg.score(x_train,y_train)


# In[ ]:




