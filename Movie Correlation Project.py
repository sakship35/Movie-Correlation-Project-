#!/usr/bin/env python
# coding: utf-8

# In[1]:


# First let's import the packages we will use in this project
# You can do this all now or as you need them
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)

# pd.options.mode.chained_assignment = None

# Reading in the data
df = pd.read_csv('/Users/sakshi/Downloads/movies.csv')


# In[2]:


df


# In[3]:


#check for missing values

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


# In[4]:


df['budget'].fillna(df['budget'].mean(), inplace=True)
df['gross'].fillna(df['gross'].mean(), inplace=True)

# Convert to integer
df['budget'] = df['budget'].astype('int64')
df['gross'] = df['gross'].astype('int64')


# In[5]:


df


# In[12]:


df = df.sort_values(by=['gross'], inplace=False, ascending=False)


# In[7]:


pd.set_option('display.max_rows', None)


# In[9]:


# drop duplicates

df['company'].drop_duplicates().sort_values(ascending=False)


# In[16]:


# Scatter plot for budget vs Gross

plt.scatter(x=df['budget'], y=df['gross'])

plt.title('Budget vs Gross Earnings')

plt.xlabel('Gross Earnings')

plt.ylabel('Budget for Film') 

plt.show()


# In[13]:


df.head()


# In[18]:


# plot budget vs gross using seaborn

sns.regplot(x='budget', y='gross', data=df, scatter_kws={"color":"red"}, line_kws={"color":"blue"})


# In[22]:


df.corr(method='pearson') #pearson, kendall, spearman


# In[ ]:


# High correlation between budget and gross


# In[24]:


correlation_matrix = df.corr(method='pearson')

sns.heatmap(correlation_matrix, annot=True)

plt.title('Correlation matrix for numeric features')

plt.xlabel('Movie features')

plt.ylabel('Movie features') 

plt.show()


# In[26]:


df_numerized = df

for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes
        
df_numerized


# In[27]:


correlation_matrix = df_numerized.corr(method='pearson')

sns.heatmap(correlation_matrix, annot=True)

plt.title('Correlation matrix for numeric features')

plt.xlabel('Movie features')

plt.ylabel('Movie features') 

plt.show()


# In[28]:


correlation_mat = df_numerized.corr()
corr_pairs = correlation_mat.unstack()
corr_pairs


# In[29]:


sorted_pairs = corr_pairs.sort_values()
sorted_pairs


# In[31]:


high_corr = sorted_pairs[(sorted_pairs) > 0.5]
high_corr


# In[ ]:


# Votes & budget have the highest correlation to gross earnings

