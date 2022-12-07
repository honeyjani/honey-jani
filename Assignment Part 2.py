#!/usr/bin/env python
# coding: utf-8

# #### Import neccesary Libraries

# In[4]:


import pandas as pd 
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# #### Import the data

# In[5]:


Amazon_data = pd.read_csv("C://Users//honey//Downloads//DS - Assignment Part 2 data set//amz_com-ecommerce_sample.csv", encoding = 'unicode_escape', engine ='python')


# In[20]:


Flipkart_data = pd.read_csv("C://Users//honey//Downloads//DS - Assignment Part 2 data set//flipkart_com-ecommerce_sample.csv", encoding = 'unicode_escape', engine ='python')


# #### Understanding the data and Pre-processing

# In[6]:


Amazon_data.shape


# `Dataset has 15 columns (features) and 20000 rows (records).`

# In[21]:


Flipkart_data.shape


# `Dataset has 15 columns (features) and 20000 rows (records).`

# In[7]:


Amazon_data.describe()


# `This provides the summary statistics for all numerical variables in the dataset.`

# In[22]:


Flipkart_data.describe()


# In[8]:


Amazon_data.info()


# `Info gives null / not-null count and Data Types for every column.`
# * `As from the above we can conclude that we have majorly object data types and 2 of int and 1 of bool data types.`
# * `Four columns in the dataset has kind of missing values.`

# In[23]:


Flipkart_data.info()


# `Info gives null / not-null count and Data Types for every column.`
# * `As from the above we can conclude that we have majorly object data types and 2 of int and 1 of bool data types.`
# * `Six columns in the dataset has kind of missing values.`

# In[9]:


Amazon_data.nunique()


# In[24]:


Flipkart_data.nunique()


# `"nunique" describes number of unique values per column.`
# 
# `As product name can be considered as an important feature - we can say that product_name has a few number of duplicates but every product has a uniq id.`

# #### Exploratory Data Analysis 

# In[17]:


percent_missing = Amazon_data.isnull().sum() * 100 / len(Amazon_data)
missing_value_df = pd.DataFrame({'column_name': Amazon_data.columns,
                                 'percent_missing': percent_missing})
missing_value_df


# In[25]:


percent_missing = Flipkart_data.isnull().sum() * 100 / len(Flipkart_data)
missing_value_df = pd.DataFrame({'column_name': Flipkart_data.columns,
                                 'percent_missing': percent_missing})
missing_value_df


# #### Data Analysis 

# *Merging Data and renaming the columns.* 

# In[28]:


sample = pd.merge(Amazon_data, Flipkart_data, on=['uniq_id'], how='left')
sample = sample[['crawl_timestamp_x','product_name_x','retail_price_x','discounted_price_x','product_name_y','retail_price_y','discounted_price_y']]
sample.rename(columns={'product_name_x': 'Amazon_Product_Name', 'product_name_y': 'Flipkart_Product_Name','retail_price_x': 'Amazon_Retail_Price','discounted_price_x': 'Amazon_Discounted_Price','retail_price_y': 'Flipkart_Retail_Price','discounted_price_y': 'Flipkart_Discounted_Price'}, inplace=True)


# *Comparing Amazon and Flipkart discounted prices by what value.* 

# In[37]:


sample['Compare_discounted_Prices'] = np.where(sample['Amazon_Discounted_Price'] > sample['Flipkart_Discounted_Price'], 'Amazon', 'Flipkart')


# `Labelling Amazon/Flipkart wherever the prices are high.`

# In[36]:


com_value = sample['Compare_discounted_Prices'].tolist()
ama_value = sample['Amazon_Discounted_Price'].tolist()
fli_value = sample['Flipkart_Discounted_Price'].tolist()

val=[]

for value, am_p, fl_p in zip(com_value,ama_value,fli_value):
    if value == "Amazon":
        val.append(am_p-fl_p)
    else:
        val.append(fl_p-am_p)

sample['Compare_value'] = val
sample[['Compare_discounted_Prices','Compare_value']]


# `Labelling Amazon/Flipkart wherever the prices are high and by what value.`

# *Checking the count of unique values for each category(Amazon and Flipkart).* 

# In[38]:


sample['Compare_discounted_Prices'].value_counts()


# *Viewing the countplot.* 

# In[49]:


#sns.countplot(sample['Compare_discounted_Prices'])
sample['Compare_discounted_Prices'].value_counts().plot(kind='bar')


# `From the above plot we can figure out that Amazon products have high discounted prices then Flipkart.`

# *Calculating Percentage difference of Amazon and Flipkart prices.* 

# In[32]:


sample['Percentage difference Amazon'] = round((sample.Amazon_Retail_Price - sample.Amazon_Discounted_Price) / sample.Amazon_Retail_Price * 100,2) 
sample['Percentage difference Flipkart'] = round((sample.Flipkart_Retail_Price - sample.Flipkart_Discounted_Price) / sample.Flipkart_Retail_Price * 100,2)
sample[['Percentage difference Amazon','Percentage difference Flipkart']]


# `This provides by how percentage the retail price is different from the discounted price for both Amazon and Flipkart.`

# ***

# #### Match similar products from the Flipkart dataset with the Amazon dataset 

# *Checking if uniq_id is same both the datasets.* 

# In[41]:


if Amazon_data['uniq_id'].tolist() == Flipkart_data['uniq_id'].tolist():
    print(True)


# *Merging Amazon and Flipkart dataset based on uniq_id.* 

# In[40]:


sample1 = pd.merge(Amazon_data, Flipkart_data, on=['uniq_id'], how='left')

sample1 = sample1[['product_name_x','retail_price_x','discounted_price_x','product_name_y','retail_price_y','discounted_price_y']]
sample1.rename(columns={'product_name_x': 'Amazon_Product_Name', 'product_name_y': 'Flipkart_Product_Name','retail_price_x': 'Amazon_Retail_Price','discounted_price_x': 'Amazon_Discounted_Price','retail_price_y': 'Flipkart_Retail_Price','discounted_price_y': 'Flipkart_Discounted_Price'}, inplace=True)
sample1


# `Uniq_id column has same values in both the datasets. Therefore merging on 'uniq_id' will be the best option to get all product names from both the datasets.`

# `The above is the combined dataset with Amazon's and Flipkart's Product names and their respective retail and discounted prices.`

# *Saving the dataframe to csv.* 

# In[42]:


sample.to_csv('C://Users//honey//OneDrive//Desktop//result.csv')


# *Results based on user input.* 

# In[44]:


user = str(input("Enter a product name:"))
ans = sample[sample['Amazon_Product_Name'].str.contains(user, case=False)]
display(ans)


# `The above is the result provided by the user input. The user has entered a product name and based on that values from both Amazon and Flipkart are being provided.`

# In[ ]:




