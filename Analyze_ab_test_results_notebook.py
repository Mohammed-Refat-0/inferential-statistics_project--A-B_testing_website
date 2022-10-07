#!/usr/bin/env python
# coding: utf-8

# # Analyze A/B Test Results 
#  
# 
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# .
# <a id='intro'></a>
# ## Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists. For this project, we will be working to understand the results of an A/B test run by an e-commerce website.  our goal is to work through this notebook to help the company understand if they should:
# - Implement the new webpage, 
# - Keep the old webpage, or 
# - Perhaps run the experiment longer to make their decision.
# <a id='probability'></a>
# ## Part I - Probability
# 
# To get started, let's import our libraries.

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# 
# **a.** Read in the dataset from the `ab_data.csv` file and take a look at the top few rows here:

# In[2]:


df=pd.read_csv('ab_data.csv')
df.head()


# **b.** Use the cell below to find the number of rows in the dataset.

# In[3]:


df.count()


# **c.** The number of unique users in the dataset.

# In[4]:


df['user_id'].nunique()


# **d.** The proportion of users converted.

# In[5]:


df[df['converted']==1].nunique()
35173/290584


# **e.** The number of times when the "group" is `treatment` but "landing_page" is not a `new_page`.

# In[6]:


df.query('(group == "treatment") and (landing_page != "new_page" )').shape[0]


# **f.** Do any of the rows have missing values?

# In[7]:


df.info()


#  

# In[8]:


# Remove the inaccurate rows, and store the result in a new dataframe df2
df2=df.query('(group == "control" and landing_page == "old_page") or (group == "treatment" and landing_page == "new_page" )')


# In[9]:


# Double Check all of the incorrect rows were removed from df2 - 
# Output of the statement below should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# ### ToDo 1.3  

# **a.** How many unique **user_id**s are in **df2**?

# In[10]:


df2.nunique()


# **b.** There is one **user_id** repeated in **df2**.  What is it?

# In[11]:


d=df2[df2.duplicated(['user_id'])]
d


# **c.** Display the rows for the duplicate **user_id**? 

# In[12]:


df2[df2['user_id']==773192]


# **d.** Remove **one** of the rows with a duplicate **user_id**, from the **df2** dataframe.

# In[13]:


# Remove one of the rows with a duplicate user_id..
df2 = df2.drop(index=2893)
# Check again if the row with a duplicate user_id is deleted or not
df2[df2['user_id']==773192]


# 
# **a.** What is the probability of an individual converting regardless of the page they receive?<br><br>

# In[14]:


Ppoulation=df2[df2['converted']==1].shape[0]
Ppoulation/df2.shape[0]


# **b.** Given that an individual was in the `control` group, what is the probability they converted?

# In[15]:


control_conversion=df2.query(' group == "control" and converted==1').shape[0]/df2.query(' group == "control"').shape[0]
control_conversion


# **c.** Given that an individual was in the `treatment` group, what is the probability they converted?

# In[16]:


treatment_conversion=df2.query(' group == "treatment" and converted==1').shape[0]/df2.query(' group == "treatment"').shape[0]
treatment_conversion


# In[17]:


# Calculate the actual difference (obs_diff) between the conversion rates for the two groups.
obs_diff=control_conversion-treatment_conversion
obs_diff


# **d.** What is the probability that an individual received the new page?

# In[18]:


df2.query('landing_page=="new_page"').shape[0]/df2.shape[0]


# **e.** Consider your results from parts (a) through (d) above, and explain below whether the new `treatment` group users lead to more conversions.

# since the probability of conversion rate of old page is higher that of new page, one could say that the new page led to fewr conversions. but we could not be sue of that conculsion without doing a hypothesis testing or checking the p value.

# <a id='ab_test'></a>
# ## Part II - A/B Test
# 
# 
# However, then the hard questions would be: 
# - Do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  
# - How long do you run to render a decision that neither page is better than another?  

# H0: Pold >= Pnew 
# H1: Pold < Pnew 

# 
# Under the null hypothesis $H_0$, assume that $p_{new}$ and $p_{old}$ are equal. Furthermore, assume that $p_{new}$ and $p_{old}$ both are equal to the **converted** success rate in the `df2` data regardless of the page. So, our assumption is: <br><br>
# <center>
# $p_{new}$ = $p_{old}$ = $p_{population}$
# </center>

# **a.** What is the **conversion rate** for $p_{new}$ under the null hypothesis? 

# In[19]:


Pnew=df2['converted'].mean()
Pnew


# **b.** What is the **conversion rate** for $p_{old}$ under the null hypothesis? 

# In[20]:


Pold=df2['converted'].mean()
Pold


# **c.** What is $n_{new}$, the number of individuals in the treatment group? <br><br>

# In[21]:


Nnew=df2[df2['group']== 'treatment'].shape[0]
Nnew


# **d.** What is $n_{old}$, the number of individuals in the control group?

# In[22]:


Nold=df2[df2['group']== 'control'].shape[0]
Nold


# **e. Simulate Sample for the `treatment` Group**<br> 
# Simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null hypothesis.  <br><br>

# In[23]:


new_page_converted = np.random.choice([1, 0], Nnew, [Pnew, 1 - Pnew])


# **f. Simulate Sample for the `control` Group** <br>
# Simulate $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null hypothesis. <br> Store these $n_{old}$ 1's and 0's in the `old_page_converted` numpy array.

# In[24]:


# Simulate a Sample for the control Group
old_page_converted = np.random.choice([1, 0], Nold,  [Pold, 1 - Pold])


# **g.** Find the difference in the "converted" probability $(p{'}_{new}$ - $p{'}_{old})$ for your simulated samples from the parts (e) and (f) above. 

# In[25]:


diff=new_page_converted.mean()-old_page_converted.mean()
diff

# note: The true answer should be zero, since they have the same probaility
# under the null hypothesis


# 
# **h. Sampling distribution** <br>
# Re-create `new_page_converted` and `old_page_converted` and find the $(p{'}_{new}$ - $p{'}_{old})$ value 10,000 times using the same simulation process you used in parts (a) through (g) above. 
# 
# <br>
# Store all  $(p{'}_{new}$ - $p{'}_{old})$  values in a NumPy array called `p_diffs`.

# In[42]:


# Sampling distribution 
p_diffs = []

for i in range(10000):
        new_page_converted = np.random.choice([0,1],n_new, p=(1-Pnew,Pnew)).mean()
        old_page_converted = np.random.choice([0,1],n_old, p=(1-Pold,Pold)).mean()
        p_diffs.append(new_page_converted - old_page_converted)


# **i. Histogram**<br> 
# 
# Also, use `plt.axvline()` method to mark the actual difference observed  in the `df2` data (recall `obs_diff`), in the chart.  

# In[43]:


p_diffs=np.array(p_diffs)
plt.hist(p_diffs)
plt.title('histogram of p_diffs')
plt.xlabel('difference') 
plt.ylabel('number of Counts') 
plt.axvline(x= obs_diff, color='g');


# **j.** What proportion of the **p_diffs** are greater than the actual difference observed in the `df2` data?

# In[44]:


act_diff = df2[df2['group'] == 'treatment']['converted'].mean() -  df2[df2['group'] == 'control']['converted'].mean()
(p_diffs > act_diff ).mean()


# 
#  - What is this value called in scientific studies?  
#  - What does this value signify in terms of whether or not there is a difference between the new and old pages? *Hint*: Compare the value above with the "Type I error rate (0.05)". 

# It is called P-value. it is commonly used to indiacte weather our conculded results is significant engouh to reject the null hypothesis or weather it could be resulted from an error. 
# since P-value is greater than the alpha (Type I error rate (0.05)), we cannot reject The null hypothesis.

# 
# 
# **l. Using Built-in Methods for Hypothesis Testing**<br>
# We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. 

# In[29]:


import statsmodels.api as sm

# number of conversions with the old_page
convert_old = df2.query(' landing_page == "old_page" and converted == 1 ').shape[0]

# number of conversions with the new_page
convert_new =  df2.query('landing_page == "new_page" and converted == 1 ').shape[0]

# number of individuals who were shown the old_page
n_old =  df2.query('landing_page == "old_page"').shape[0]

# number of individuals who received new_page
n_new = df2.query('landing_page == "new_page"').shape[0]


# **m.** Now use `sm.stats.proportions_ztest()` to compute your test statistic and p-value.  [Here](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportions_ztest.html) is a helpful link on using the built in.
# 
# The syntax is: 
# ```bash
# proportions_ztest(count_array, nobs_array, alternative='larger')
# ```
# where, 
# - `count_array` = represents the number of "converted" for each group
# - `nobs_array` = represents the total number of observations (rows) in each group
# - `alternative` = choose one of the values from `[‘two-sided’, ‘smaller’, ‘larger’]` depending upon two-tailed, left-tailed, or right-tailed respectively. 
# >**Hint**: <br>
# It's a two-tailed if you defined $H_1$ as $(p_{new} = p_{old})$. <br>
# It's a left-tailed if you defined $H_1$ as $(p_{new} < p_{old})$. <br>
# It's a right-tailed if you defined $H_1$ as $(p_{new} > p_{old})$. 
# 
# The built-in function above will return the z_score, p_value. 
# 
# ---
# ### About the two-sample z-test
# Recall that you have plotted a distribution `p_diffs` representing the
# difference in the "converted" probability  $(p{'}_{new}-p{'}_{old})$  for your two simulated samples 10,000 times. 
# 
# Another way for comparing the mean of two independent and normal distribution is a **two-sample z-test**. You can perform the Z-test to calculate the Z_score, as shown in the equation below:
# 
# $$
# Z_{score} = \frac{ (p{'}_{new}-p{'}_{old}) - (p_{new}  -  p_{old})}{ \sqrt{ \frac{\sigma^{2}_{new} }{n_{new}} + \frac{\sigma^{2}_{old} }{n_{old}}  } }
# $$
# 
# where,
# - $p{'}$ is the "converted" success rate in the sample
# - $p_{new}$ and $p_{old}$ are the "converted" success rate for the two groups in the population. 
# - $\sigma_{new}$ and $\sigma_{new}$ are the standard deviation for the two groups in the population. 
# - $n_{new}$ and $n_{old}$ represent the size of the two groups or samples (it's same in our case)
# 
# 
# >Z-test is performed when the sample size is large, and the population variance is known. The z-score represents the distance between the two "converted" success rates in terms of the standard error. 
# 
# Next step is to make a decision to reject or fail to reject the null hypothesis based on comparing these two values: 
# - $Z_{score}$
# - $Z_{\alpha}$ or $Z_{0.05}$, also known as critical value at 95% confidence interval.  $Z_{0.05}$ is 1.645 for one-tailed tests,  and 1.960 for two-tailed test. You can determine the $Z_{\alpha}$ from the z-table manually. 
# 
# Decide if your hypothesis is either a two-tailed, left-tailed, or right-tailed test. Accordingly, reject OR fail to reject the  null based on the comparison between $Z_{score}$ and $Z_{\alpha}$. We determine whether or not the $Z_{score}$ lies in the "rejection region" in the distribution. In other words, a "rejection region" is an interval where the null hypothesis is rejected iff the $Z_{score}$ lies in that region.
# 
# >Hint:<br>
# For a right-tailed test, reject null if $Z_{score}$ > $Z_{\alpha}$. <br>
# For a left-tailed test, reject null if $Z_{score}$ < $Z_{\alpha}$. 
# 
# 
# 
# 
# Reference: 
# - Example 9.1.2 on this [page](https://stats.libretexts.org/Bookshelves/Introductory_Statistics/Book%3A_Introductory_Statistics_(Shafer_and_Zhang)/09%3A_Two-Sample_Problems/9.01%3A_Comparison_of_Two_Population_Means-_Large_Independent_Samples), courtesy www.stats.libretexts.org 

# In[30]:


import statsmodels.api as sm
# ToDo: Complete the sm.stats.proportions_ztest() method arguments

z_score, p_value = sm.stats.proportions_ztest([convert_old,convert_new],[n_old, n_new],alternative='smaller') 
print(z_score, p_value)


# **n.** What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?<br><br>

# it means that we have a confidence level of less than 95% when saying that the conversion rate of new page is more than that of the old page. and since  the p-value is larger than the alpha and similar the the one computed before, we fail to reject the null, and therefore have to accept the null hypothesis.

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# In this final part, we will see that the result you achieved in the A/B test in Part II above can also be achieved by performing regression.<br><br> 
# 
# **a.** Since each row in the `df2` data is either a conversion or no conversion, what type of regression should you be performing in this case?

# Logistic regression

# **b.** The goal is to use **statsmodels** library to fit the regression model you specified in part **a.** above to see if there is a significant difference in conversion based on the page-type a customer receives. However, you first need to create the following two columns in the `df2` dataframe:
#  1. `intercept` - It should be `1` in the entire column. 
#  2. `ab_page` - It's a dummy variable column, having a value `1` when an individual receives the **treatment**, otherwise `0`.  

# In[31]:


import statsmodels.api as sm

df2['intercept']=1
df2[['l','ab_page']]=pd.get_dummies(df2['group'])
df2=df2.drop('l',axis=1)
df2.head()


# **c.** Use **statsmodels** to instantiate your regression model on the two columns you created in part (b). above, then fit the model to predict whether or not an individual converts. 
# 

# In[32]:


LogModel=sm.Logit(df2['converted'],df2[['intercept','ab_page']])
R=LogModel.fit() 


# **d.** Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[33]:


R.summary2()


# **e.** What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  

# The associated P-value is 0.1899. it is different becuase in this part we performed a two-sided test, unlike the pervious part. 
# since the p-value is higher than 0.05, we  still accept the null hypothesis.

# **f.** Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# considering other factors make the model more insightful, since other factors may be more correlated with the dependent variable. a numbers of disadvantage rise from using a multi-independent varaible model, like multicollinearity

# **g. Adding countries**<br> 
# Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in. 
# 
# 1. You will need to read in the **countries.csv** dataset and merge together your `df2` datasets on the appropriate rows. You call the resulting dataframe `df_merged`. [Here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# 2. Does it appear that country had an impact on conversion?  To answer this question, consider the three unique values, `['UK', 'US', 'CA']`, in the `country` column. Create dummy variables for these country columns. 
#  Provide the statistical output as well as a written response to answer this question.

# In[34]:


# Read the countries.csv
countries= pd.read_csv('countries.csv')
countries.head()


# In[35]:


# Join with the df2 dataframe
df_merged= countries.set_index('user_id').join(df2.set_index('user_id'))
df_merged.head()


# In[36]:


# Create the necessary dummy variables
df_merged[['US', 'UK', 'CA']] = pd.get_dummies(df_merged['country'])
df_merged=df_merged.drop('CA',axis=1)
df_merged.head()


# **h. Fit your model and obtain the results**<br> 
# Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if are there significant effects on conversion.  **Create the necessary additional columns, and fit the new model.** 
# 
# 
# Provide the summary results (statistical output), and your conclusions (written response) based on the results. 

# In[46]:


# Fit your model, and summarize the results

df_merged['ab_page_US'] = df_merged['ab_page'] * df_merged['US']
df_merged['ab_page_UK'] = df_merged['ab_page'] * df_merged['UK']

final_model = sm.Logit(df_merged['converted'], df_merged[['intercept','ab_page','US','UK','ab_page_US','ab_page_UK']])
final_Results= final_model.fit()
final_Results.summary2()


# The countries appear to have no influence on the conversion rate, since all of the p-values are above 0.05.
# conclusion: we accept the null hypothesis
# 
# final conclusion: we should not switch to the new page

# <a id='finalcheck'></a>
# ## Final Check!
# 
# Congratulations!  You have reached the end of the A/B Test Results project!  You should be very proud of all you have accomplished!

# In[ ]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])


# In[ ]:




