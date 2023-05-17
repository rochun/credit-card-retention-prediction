# -*- coding: utf-8 -*-
"""

# Introduction

Credit card companies have vast amounts of data points that can forecast consumer behavior. 

For this project, I wanted to research my own findings on the relationships between credit card usage and different demographics.

Link to kaggle: https://www.kaggle.com/sakshigoyal7/credit-card-customers

Link to dataset: https://drive.google.com/file/d/1GNeF8anEURRg17bFNpJ_nTdkOKDjItMe/view?usp=sharing

# First Part: Data Cleaning and Data Checks

# Import Libraries
"""

from google.colab import drive
drive.mount('/content/gdrive')

import pandas as pd

cc_df = pd.read_csv('/content/gdrive/MyDrive/creditcard.csv')
cc_df.head()

"""# Data Wrangling and Checking for Nulls"""

cc_df.info()

cc_df.shape

cc_df.dtypes

cc_df.isna().sum()

# filter out unneed columns
cc_df_drop = cc_df.drop(columns=['CLIENTNUM','Contacts_Count_12_mon','Unnamed: 21'])
cc_df_drop.head()

# filter out individuals who are no longer customers
cc_df_new = cc_df_drop[cc_df_drop['Attrition_Flag'] == 'Existing Customer']
cc_df_new.Attrition_Flag.unique()

cc_df_new.shape

pd.set_option('display.max_rows', 20)
cc_df_new.Income_Category.value_counts()

cc_df_new.Education_Level.value_counts()

cc_df_new.Gender.value_counts()

"""## Data Exploration and Visualizations"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
sns.set_style('white')

import warnings
warnings.filterwarnings('ignore')

# Statistics of the credit limits
cc_df_new['Credit_Limit'].describe()

# Credit Limit density chart to determine the distribution
sns.histplot(x=cc_df_new['Credit_Limit'], stat='density' ,bins=80,kde=True)
plt.title('Credit Limit Density Chart')
plt.xlabel('Credit Limit')
plt.ylabel('Density')
plt.show()

"""Most of the credit limit is with in (0,20000]"""

plt.figure(figsize=(8,8))
cc_remove_unknown = cc_df_new[cc_df_new['Education_Level'] != 'Unknown']
sns.barplot(x=cc_remove_unknown['Education_Level'],y=cc_remove_unknown['Credit_Limit'], order=['Uneducated','High School','College','Graduate','Post-Graduate','Doctorate'])
plt.title('Education Level and Average Credit Limit')
plt.xlabel('Education Level')
plt.ylabel('Average Credit Limit')
plt.show()

"""The bar graph indicates similar average credit lines among the education level categories."""

cc_df_new['Avg_Utilization_Ratio'].describe()

sns.histplot(x=cc_df_new['Avg_Utilization_Ratio'] ,bins=80,kde=True)
plt.title('Average Utilization Ratio Density')
plt.xlabel('Average Utilization Ratio')
plt.ylabel('Count')
plt.show()

"""From the histogram, we can see that there is an overwhelming amount of 0.0 utilization rates in the dataset"""

cc_df_new['Total_Trans_Ct'].describe()

# Determine whether there is a normal distribution of transaction counts
sns.histplot(x=cc_df_new['Total_Trans_Ct'], stat='density' ,bins=80,kde=True)
plt.title('Total Transactions Count Density')
plt.xlabel('Total Transactions Count')
plt.ylabel('Density')
plt.show()

"""# Second Part: Hypothesis Testing

## Hypothesis 1: I hypothesize that the credit limit will differ based on education levels

It is a general understanding that income levels are higher as the education level increases.


Does a higher education level indicate a greater credit limit?
"""

sns.catplot(x='Education_Level',y='Credit_Limit',data=cc_remove_unknown)
plt.title('Education Level and Credit Limit Category Plot')
plt.xlabel('Education Level')
plt.ylabel('Credit Limit')
plt.xticks(rotation=40)
plt.show()

"""The category plot visually indicates a similar distribution among all education levels"""

from scipy import stats

catNumbers = {}

for cat in cc_remove_unknown['Education_Level'].unique():
  catNumbers[cat] = cc_remove_unknown.loc[cc_remove_unknown['Education_Level'] == cat, 'Credit_Limit'].values

input = catNumbers.values()

# Use the Kruskal Wallis Test to determine if credit limits differs based on education level
stats.kruskal(*input)

# Since the p-value > 0.05 (alpha), the hypothesis is not statistically significant

"""## Hypothesis 2: I hypothesize that individuals with incomes of less than or equal to 60k will have higher utilization ratios than those making more than 60k

Individuals with low income often live paycheck-to-paycheck and have to rely on credit to make essential purchases.

Do individuals with low to middle income use a higher percentage of their available credit than high income individuals?
"""

# remove unknown income rows from dataframe
cc_df_hyp2 = cc_df_new[cc_df_new['Income_Category'] != 'Unknown']
cc_df_hyp2.info()

cc_df_hyp2['Avg_Utilization_Ratio'].describe()

# Create a dataframe that only includes Income below or equal to 60k
lessthansixty = cc_df_hyp2.loc[(cc_df_hyp2['Income_Category'] == 'Less than $40K')| (cc_df_hyp2['Income_Category'] == '$40K - $60K'), ['Income_Category','Avg_Utilization_Ratio']]

lessthansixty.info()

# Create a dataframe that only includes Income above 60K
morethansixty = cc_df_hyp2.loc[(cc_df_hyp2['Income_Category'] != 'Less than $40K') & (cc_df_hyp2['Income_Category'] != '$40K - $60K'), ['Income_Category','Avg_Utilization_Ratio']]

morethansixty.info()

"""used a violin plot to capture the density of categories side by side"""

sns.violinplot(x=lessthansixty['Income_Category'],y=lessthansixty['Avg_Utilization_Ratio'], data=lessthansixty)
plt.title('<=60K and Utilization Ratio')
plt.xlabel('Income')
plt.ylabel('Utilization Ratio')
plt.show()

"""The plot above shows inverse densities, which indicates that the average Util_Ratio will lie between 0.3-0.6"""

lessthansixty.describe()
# individuals making less than or equal to 60k has a mean utilization ratio of 0.39

sns.violinplot(x=morethansixty['Income_Category'],y=morethansixty['Avg_Utilization_Ratio'], data=morethansixty)
plt.title('>60K and Utilization Ratio')
plt.xlabel('Income')
plt.ylabel('Utilization Ratio')
plt.show()

"""The plot above shows that all the categories are concentrated below 0.2. This indicates an average between 0.0-0.4."""

morethansixty.describe()
# individuals making more than 60k have a mean utilization ratio of 0.19

stats.ttest_ind(lessthansixty.Avg_Utilization_Ratio, morethansixty.Avg_Utilization_Ratio,equal_var=False)
# Since the p-value is less than alpha=0.05, we can reject the hypothesis that >=60k and 60k< have equal averages.

"""## Hypothesis 3: I hypothesize that there is a correlation between age and total transaction counts

I suspect that the younger someone is, the higher the annual transaction cost will be. Younger people are stereotyped to be more reckless with spending. 

Should the bank target credit card advertisements more towards the younger demographic in order to capture higher transaction fees?
"""

cc_df_hyp3 = cc_df_new[['Customer_Age','Total_Trans_Ct']].sample(frac=0.2)
# Create a new data frame with only customer age and transaction count
# Only included a 20% random sample of the data for better graphical representation
cc_df_hyp3.info()

"""Put plots into bins to eliminate the high amount of data points"""

sns.regplot(x=cc_df_hyp3.Customer_Age,y=cc_df_hyp3.Total_Trans_Ct,x_bins=50)
plt.title('Age and Annual Transaction Count')
plt.xlabel('Age')
plt.ylabel('Annual Transaction Count')
plt.show()

"""use pearson R function to find the correlation between two independent and normally distributed continuous variables."""

stats.pearsonr(cc_df_new.Customer_Age,cc_df_new.Total_Trans_Ct)
# the Pearson R test indicates a weak negative correlation between age and annual transaction count
