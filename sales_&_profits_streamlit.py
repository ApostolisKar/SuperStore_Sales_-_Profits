import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
import seaborn as sns
import streamlit as st
from matplotlib.ticker import PercentFormatter
import matplotlib.ticker as tkr
import plotly.express as px


# load the main file
df = pd.read_csv('Superstore.csv', encoding='latin-1')

df['Ship Date'] = pd.to_datetime(df['Ship Date'])
df['Order Date'] = pd.to_datetime(df['Order Date'])

df['Order Date'].agg(['min','max'])
df.describe()

df['Profit_%']=df['Profit']/df['Sales']

df['Order_Year']=df['Order Date'].dt.year
df['Order_Month']=df['Order Date'].dt.month
df['Order_Day']=df['Order Date'].dt.day


# create time_difference to capture part of data below
df["time_difference"] = df['Order Date'].max() - df["Order Date"]

#last year
df["last_year"] = df["time_difference"] <= timedelta(days=365)

df_RFM = df[df['last_year']==True].copy()

df_RFM['Recency'] = (df_RFM['Order Date'].max() - df_RFM['Order Date']).dt.days

df_RFM_Sales = df_RFM.groupby(['Customer ID']).agg({'Recency': np.min,
                                           'Order Date':pd.Series.nunique,
                                           'Sales':np.sum}).reset_index()
#Rename columns
df_RFM_Sales.rename(columns={'Recency':'Recency','Order Date':'Frequency','Sales':'Monetary'},inplace= True)


# Create Scores RFM scores based on quantiles of distribution

#Date from customer's last purchase.The nearest date gets 4 and the furthest date gets 1.
df_RFM_Sales["recency_score"] = pd.qcut(df_RFM_Sales['Recency'].rank(method="first"),
                                  4,
                                  labels=[4, 3, 2, 1])

# Total number of purchases.The least frequency gets 1 and the maximum frequency gets 4.
df_RFM_Sales["frequency_score"] = pd.qcut(df_RFM_Sales["Frequency"].rank(method="first"),
                                    4,
                                    labels=[1, 2, 3, 4])

#Total spend by the customer.The least money gets 1, the most money gets 4.
df_RFM_Sales["monetary_score"] = pd.qcut(df_RFM_Sales["Monetary"].rank(method="first"),
                                   4,
                                   labels=[1, 2, 3, 4])


df_RFM_Sales["RFM_Segment"] = df_RFM_Sales["recency_score"].astype(str) + df_RFM_Sales[
    "frequency_score"].astype(str) + df_RFM_Sales["monetary_score"].astype(str)

df_RFM_Sales['RFM_Score'] = df_RFM_Sales[[
    'recency_score', 'frequency_score', 'monetary_score'
]].sum(axis=1)


segt_map = {
    r'[3-4][3-4]4': 'VIP',
    r'[2-3-4][1-2-3-4]4': 'Top Recent',
    r'1[1-2-3-4]4': 'Top at Risk ',

    
    
    r'[3-4][3-4]3': 'High Promising',
    r'[2-3-4][1-2]3': 'High New',
    r'2[3-4]3': 'High Loyal',

    
    
    r'[3-4][3-4]2': 'Medium Potential',
    r'[2-3-4][1-2]2': 'Medium New',
    r'2[3-4]2': 'Medium Loyal',

    
    
    r'4[1-2-3-4]1': 'Low New',
    r'[2-3][1-2-3-4]1': 'Low Loyal',
    
    r'1[1-2-3-4][1-2-3]': 'Need Activation'
}
df_RFM_Sales['Segment_labels'] = df_RFM_Sales['RFM_Segment']
df_RFM_Sales['Segment_labels'] = df_RFM_Sales['Segment_labels'].replace(segt_map, regex=True)



seg_pareto = df_RFM_Sales.groupby(["Segment_labels"]).agg({'Monetary': np.sum,
                 
                                                           "Customer ID": pd.Series.nunique}).reset_index()

seg_pareto["Monetary%"] = seg_pareto["Monetary"]/seg_pareto["Monetary"].sum()
seg_pareto = seg_pareto.sort_values(by=['Monetary%'], ascending=False)
seg_pareto["CumulativePercentage"] = (seg_pareto["Monetary"].cumsum()/ 
                                      seg_pareto["Monetary"].sum()*100).round(2)
seg_pareto["CumulativeSum"] = (seg_pareto["Customer ID"].cumsum()/ 
                                      seg_pareto["Customer ID"].sum()*100).round(2)

seg_pareto.reset_index()


df_RFM_Profit = df_RFM.groupby(['Customer ID']).agg({'Recency': np.min,
                                           'Order Date':pd.Series.nunique,
                                           'Profit':np.sum}).reset_index()


#Rename columns
df_RFM_Profit.rename(columns={'Recency':'Recency','Order Date':'Frequency','Profit':'Monetary'},inplace= True)


# Create Scores RFM scores based on quantiles of distribution

#Date from customer's last purchase.The nearest date gets 4 and the furthest date gets 1.
df_RFM_Profit["recency_score"] = pd.qcut(df_RFM_Profit['Recency'].rank(method="first"),
                                  4,
                                  labels=[4, 3, 2, 1])

# Total number of purchases.The least frequency gets 1 and the maximum frequency gets 4.
df_RFM_Profit["frequency_score"] = pd.qcut(df_RFM_Profit["Frequency"].rank(method="first"),
                                    4,
                                    labels=[1, 2, 3, 4])

#Total spend by the customer.The least money gets 1, the most money gets 4.
df_RFM_Profit["monetary_score"] = pd.qcut(df_RFM_Sales["Monetary"].rank(method="first"),
                                   4,
                                   labels=[1, 2, 3, 4])


df_RFM_Profit["RFM_Segment"] = df_RFM_Profit["recency_score"].astype(str) + df_RFM_Profit[
    "frequency_score"].astype(str) + df_RFM_Profit["monetary_score"].astype(str)

df_RFM_Profit['RFM_Score'] = df_RFM_Profit[[
    'recency_score', 'frequency_score', 'monetary_score'
]].sum(axis=1)


df_RFM_Profit['Segment_labels'] = df_RFM_Profit['RFM_Segment']
df_RFM_Profit['Segment_labels'] = df_RFM_Profit['Segment_labels'].replace(segt_map, regex=True)


seg_pareto_profit = df_RFM_Profit.groupby(["Segment_labels"]).agg({'Monetary': np.sum,
                 
                                                           "Customer ID": pd.Series.nunique}).reset_index()


seg_pareto_profit["Monetary%"] = seg_pareto_profit["Monetary"]/seg_pareto_profit["Monetary"].sum()
seg_pareto_profit = seg_pareto_profit.sort_values(by=['Monetary%'], ascending=False)
seg_pareto_profit["CumulativePercentage"] = (seg_pareto_profit["Monetary"].cumsum()/ 
                                      seg_pareto_profit["Monetary"].sum()*100).round(2)
seg_pareto_profit["CumulativeSum"] = (seg_pareto_profit["Customer ID"].cumsum()/ 
                                      seg_pareto_profit["Customer ID"].sum()*100).round(2)

seg_pareto_profit.reset_index()

x = seg_pareto_profit.head(3)

st.title("Pareto Rule")


x


#define aesthetics for plot
color1 = 'steelblue'
color2 = 'red'
#line_size = 1

#create basic bar plot
fig, ax = plt.subplots()
ax.bar(seg_pareto_profit['Segment_labels'], seg_pareto_profit['Monetary'], color=color1)


#add cumulative percentage line to plot
ax2 = ax.twinx()
ax2.plot(seg_pareto_profit['Segment_labels'], seg_pareto_profit['CumulativePercentage'], color=color2, marker="D", ms=1)
ax2.yaxis.set_major_formatter(PercentFormatter())
ax2.axhline(80, color = "#008878", linestyle = "dashed", alpha = 1 )


    
#specify axis colors
ax.tick_params(axis='y', colors=color1, labelsize= 8)
#ax.set_xticklabels([])
ax2.tick_params(axis='y', colors=color2, labelsize= 8)

ax.tick_params(axis='x', labelsize= 6)
ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda y,  p: format(int(y), ',')))

#ax.tick_params(axis='x', labelsize= )
#display Pareto chart
plt.show()

fig





# EDA
# Top 10 States by Sales

df.groupby('State')['Sales'].sum().sort_values(ascending=True).tail(10).plot.barh()

# Top 10 States by Discount

df.groupby('State')['Discount'].sum().sort_values(ascending=True).tail(10).plot.barh()

# Top 10 States by Profit
df.groupby('State')['Profit'].sum().sort_values(ascending=True).tail(10).plot.barh()





# Sales and Profit scatter plot

#fig1 = sns.scatterplot(data=df,x='Sales',y='Profit')
#plt.xlim(0, 26000)
#plt.ylim(-7500, 9500)



#plt.figure(figsize = (10,15))
#fig1 = px.scatter(df, x='Sales', y='Profit')
#st.pyplot()

#fig, ax = plt.subplots()
#px.scatter(df, x='Sales', y='Profit')
# other plotting actions...
#st.pyplot(fig)

st.title("Sales vs. Profit Scatter Chart")
st.scatter_chart(data=df, x='Sales', y='Profit',use_container_width=True, )


st.title("Sales vs Discount Scatter Chart") 
fig, ax = plt.subplots() 
ax.scatter(df['Discount'], df['Sales'], label="Sales vs Discount") 
ax.set_xlabel("Discount") 
ax.set_ylabel("Sales") 
ax.set_title("Sales vs Discount") 
ax.legend() 
st.pyplot(fig)



st.title("Profit vs Discount Scatter Plot") 
fig, ax = plt.subplots() 
sns.scatterplot(x=df['Profit'], y=df['Discount'], ax=ax, label="Profit vs Discount") 
ax.set_title("Profit vs Discount") 
ax.legend() 
st.pyplot(fig)




# Sales 
st.header("Top 10 States by Sales") 
sales_data = df.groupby('State')['Sales'].sum().sort_values(ascending=True).tail(10) 
fig, ax = plt.subplots() 
sales_data.plot(kind='barh', ax=ax) 
st.pyplot(fig)



# Profit 
st.header("Top 10 States by Profit") 
profit_data = df.groupby('State')['Profit'].sum().sort_values(ascending=True).tail(10) 
fig, ax = plt.subplots() 
profit_data.plot(kind='barh', ax=ax)
st.pyplot(fig)

#Discount 
st.header("Top 10 States by Discount") 
discount_data = df.groupby('State')['Discount'].sum().sort_values(ascending=True).tail(10) 
fig, ax = plt.subplots() 
discount_data.plot(kind='barh', ax=ax) 
st.pyplot(fig)
