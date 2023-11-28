import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

        
customers = pd.read_csv('olist_customers_dataset.csv')
sellers = pd.read_csv('olist_sellers_dataset.csv')
geolocation = pd.read_csv('olist_geolocation_dataset.csv')
order_items = pd.read_csv('olist_order_items_dataset.csv')
order_payments = pd.read_csv('olist_order_payments_dataset.csv')
order_reviews = pd.read_csv('olist_order_reviews_dataset.csv')
orders = pd.read_csv('olist_orders_dataset.csv')
products = pd.read_csv('olist_products_dataset.csv')
product_category_name = pd.read_csv('product_category_name_translation.csv')   

order_items_products = pd.merge(order_items,products,on='product_id',how='outer')
order_items_products_sellers = pd.merge(order_items_products,sellers, on='seller_id',how='outer')
two_order_items_products_sellers = pd.merge(order_items_products_sellers,orders,on='order_id',how='outer')
two_order_items_products_sellers_customer = pd.merge(two_order_items_products_sellers,customers,on='customer_id',how='outer')
two_order_items_products_sellers_customer_reviews = pd.merge(two_order_items_products_sellers_customer,order_reviews,on='order_id',how='outer')
df = pd.merge(two_order_items_products_sellers_customer_reviews,order_payments,on='order_id',how='outer')

#Information about dataframe
df.head()     

df.describe()

df.shape

df.info()

#Counting missing values
null_values = df.isnull().sum()
null_values

#Check for duplicates row
duplicated_rows = df.duplicated().sum()
duplicated_rows

#remove rows with missing values in the specified subset of columns, and make new Data frame with the rows containing missing values in the specific columns removed
df_2 = df.dropna(subset = ['shipping_limit_date', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date', 'review_creation_date', 'review_answer_timestamp','order_purchase_timestamp', 'order_approved_at'])

# counts of missing values in each column after the removal of specific rows.
null_values_df_2 = df_2.isnull().sum()
null_values_df_2

df_2.shape

from datetime import datetime

# Calculate purchase-delivery difference
intermediate_time = df_2['order_delivered_customer_date'].apply(lambda x: datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S").date()) - df_2['order_purchase_timestamp'].apply(lambda x: datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S").date())
df_2.loc[:, 'purchase-delivery difference'] = intermediate_time.apply(lambda x: x.days)

# Calculate estimated-actual delivery difference
intermediate_time = df_2['order_estimated_delivery_date'].apply(lambda x: datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S").date()) - df_2['order_delivered_customer_date'].apply(lambda x: datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S").date())
df_2.loc[:, 'estimated-actual delivery difference'] = intermediate_time.apply(lambda x: x.days)

df_2.info()

#Copy of dataframe2 for other uses
df_3 = df_2.copy()

# Fill missing values in 'product_category_name' with the mode
df_3.loc[:, 'product_category_name'].fillna(value=df_2['product_category_name'].mode()[0], inplace=True)

# Fill missing values in 'product_name_lenght' with the mode
df_3.loc[:, 'product_name_lenght'].fillna(value=df_2['product_name_lenght'].mode()[0], inplace=True)

# Fill missing values in 'product_description_lenght' with the median
df_3.loc[:, 'product_description_lenght'].fillna(value=df_2['product_description_lenght'].median(), inplace=True)

# Fill missing values in 'product_photos_qty' with the mode
df_3.loc[:, 'product_photos_qty'].fillna(value=df_2['product_photos_qty'].mode()[0], inplace=True)

# Fill missing values in 'product_weight_g' with the mode
df_3.loc[:, 'product_weight_g'].fillna(value=df_2['product_weight_g'].mode()[0], inplace=True)

# Fill missing values in 'product_length_cm' with the mode
df_3.loc[:, 'product_length_cm'].fillna(value=df_2['product_length_cm'].mode()[0], inplace=True)

# Fill missing values in 'product_height_cm' with the mode
df_3.loc[:, 'product_height_cm'].fillna(value=df_2['product_height_cm'].mode()[0], inplace=True)

# Fill missing values in 'product_width_cm' with the mode
df_3.loc[:, 'product_width_cm'].fillna(value=df_2['product_width_cm'].mode()[0], inplace=True)

# Fill missing values in 'review_comment_message' with 'indisponível'
df_3.loc[:, 'review_comment_message'].fillna(value='indisponível', inplace=True)

# Fill missing values in 'review_comment_title' with 'indisponível'
df_3.loc[:, 'review_comment_title'].fillna(value='indisponível', inplace=True)

#Check for null values
null_values_df_3 = df_3.isnull().sum()
null_values_df_3

df_3.describe()

#Remove rows with missing values
df_4 = df_3.dropna(subset = ['payment_sequential','payment_type', 'payment_installments','payment_value' ])
null_values_df_4 = df_4.isnull().sum()
null_values_df_4

df_4.shape

df_4.describe()

df_4.loc['price_category'] = df_4['price'].apply(lambda x:'expensive' if x>=132.9 else ('affordable' if x>=39.9 and x<132.9 else 'cheap'))

df_4['order_status'].value_counts()
(len(df_4[df_4["order_status"] == "canceled"]) / len(df_4["order_status"])) * 100

df_5 = df_4.merge(product_category_name, on='product_category_name', how='left')

# Replace the original 'product_category_name' column with the translated names
df_5['product_category_name'] = df_5['product_category_name_english']

df_5.columns

#average 'purchase-delivery difference' for each product category, providing insights into the delivery time for different types of products.
category_mean_diff = df_5.groupby('product_category_name')['purchase-delivery difference'].mean()

# Group the data by 'product_category_name' and count the number of purchases per category
most_bought_category = df_5['product_category_name'].value_counts()

# Create a bar plot
plt.figure(figsize=(12, 6))
most_bought_category.plot(kind='bar')
plt.xlabel('Product Category')
plt.ylabel('Number of Orders')
plt.title('Most Purchased Product Categories')
plt.show()

# Group by 'geolocation_city' and count the number of orders in each city
city_order_counts = df_4['customer_state'].value_counts()

# Create a bar plot
plt.figure(figsize=(12, 6))
city_order_counts.plot(kind='bar')
plt.xlabel('State')
plt.ylabel('Number of Orders')
plt.title('Number of Orders by state')
plt.xticks(rotation=90) 
plt.show()

# Convert 'order_purchase_timestamp' to datetime format
df_4['order_purchase_timestamp'] = pd.to_datetime(df_4['order_purchase_timestamp'])

# Extract date from timestamp
df_4['order_date'] = df_4['order_purchase_timestamp'].dt.date

# Group by 'order_date' and count the number of orders per day
daily_order_counts = df_4['order_date'].value_counts().sort_index()

# Create a line plot
plt.figure(figsize=(12, 6))
daily_order_counts.plot(kind='line', marker='o')
plt.xlabel('Order Date')
plt.ylabel('Number of Orders')
plt.title('Number of Orders Over Time')
plt.show()

