import pandas as pd
import numpy as np
import random
import networkx as nx                                                                                                                                                           
import time as time
from collections import Counter
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from scipy import spatial
import matplotlib.pyplot as plt
import scipy.sparse.linalg
from scipy.spatial import distance

from pyspark.sql.types import *
from pyspark.sql.functions import *
from  pyspark.sql.types import TimestampType

import mlflow
import mlflow.pyfunc

# COMMAND ----------


# COMMAND ----------


# get last view of an item by a user based on timestamp
views_full = sqlContext.table("foldername.item_views_20210219").cache()
views_full = views_full.withColumn("timestamp", views_full["timestamp"].cast(TimestampType()))
views_full.createOrReplaceTempView('views_full')

# COMMAND ----------

views_dedupe = spark.sql('''
select QUID, PRODUCT_NBR, PRODUCT_DESC, MAX(`timestamp`) as `timestamp`
from views_full
where QUID is not null and PRODUCT_NBR is not null
group by QUID, PRODUCT_NBR, PRODUCT_DESC''').cache()

# COMMAND ----------

views_dedupe

# COMMAND ----------

views_dedupe.show()

# COMMAND ----------

views_dedupe = views_dedupe.toPandas()

# COMMAND ----------

views_dedupe.sort_values(by=['PRODUCT_NBR']).head(10)

# COMMAND ----------

print(views_dedupe.groupby(['PRODUCT_DESC','PRODUCT_NBR'])['PRODUCT_NBR'].agg(['count']))

# COMMAND ----------

groupby_df = (views_dedupe.groupby(['PRODUCT_DESC','PRODUCT_NBR'])['PRODUCT_NBR'].agg(['count']).sort_values(by='count', ascending=False).reset_index().drop_duplicates('PRODUCT_DESC', keep='first'))

# COMMAND ----------

groupby_df

# COMMAND ----------

df = groupby_df[:int(len(groupby_df) * .1)]

# COMMAND ----------

groupby_df.rename(columns = {'count': 'score'},inplace=True)
training_data = groupby_df.sort_values(['score', 'PRODUCT_NBR'], ascending = [0,1]) 
training_data['Rank'] = training_data['score'].rank(ascending=0, method='first') 
recommendations = training_data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC recommendations based on top products viewed 

# COMMAND ----------

print(recommendations)

# COMMAND ----------

def recommend(id):     
    recommend_products = recommendations 
    recommend_products['PRODUCT_NBR'] = id 
    column = recommend_products.columns.tolist() 
    column = column[-1:] + column[:-1] 
    recommend_products = recommend_products[column] 
    return recommend_products 

print(recommend(11))

# COMMAND ----------

counts = df['QUID'].value_counts()
data = df[df['QUID'].isin(counts[counts >= 50].index)]
data.groupby('PRODUCT_NBR')['count'].mean().sort_values(ascending=False) 
final_ratings = data.pivot(index = 'QUID', columns ='PRODUCT_NBR', values = 'count').fillna(0)

num_of_ratings = np.count_nonzero(final_ratings)
possible_ratings = final_ratings.shape[0] * final_ratings.shape[1]
density = (num_of_ratings/possible_ratings)
density *= 100
final_ratings_T = final_ratings.transpose()

grouped = data.groupby('QUID').agg({'QUID': 'count'}).reset_index()
grouped.rename(columns = {'QUID': 'score'},inplace=True)
training_data = grouped.sort_values(['score', 'PRODUCT_NBR'], ascending = [0,1]) 
training_data['Rank'] = training_data['score'].rank(ascending=0, method='first') 
recommendations = training_data.head()

# COMMAND ----------

def recommend(id):     
    recommend_products = recommendations 
    recommend_products['user_id'] = id 
    column = recommend_products.columns.tolist() 
    column = column[-1:] + column[:-1] 
    recommend_products = recommend_products[column] 
    return recommend_products 

print(recommend(11))

# COMMAND ----------



# COMMAND ----------

views_dedupe['PRODUCT_DESC'].nlargest(n=5)

# COMMAND ----------
