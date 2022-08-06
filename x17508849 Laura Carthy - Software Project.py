#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd


# In[39]:


import numpy as np


# In[40]:


df=pd.read_csv("NYPD_Arrests_Data__Historic_.csv")


# In[41]:


df.head()


# In[42]:


df.shape


# In[43]:


df.describe


# In[44]:


df.info()


# In[45]:


df.isnull().sum()


# In[46]:


df = df.dropna()


# In[47]:


print(df)


# In[48]:


df.isnull().sum()


# In[49]:


import matplotlib.pyplot as plt

AGE_GROUP = [15,18,20,22,23,25,27,30,32,36,40,45,48,51,54,59,62,64,68,71]

range = (0,100)
bins = 10

plt.hist(AGE_GROUP, bins, range, color = 'orange', histtype = 'bar', rwidth = 0.8)

plt.xlabel('age ranges')
plt.ylabel('number of people')
plt.title('Histogram 0.1')
plt.show()


# In[50]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(x='PERP_SEX', data=df)
plt.show()


# In[57]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20,20)) 

sns.histplot(x='PERP_RACE', data=df)
plt.show()


# In[67]:


import seaborn as sns
import matplotlib.pyplot as plt
  
plt.figure(figsize=(20,20)) 
    
sns.scatterplot( x="PERP_RACE", y='ARREST_PRECINCT', data=df,
                hue='PERP_SEX', size='ARREST_PRECINCT')
  
# Placing Legend outside the Figure
plt.legend(bbox_to_anchor=(1, 1), loc=2)
  
plt.show()


# In[73]:


pip install prettytable


# In[74]:


#from prettytable import PrettyTable
 
# Specify the Column Names while initializing the Table
#myTable = PrettyTable(["PERP_RACE", "AGE_GROUP", "PERP_SEX", "ARREST_PRECINCT"])
 
# Add rows
#myTable.add_row(["", "", "", ""])
#myTable.add_row(["", "", "", ""])
#myTable.add_row(["", "", "", ""])
#myTable.add_row(["", "", "", ""])
#myTable.add_row(["", "", "", ""])
#myTable.add_row(["", "", "", ""])
#myTable.add_row(["", "", "", ""])
 
#print(myTable)


# In[77]:


pip install Shapely


# In[78]:


pip install matplotlib


# In[1]:


pip install geopandas


# In[12]:


import pandas as pd

df = pd.read_csv(f'NYPD_Arrests_Data__Historic_.csv')
df


# In[13]:


import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt

geometry = [Point(xy) for xy in zip(df['Longitude'], df["Latitude"])]
gdf = GeoDataFrame(df, geometry=geometry)


world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
gdf.plot(ax=world.plot(figsize=(15, 15)), marker='o', color='red', markersize=15);


# In[ ]:


import plotly.express as px
import pandas as pd

fig = px.scatter_geo(df,lat='Latitude',lon='Longitude', hover_name="PERP_RACE")
fig.update_layout(title = 'Races of perps arrested in these locations', title_x=0.5)
fig.show()


# In[15]:


corr = df.corr()

corr.style.background_gradient(cmap='coolwarm')


# In[ ]:





# In[ ]:





# In[ ]:




