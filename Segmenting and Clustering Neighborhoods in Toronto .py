#!/usr/bin/env python
# coding: utf-8

# In[55]:


import requests
import pandas as pd


# ### Using pandas to obtain the table in wikipedia article

# In[56]:


website_url = requests.get("https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M").text
df = pd.read_html(website_url)[0]


# ### The table is saved in " df "

# In[57]:


df.head()


# ### Dropping all the rows that do not have the Borough value assigned

# In[58]:


indexList = df[(df["Borough"] == "Not assigned")].index


# In[59]:


df.drop(index=indexList, inplace=True)


# ### Renaming rows with not assigned Neigbourhood values to their Borough names

# In[60]:


df[df["Neighbourhood"] == "Not assigned"]


# In[61]:


df.loc[10]["Neighbourhood"] = df.loc[10]["Borough"]


# In[62]:


df.loc[10]


# ### Using shape to print the number of rows and columns

# In[63]:


df.shape


# In[64]:


df.head()


# ### Combining rows with the same Postcode values using group by 

# In[65]:


df = df.groupby(["Postcode", "Borough"], sort=False).agg(','.join)
df.head()


# ### Converting multi index dataframe to single index 

# In[66]:


df = df.reset_index(level=[0,1])
df.head()


# ### Renaming Postcode to PostalCode

# In[67]:


df.rename(columns = {"Postcode":"PostalCode"}, inplace=True)
df.head()


# ### Checking the shape of the dataframe

# In[68]:


df.shape


# ### Getting the latitude and longitude of the PostalCode 

# In[69]:


location_data = pd.read_csv(r"Geospatial_Coordinates.csv")
location_data.rename(columns={"Postal Code" : "PostalCode"}, inplace=True)
location_data.head()


# In[70]:


location_data.shape


# ### Merging df and location_data

# In[71]:


df_combined = pd.merge(left=df, right=location_data, left_on='PostalCode', right_on='PostalCode')
df_combined.head()


# In[72]:


import folium
from geopy.geocoders import Nominatim


# ### Finding the coordinates of Toronto

# In[73]:


geolocator = Nominatim(user_agent="toronto_explorer", timeout=3)
location = geolocator.geocode("Toronto")
latitude = location.latitude 
longitude = location.longitude
print("The latitude is {} and longitude is {}".format(latitude, longitude))


# ### Creating a map of Toronto with neighbourhoods superimposed

# In[74]:


map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)
for i, j in zip(df_combined["Latitude"], df_combined["Longitude"]):
    folium.CircleMarker([i, j], radius=5).add_to(map_toronto)
map_toronto


# ### Selecting only those rows which contain the Borough "Toronto"

# In[75]:


df_toronto = df_combined[df_combined.Borough.str.contains("Toronto")]
df_toronto.reset_index(inplace=True, drop=True)
df_toronto.head()


# ### Visualizing the new dataframe

# In[76]:


map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)
for i, j in zip(df_toronto["Latitude"], df_toronto["Longitude"]):
    folium.CircleMarker([i, j], radius=5).add_to(map_toronto)
map_toronto


# ### Foursquare credentials and version

# In[77]:


CLIENT_ID = 'I30QFOWBDCH5BSYJEBQGHON3RMAOQWO2HJPD1BKP4RRA1ATH' 
CLIENT_SECRET = 'S5B3U0GBMGGVPZ4VV3JHXEPY1Y4MGKBYG0C0ZHP1KVATPTZS' 
VERSION = '20180605'


# ### Exploring the first neighbourhood in the dataframe

# In[78]:


df_toronto["Neighbourhood"][0]


# ### Get the neighbourhood's latitude and longitude

# In[79]:


neighborhood_latitude = df_toronto["Latitude"][0] 
neighborhood_longitude = df_toronto["Longitude"][0] 


# ### Get top 100 venues within a 500 meters radius of Harbourfront

# In[80]:


url = "https://api.foursquare.com/v2/venues/explore?ll={},{}&radius=500&client_id={}&client_secret={}&v={}&limit=200".format(neighborhood_latitude, neighborhood_longitude, CLIENT_ID, CLIENT_SECRET, VERSION)
results = requests.get(url).json()


# ### Reusing functions from the previous notebook

# In[81]:


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[82]:


from pandas.io.json import json_normalize
venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()


# ### Explore Neighbourhoods in Toronto
# ### Let's create a function to repeat the same process to all the neighborhoods in Manhattan

# In[83]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        LIMIT=200
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighbourhood', 
                  'Neighbourhood Latitude', 
                  'Neighbourhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# #### Now write the code to run the above function on each neighborhood and create a new dataframe called *toronto_venues*.

# In[84]:


toronto_venues = getNearbyVenues(names=df_toronto['Neighbourhood'],
                                   latitudes=df_toronto['Latitude'],
                                   longitudes=df_toronto['Longitude']
                                  )


# ## Analyze each neighbourhood

# In[85]:


# one hot encoding
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
toronto_onehot['Neighbourhood'] = toronto_venues['Neighbourhood'] 

# move neighborhood column to the first column
fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]

toronto_onehot.head()


# #### Next, let's group rows by neighborhood and by taking the mean of the frequency of occurrence of each category

# In[86]:


toronto_grouped = toronto_onehot.groupby('Neighbourhood').mean().reset_index()
toronto_grouped.head()


# #### Let's print each neighborhood along with the top 5 most common venues

# In[87]:


num_top_venues = 5

for hood in toronto_grouped['Neighbourhood']:
    print("----"+hood+"----")
    temp = toronto_grouped[toronto_grouped['Neighbourhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# #### Let's put that into a pandas dataframe
# First, let's write a function to sort the venues in descending order.

# In[88]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# Now let's create the new dataframe and display the top 10 venues for each neighborhood.

# In[89]:


import numpy as np
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighbourhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighbourhoods_venues_sorted = pd.DataFrame(columns=columns)
neighbourhoods_venues_sorted['Neighbourhood'] = toronto_grouped['Neighbourhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighbourhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighbourhoods_venues_sorted.head()


# ## 4. Cluster Neighborhoods
# Run *k*-means to cluster the neighborhood into 5 clusters.

# In[90]:


from sklearn.cluster import KMeans
# set number of clusters
kclusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighbourhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# Let's create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.

# In[91]:


# add clustering labels
neighbourhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_merged = df_toronto

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighbourhoods_venues_sorted.set_index('Neighbourhood'), on='Neighbourhood')

toronto_merged.head() # check the last columns!


# Finally, let's visualize the resulting clusters

# In[96]:


import matplotlib.cm as cm
import matplotlib.colors as colors

# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighbourhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters

