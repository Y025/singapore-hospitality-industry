import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static

#Add function for categorical legend
def add_categorical_legend(folium_map, title, colors, labels):
    if len(colors) != len(labels):
        raise ValueError("colors and labels must have the same length.")

    color_by_label = dict(zip(labels, colors))

    legend_categories = ""
    for label, color in color_by_label.items():
        legend_categories += f"<li><span style='background:{color}'></span>{label}</li>"

    legend_html = f"""
    <div id='maplegend' class='maplegend'>
      <div class='legend-title'>{title}</div>
      <div class='legend-scale'>
        <ul class='legend-labels'>
        {legend_categories}
        </ul>
      </div>
    </div>
    """
    script = f"""
        <script type="text/javascript">
        var oneTimeExecution = (function() {{
                    var executed = false;
                    return function() {{
                        if (!executed) {{
                             var checkExist = setInterval(function() {{
                                       if ((document.getElementsByClassName('leaflet-top leaflet-right').length) || (!executed)) {{
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].style.display = "flex"
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].style.flexDirection = "column"
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].innerHTML += `{legend_html}`;
                                          clearInterval(checkExist);
                                          executed = true;
                                       }}
                                    }}, 100);
                        }}
                    }};
                }})();
        oneTimeExecution()
        </script>
      """


    css = """

    <style type='text/css'>
      .maplegend {
        z-index:9999;
        float:right;
        background-color: rgba(255, 255, 255, 1);
        border-radius: 5px;
        border: 2px solid #bbb;
        padding: 10px;
        font-size:12px;
        positon: relative;
      }
      .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
      .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
      .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
      .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 0px solid #ccc;
        }
      .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
      .maplegend a {
        color: #777;
        }
    </style>
    """

    folium_map.get_root().header.add_child(folium.Element(script + css))

    return folium_map


#Load the data
df1 = pd.read_csv("datas/listings_1.csv")
df2 = pd.read_csv("datas/listings_2.csv")
df3 = pd.read_csv("datas/listings_3.csv")
df4 = pd.read_csv("datas/listings_4.csv")
df = pd.concat([df1, df2, df3, df4], axis = 0)

#Reset the index
df.reset_index(inplace = True, drop=True)

# Changing these columns into datetime format
date_columns = ['host_since', 'calendar_updated',
                'calendar_last_scraped', 'first_review',
                'last_review']
df[date_columns] = df[date_columns].apply(pd.to_datetime)

# Changing price columns to float
df['price'] = df['price'].apply(lambda x: str(x).replace('$',''))
df['price'] = df['price'].apply(lambda x: str(x).replace(',','')).astype(float)

#Dropping these columns out
df = df.drop(columns = ['bathrooms', 'calendar_updated', 'license'])
df = df.dropna(subset=['first_review'])
df = df.dropna(subset=['bedrooms'])

# Selecting the significant columns only
significant_columns = ['id','name','description',
 'host_id',
'host_identity_verified', 'neighbourhood',
'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'latitude',
'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms_text',
 'bedrooms', 'beds', 'amenities', 'price',
'minimum_nights', 'has_availability',
'availability_365','number_of_reviews',
'number_of_reviews_ltm', 'number_of_reviews_l30d',
'review_scores_rating', 'review_scores_accuracy',
'review_scores_cleanliness', 'review_scores_checkin',
'review_scores_communication', 'review_scores_location',
'review_scores_value', 'instant_bookable',
'calculated_host_listings_count', 'reviews_per_month']

df_final = df[significant_columns]

# We will fill missing values with 0 for review_score and friends
missing_columns = ['review_scores_accuracy',
'review_scores_cleanliness', 'review_scores_checkin',
'review_scores_communication', 'review_scores_location',
'review_scores_value']

df_final[missing_columns] = df_final[missing_columns].fillna(0)

# We will fill the beds missing values with the mean
beds_mean = math.floor(df_final['beds'].mean())
df_final['beds'] = df_final['beds'].fillna(beds_mean)

# We will check if the host has a name to indicate if the host is verified
host_id_unv = df_final[df_final['host_identity_verified'].isna()]['host_id']
unv_host = df[df['host_id'].isin(host_id_unv)]['host_name']
Nancy_id = int(df[df['host_name']=='Nancy']['host_id'].unique())
Abigail_id = int(df[df['host_name']=='Abigail']['host_id'].unique())
# Get the index for abigail and nancy missing data
index_an = []
for index,values in host_id_unv.items():
  if (values == Nancy_id) | (values==Abigail_id):
    index_an.append(index)
# Change host verified to true for the obtained indexes
for i in index_an:
    df_final.loc[i, 'host_identity_verified'] = 't'
# Fill nan values with false
df_final['host_identity_verified'] = df_final['host_identity_verified'].fillna('f')

#Obtaining bathrooms data from the bathroom_text column using regex
df_final['bathrooms'] = df_final['bathrooms_text'].str.extract('(\d\W\d|\d*)', expand=True)
df_final['bathrooms'] = df_final.bathrooms.replace([""], 1)
df_final['bathrooms'] = df_final['bathrooms'].astype('float')
df_final['bathrooms'] = df_final['bathrooms'].fillna(round(df_final['bathrooms'].mean()))

#Lets drop the bathrroms_text column since we wont be using it further
df_final = df_final.drop(columns = ['bathrooms_text'])

# Eliminate Outliers on the price columns
q1 = np.quantile(df_final['price'], 0.25)
q3 = np.quantile(df_final['price'], 0.75)
iqr = q3 - q1
upper_boundary = q3 + 1.5*iqr
sub = df_final[df_final['price'] < upper_boundary]

# ---------------------------
# VISUALIZATION SECTION
# ---------------------------


#Visualization : Price vs Number of Reviews
fig1, ax1 = plt.subplots()
ax1 = sns.scatterplot(x='price', y='number_of_reviews', data=sub)
plt.title("Price Vs Number of Reviews")

#Visualization : Price vs Number of Reviews compared to room_type
fig2, ax2 = plt.subplots()
ax2 = sns.scatterplot(x='price', y='number_of_reviews', data=sub, hue = 'room_type')
plt.title("Price Vs Number of Reviews Per Room_Type")

#Visualization : Beds vs Number of Reviews based on mean
fig3, ax3 = plt.subplots()
ax3 = sns.barplot(x='beds', y = 'number_of_reviews', data =sub, ci=None)
plt.title("Beds Vs Number of Reviews on Mean")

#Visualization : Beds vs Number of Reviews based on count
fig4, ax4 = plt.subplots()
ax4 = sns.countplot(x='beds', data =sub)
plt.title("Beds Vs Number of Reviews on Count")

#Visualization : Accomodates vs Number of Reviews based on mean
fig5, ax5 = plt.subplots()
ax5 = sns.barplot(x='accommodates', y = 'number_of_reviews', data =sub, ci=None)
plt.title("Accommodates Vs Number of Reviews on Mean")

#Visualization : Accomodates vs Number of Reviews based on count
fig6, ax6 = plt.subplots()
ax6 = sns.countplot(x='accommodates', data=sub)
plt.title("Accommodates Vs Number of Reviews on Count")

#Visualization : reviews scores rating vs Number of Reviews
fig7, ax7 = plt.subplots()
ax7 = sns.scatterplot(x='review_scores_rating', y = 'number_of_reviews', data =sub, ci=None)
plt.title("Review Scores Rating Vs Number of Reviews")

#Visualization : minimum nights vs Number of Reviews on count
fig8, ax8 = plt.subplots()
ax8 = sns.scatterplot(x='minimum_nights', y = 'number_of_reviews', data =sub, ci=None)
plt.title("Minimum Nights Vs Number of Reviews on Count")

#Visualization : bathrooms vs Number of Reviews on mean
fig9, ax9 = plt.subplots()
ax9 = sns.barplot(x='bathrooms', y = 'number_of_reviews', data =sub, ci=None)
plt.title("Bathrooms Vs Number of Reviews on mean")

#Visualization : bathrooms vs Number of Reviews on count
fig10, ax10 = plt.subplots()
ax10 = sns.countplot(x='bathrooms', data=sub)
plt.title("Bathrooms Vs Number of Reviews on Count")

#Visualization : host_identity_verified vs Number of Reviews on mean
fig11, ax11 = plt.subplots()
ax11 = sns.barplot(x='host_identity_verified', y='number_of_reviews', data=sub, ci=None)
plt.title("Host Identity Verified Vs Number of Reviews on mean")

#Visualization : host_identity_verified vs Number of Reviews for each region on mean
df_nh_nor = sub.groupby(['neighbourhood_group_cleansed','host_identity_verified']).agg({'number_of_reviews':['mean','sum']})
df_nh_nor = df_nh_nor.unstack().reset_index()

df_nh_nor = df_nh_nor.melt(id_vars=['neighbourhood_group_cleansed'], var_name=['Verified_status', 'aggfunc','Host_Verified_Status'])
df_nh_nor.drop(columns=['Verified_status'], inplace = True)

fig12, ax12 = plt.subplots()
ax12 = sns.catplot(data=df_nh_nor,
            x='neighbourhood_group_cleansed',
            y='value',
            col= 'aggfunc',
            hue='Host_Verified_Status',
            kind='bar',sharex=False, sharey=False)
# ax12.axes[0,0].set_xticklabels(df_nh_nor['neighbourhood_group_cleansed'],rotation=90)
# ax12.axes[0,1].set_xticklabels(df_nh_nor['neighbourhood_group_cleansed'],rotation=90)
ax12.axes[0,0].set_ylabel('Number of Reviews')
ax12.axes[0,0].set_xlabel('Region')
ax12.axes[0,1].set_ylabel('Number of Reviews')
ax12.axes[0,1].set_xlabel('Region')
ax12.axes[0,0].set_title('Mean Number of Reviews per Region')
ax12.axes[0,1].set_title('Sum Number of Reviews per Region')
ax12 = plt.subplots_adjust(hspace=0.4, wspace=0.4)


#Visualization : isntant bookable vs Number of Reviewson mean
fig13, ax13 = plt.subplots()
ax13 = sns.barplot(x='instant_bookable', y = 'number_of_reviews', data =sub, ci=None)
plt.title("Instant Bookable Vs Number of Reviews on mean")

#Visualization : bedrooms vs Number of Reviews on mean
fig14, ax14 = plt.subplots()
ax14 = sns.barplot(x='bedrooms', y = 'number_of_reviews', data =sub, ci=None)
plt.title("Bedrooms Vs Number of Reviews on mean")

#Visualization : bedrooms vs Number of Reviews on count
fig15, ax15 = plt.subplots()
ax15 = sns.countplot(x='bedrooms', data=sub)
plt.title("Bedrooms Vs Number of Reviews on count")

#Visualization : availability 365 vs Number of Reviews
fig16, ax16 = plt.subplots()
ax16 = sns.scatterplot(x='availability_365', y = 'number_of_reviews', data =sub, ci=None)
plt.title("Availablity 365 Vs Number of Reviews")

#Visualization : room_type vs Number of Reviews on count
fig17, ax17 = plt.subplots()
ax17 = sns.histplot(x='room_type', data =sub, hue='room_type')
plt.title("Room Type Vs Number of Reviews on count")

#Visualization : room_type vs Number of Reviews on mean
fig18, ax18 = plt.subplots()
ax18 = sns.barplot(x='room_type', y='number_of_reviews',data =sub, ci=None)
plt.title("Room Type Vs Number of Reviews on mean")

#Visualization : Nuimber of Listings per Neighbourhood per Region
fig19, ax19 = plt.subplots()
plot_r_m = sub['neighbourhood_group_cleansed'].value_counts().sort_values(ascending=True)
ax19 = plot_r_m.plot.barh(figsize = (10,8), color = 'b', width = 0.5)
ax19 = plt.title('Number of Listings by Region', fontsize = 15)
ax19 = plt.xlabel('Number of Listings', fontsize = 12)

#Visualization : Nuimber of Listings per Neighbourhood per Neighbourhood
fig20, ax20 = plt.subplots()
plot_n_m = sub['neighbourhood_cleansed'].value_counts().sort_values(ascending=True)
plot_n_m = plot_n_m[-20:]
ax20 = plot_n_m.plot.barh(figsize = (10,8), color = 'b', width = 0.5)
ax20 = plt.title('Number of Listings on Singapore\'s Top 20 Neighbourhood', fontsize = 15)
ax20 = plt.xlabel('Number of Listings', fontsize = 12)

#Visualization : Listings price on map
fig21, ax21 = plt.subplots()
fig21 = plt.figure(figsize = (10,8))
i = plt.imread('datas/map2.png')
ax21 = plt.imshow(i, zorder=0, extent=[103.638880, 103.969660,  1.245350,1.488140])
ax21 = plt.gca()
ax21 = sub.plot(kind = 'scatter', x='longitude', y='latitude', label = 'availabilty_365',
         c='price', ax=ax21,
         cmap=plt.get_cmap('jet'), colorbar=True, alpha = 0.4, zorder=5)
ax21 = plt.title('Airbnb Listings Price Distribution', fontsize = 16)
ax21 = plt.legend()

# Feature Importance
list_parameters = ['host_identity_verified','room_type', 'accommodates','bathrooms',
                   'bedrooms', 'beds', 'price', 'minimum_nights', 'availability_365'
                   ,'review_scores_rating','instant_bookable' ]
subf = sub[list_parameters]
#Choose the target and independent datas
X = pd.get_dummies(subf)
y = sub[['number_of_reviews']]

# Split the data and apply standardization
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

#Fit the model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
importances = pd.DataFrame(data={
    'Attribute': X_train.columns,
    'Importance': model.coef_[0]
})
importances = importances.sort_values(by='Importance', ascending=False)
#Visualization : Feature Importance
fig22, ax22 = plt.subplots(figsize=(10,5))
ax22 = plt.bar(x=importances['Attribute'], height=importances['Importance'], \
    color=["#b80926", "#b80926", "#b80926", '#087E8B','#087E8B','#087E8B','#087E8B','#087E8B','#087E8B',\
    '#087E8B','#087E8B','#087E8B','#087E8B','#087E8B', "#b80926","#b80926"])
ax21 = plt.title('Feature importances obtained from coefficients', size=20)
ax21 = plt.xticks(rotation='vertical')

# Visualization Sum of Reviews per Region
fig23, ax23 = plt.subplots()
plot_n_n = sub.groupby('neighbourhood_group_cleansed')['number_of_reviews'].sum().sort_values(ascending=True)
ax23 = plot_n_n.plot.barh(figsize = (10,8), color = 'b', width = 0.5)
ax23 = plt.title('Sum Number of Reviews by Region', fontsize = 15)
ax23 = plt.xlabel('Number of Reviews', fontsize = 12)


# Visualization Sum of Reviews per Region
fig24, ax24 = plt.subplots()
plot_n_1 = sub.groupby('neighbourhood_cleansed')['number_of_reviews'].sum().sort_values(ascending=True)
plot_n_1 = plot_n_1[-20:]
ax24 = plot_n_1.plot.barh(figsize = (10,8), color = 'b', width = 0.5)
ax24 = plt.title('Sum Number of Reviews on Singapore\'s Top 20 Neighbourhood', fontsize = 15)
ax24 = plt.xlabel('Number of Reviews', fontsize = 12)

# CHOROPLETH REGION!!--------------
# Create Map
m = folium.Map(location=[1.345350, 103.838880], zoom_start = 12, tiles=None)
folium.TileLayer('CartoDB positron', name='Light Map', control=False).add_to(m)

# Create region code
code_region_list = []
for row in sub['neighbourhood_group_cleansed']:
  if row == 'North Region':
    code_region_list.append(1)
  elif row == 'Central Region':
    code_region_list.append(2)
  elif row == 'East Region':
    code_region_list.append(3)
  elif row == 'North-East Region':
    code_region_list.append(4)
  elif row == 'West Region':
    code_region_list.append(5)

sub['region_code'] = code_region_list

## REGION CHOROPLETH FIX!!
import json
with open ("datas/neighbourhoods.geojson", 'r') as jsonFile:
    singmapdata = json.load(jsonFile)

choropleth2 = folium.Choropleth(
    geo_data = singmapdata,
    name = 'Choropleth Map of Singapore Listings',
    data = sub,
    columns=['neighbourhood_group_cleansed', "region_code"],
    key_on = "feature.properties.neighbourhood_group",
    fill_color = "Spectral",
    fill_opacity = 0.7,
    line_color='black'
).add_to(m)

for key in choropleth2._children:
    if key.startswith('color_map'):
        del(choropleth2._children[key])
choropleth2.add_to(m)

 # Filling the region data
m = add_categorical_legend(m, 'Region',
                            colors = ["#dc747e", "#f7ad88", "#ebf6b4", "#b3ddaf","#6da8c7" ],
                            labels = ['North', 'Central', 'East', 'North-East', 'West'])

# Sum of Number of Reviews Per Region
lat_reg = ["1.403782", "1.30777778", "1.36277778", "1.39361111",  "1.37416667"]
lon_reg = ["103.79414", "103.81972222", "103.98250000", "103.89277778", "103.72250000"]
sum_reg = [20960, 110849, 14798, 7170, 8627]
count_nol = [742, 5039, 488, 250, 657]

# CHOROPLETH NEIGHBOURHOOD!
mn = folium.Map(location=[1.345350, 103.838880], zoom_start = 12, tiles=None)
folium.TileLayer('CartoDB positron', name='Light Map', control=False).add_to(mn)

sub_neig = sub.groupby('neighbourhood_cleansed').agg({'number_of_reviews':['mean','sum','count']}).sort_values(by=('number_of_reviews', 'sum'), ascending=False)
sub_neig = sub_neig['number_of_reviews'].reset_index()

centro = []

import requests
from shapely.geometry import shape
features = singmapdata['features']
for feature in features:
  s = shape(feature['geometry'])
  centro.append(s.centroid)

n_list = []
for i in range(0, 55):
  n_list.append(singmapdata['features'][i]['properties']['neighbourhood'])


#Obtain the latitude and langitude
lon = list(map(lambda p: p.x, centro))
lat = list(map(lambda p: p.y, centro))
sub_neig_f = pd.DataFrame({'Neighbourhood':n_list,
                           'Longitude':lon,
                           'Latitude':lat})

#Prepare the dataframe
sub_neig_pair = sub.groupby('neighbourhood_cleansed')['number_of_reviews'].sum()
sub_neig_pair = sub_neig_pair.reset_index()
sub_neig_pair1 = sub.groupby('neighbourhood_cleansed')['number_of_reviews'].count()
sub_neig_pair1 = sub_neig_pair1.reset_index()

sub_neig_pair = sub_neig_pair.rename(columns={"neighbourhood_cleansed": "n1", "number_of_reviews": "sum"})
sub_neig_pair1 = sub_neig_pair1.rename(columns={"neighbourhood_cleansed": "n2", "number_of_reviews": "count"})
#Joining the datas
sub_neig_final = sub_neig_f.merge(sub_neig_pair, left_on ='Neighbourhood',\
                                  right_on = 'n1',
                                  how = 'outer')
sub_neig_final = sub_neig_final.merge(sub_neig_pair1, left_on ='Neighbourhood',\
                                  right_on = 'n2',
                                  how = 'outer')
sub_neig_final = sub_neig_final.drop(columns=['n1','n2'])
sub_neig_final = sub_neig_final.fillna(0)
sub_neig_final['sum'] = sub_neig_final['sum'].astype(int)
sub_neig_final['count'] = sub_neig_final['count'].astype(int)




#Masuk If, ganti di bagian columns nya





# ---------------------------
# DASHBOARDING SECTION
# ---------------------------

# Mulai dari Judul
st.set_page_config(layout='centered')
st.title('Singapore Hospitality Industry')
st.subheader("Is hospitality industry in Singapore still worth it to invest ?")
st.write("Hospitality industry was impacted globaly by covid-19. Light breaker procedure to reduce the spread of the desease, results in decrease of activity \
one of which is reserving and renting a property. Important and significant listings parameters needed to be analyzed to \
    increase the sales in accomodation listings. Out of the popular hotel sites, Airbnb was chosen as it serves as a source of reliable data provider. \
        It has collected all the listings and reviews from their own websites, up to around 15198 rows of data. \
            This would increase the reliability of the results even after data cleaning. \
                In addition, data on the reviews made by customers and price per date data have also been collected.")
st.write("This project will dig into singapore's airbnb data that starts from 29 Sept 2021 to 22 June 2022 which was obtained from \
    http://insideairbnb.com/get-the-data . \
The goal is to find the best paramters to focus on and take action from. Also we will see if \
    there are still business opportunities in the Singapore hospitality industry. \
The datas are licensed under a Creative Commons Attribution 4.0 International License.")
# Buat keadaan listings singapore sekarang
st.markdown("---")
st.subheader("Accommodation Condition in Singapore")
tab00,tab01, tab02, tab03 = st.tabs(["Airbnb Listings in Singapore", "Neighbourhood", "Region", "Listings Map Distribution"])
with tab00:
    st.markdown("##### Number of Listings Distribution on Map")
    #Number of listings on map
    tiles = st.sidebar.radio("Choose the Tile Preset For Airbnb Listings in Singapore",("stamenterrain", 'stamentoner', 'OpenStreetMap'))
    Singapore=folium.Map(location=[1.3521,103.8198],tiles=tiles,zoom_start=12)
    marker_cluster = MarkerCluster().add_to(Singapore)
    locations = sub[['latitude', 'longitude']]
    locationlist = locations.values.tolist()
    for point in range(0, len(locationlist)):
        folium.Marker(locationlist[point]).add_to(marker_cluster)
    #Show on Streamlit
    folium_static(Singapore)
with tab01:
    neig_inf = st.radio("Select The Information", ['Number of Listings Per Neighbourhood', 'Sum Number of Reviews Per Neighbourhood'])
    if neig_inf == "Number of Listings Per Neighbourhood":
        choropleth1 = folium.Choropleth(
            geo_data = singmapdata,
            name = 'Choropleth Map of Singapore Listings per Neighbourhood',
            data = sub_neig,
            columns=['neighbourhood_cleansed', "count"],
            key_on = "feature.properties.neighbourhood",
            fill_opacity = 0.7,
            fill_nan = 'grey',
            line_color='black',
            legend_name = 'Number of Listings Per Neighbourhood'
        ).add_to(mn)
        for lat, lon, nor in zip(sub_neig_final['Latitude'], sub_neig_final['Longitude'], sub_neig_final['count']):
            folium.Marker(location=[lat, lon],
                    icon= folium.DivIcon(
                    icon_size=(32,32),
                    icon_anchor = (0, 0),
                    html = f'<div style="font-size:15pt;color:white;text-shadow: 1px 2px #5c5a5a">{nor}</div>')
                ).add_to(mn)
        folium_static(mn)
        st.subheader("Number of Listings Per Neighbourhood")
        st.pyplot(fig = fig20)
    elif neig_inf == "Sum Number of Reviews Per Neighbourhood":
        choropleth1 = folium.Choropleth(
            geo_data = singmapdata,
            name = 'Choropleth Map of Singapore Listings per Neighbourhood',
            data = sub_neig,
            columns=['neighbourhood_cleansed', "sum"],
            key_on = "feature.properties.neighbourhood",
            fill_opacity = 0.7,
            fill_nan = 'grey',
            line_color='black',
            legend_name = 'Number of Reviews Per Neighbourhood'
        ).add_to(mn)
        for lat, lon, nor in zip(sub_neig_final['Latitude'], sub_neig_final['Longitude'], sub_neig_final['sum']):
            folium.Marker(location=[lat, lon],
                        icon= folium.DivIcon(
                        icon_size=(32,32),
                        icon_anchor = (0, 0),
                        html = f'<div style="font-size:15pt;color:white;text-shadow: 1px 2px #5c5a5a">{nor}</div>')
                        ).add_to(mn)
        folium_static(mn)
        st.subheader("Sum Number of Reviews Per Neighbourhood")
        st.pyplot(fig = fig24)
with tab02:
    reg_inf = st.radio("Select The Information", ['Number of Listings Per Region', 'Sum Number of Reviews Per Region'])
    if reg_inf == 'Sum Number of Reviews Per Region':
        for lat, lon, norev in zip(lat_reg, lon_reg, sum_reg):
            folium.Marker(location=[lat, lon],
                icon= folium.DivIcon(
                  icon_size=(32,32),
                  icon_anchor = (0, 0),
                  html = f'<div style="font-size:15pt;color:white;text-shadow: 1px 2px #5c5a5a">{norev}</div>')
                ).add_to(m)
        folium_static(m)
        st.subheader("Sum Number of Reviews Per Region")
        st.pyplot(fig = fig23)
    elif reg_inf == 'Number of Listings Per Region':
        for lat, lon, nolist in zip(lat_reg, lon_reg, count_nol):
            folium.Marker(location=[lat, lon],
                icon= folium.DivIcon(
                  icon_size=(32,32),
                  icon_anchor = (0, 0),
                  html = f'<div style="font-size:15pt;color:white;text-shadow: 1px 2px #5c5a5a">{nolist}</div>')
                ).add_to(m)
        folium_static(m)
        st.subheader("Number of Listings Per Region")
        st.pyplot(fig = fig19)
with tab03:
    st.pyplot(fig = fig21)
# Buat visualisasi dan penjelasan terhadap feature importance

st.markdown("---")
st.subheader("Significant Parameters")
st.write("We will try find significant data parameters on airbnb listings using the importance parameter on logistic regression. The graph below shows features importance level:")
st.pyplot(fig = fig22)

st.write("From finding the parameters importance levels, we will divide the result into 2, which have \
     directly and inversely proportional relationship to number of reviews. \
         The significant parameters ares : Price(Harga), Bahtrooms(Kamar Mandi), \
             Beds(Jumlah Kasur) , Accommodates(Jumlah orang yang dapat diakomodasi), and \
                 Review Scores Rating(Rating Listings) and the rest are do not have enough significance levels.\
                     \n The lower the price, number of bathrooms, and bedrooms, then the number of \
                     reviews will increase. In contrary to that, the higher accommodation and the review score rating values, \
                         the number of reviews for a listings will also increase.")
st.markdown("---")






Sig_Par  = ["High Significance Parameters", "Low Significance Parameters"]
tab1, tab2 = st.tabs(Sig_Par)
tab1.subheader("High Significance Parameters")
tab2.subheader("Low Significance Parameters")
# High Parameter Visualization
with tab1:
    tab1_par = st.radio("Please Select The Parameter",\
         ['Price', 'Bathrooms', 'Beds', 'Accommodates', 'Scores Rating'])
    st.subheader(tab1_par)
    if tab1_par == 'Price':
        col1, col2 = st.columns([2,1])
        with col1:
            st.pyplot(fig = fig1)
        with col2:
            st.subheader('Price vs Number of Reviews')
            st.write("We can see slight decrease on the number of \
                reviews as the price goes up. Other than that, most of them\
                    has low number of reviews no matter the listings prices are.")
        col3, col4 = st.columns([2,1])
        with col3:
            st.pyplot(fig = fig2)
        with col4:
            st.subheader("Price vs Number of Reviews on Per Room Type")
            st.write("We can see a distinct price difference on private room type group \
                and entire home/apt room type. Shared room generaly have lower price range \
                    than other room type. Meanwhile hotel room price range are distributed evenly \
                        from 10 to 325 price range.")
    elif tab1_par == 'Beds':
        col1, col2 = st.columns([2,1])
        with col1:
            st.pyplot(fig = fig3)
        with col2:
            st.subheader("Beds vs Number of Reviews on Mean")
            st.write("We see increasing of number of reviews mean as\
                the further the number of bed increase.")
        col3, col4 = st.columns([2,1])
        with col3:
            st.pyplot(fig = fig4)
        with col4:
            st.subheader("Beds Vs Number of Reviews on Count")
            st.write("The majority of listings have 1 beds. Followed by 2 and 3 beds. ")
    elif tab1_par == "Accommodates":
        col1, col2 = st.columns([2,1])
        with col1:
            st.pyplot(fig=fig5)
        with col2:
            st.subheader("Accommodates Vs Number of Reviews on Mean")
            st.write("We see increasing of number of reviews mean as\
                the further the number of accepted accomodates increase.")
        col3, col4 = st.columns([2,1])
        with col3:
            st.pyplot(fig = fig6)
        with col4:
            st.subheader("Accommodates vs Number of Reviews on Count")
            st.write("The majority of the accepted accommodates range from 1 to 6.")
    elif tab1_par == "Scores Rating":
        col1, col2 = st.columns([2,1])
        with col1:
            st.pyplot(fig=fig7)
        with col2:
            st.subheader("Scores Rating Vs Number of Reviews on Mean")
            st.write("The majority of listings score rating ranging from 4 to 5 with few have rating ranging from 1 to 3.")
    elif tab1_par == 'Bathrooms':
        col1, col2 = st.columns([2,1])
        with col1:
            st.pyplot(fig = fig9)
        with col2:
            st.subheader("Bathrooms vs Number of Reviews on Count")
            st.write("There are spikes of mean number of reviews for bathrooms 4.5, 7.5, and 8.5.")
        col3, col4 = st.columns([2,1])
        with col3:
            st.pyplot(fig = fig10)
        with col4:
            st.subheader("Bathrooms vs Number of Reviews on Mean")
            st.write("Majority of the listings have bathrooms ranging from 1, 2 and 3.")
# Low Parameter Visualization
with tab2:
    tab2_par = st.radio("Please Select The Parameter", \
        ['Bedrooms', 'Availability 365', 'Host Identity Verified', \
            'Minimum Nights', 'Instant Bookable', 'Room Type'])
    st.subheader(tab2_par)
    if tab2_par == 'Minimum Nights':
        col1, col2 = st.columns([2,1])
        with col1:
            st.pyplot(fig = fig8)
        with col2:
            st.subheader('Minimum Nights vs Number of Reviews')
    elif tab2_par == 'Host Identity Verified':
        col1, col2 = st.columns([2,1])
        with col1:
            st.pyplot(fig = fig11)
        with col2:
            st.subheader('Host Identity Verified vs Number of Reviews Mean')
    elif tab2_par == 'Instant Bookable':
        col1, col2 = st.columns([2,1])
        with col1:
            st.pyplot(fig = fig13)
        with col2:
            st.subheader('Instant Bookable vs Number of Reviews Mean')
    elif tab2_par == 'Bedrooms':
        col1, col2 = st.columns([2,1])
        with col1:
            st.pyplot(fig = fig14)
        with col2:
            st.subheader('Bedrooms vs Number of Reviews Mean')
        col3, col4 = st.columns([2,1])
        with col3:
            st.pyplot(fig = fig15)
        with col4:
            st.subheader('Bedrooms vs Number of Reviews Count')
    elif tab2_par == 'Availability 365':
        col1, col2 = st.columns([2,1])
        with col1:
            st.pyplot(fig = fig16)
        with col2:
            st.subheader('Availability 365 vs Number of Reviews Mean')
    elif tab2_par == 'Room Type':
        col1, col2 = st.columns([2,1])
        with col1:
            st.pyplot(fig = fig17)
        with col2:
            st.subheader('Room Type vs Number of Reviews Count')
        col3, col4 = st.columns([2,1])
        with col3:
            st.pyplot(fig = fig18)
        with col4:
            st.subheader('Bedrooms vs Number of Reviews Mean')

st.markdown("---")

st.subheader("Price and Number of Reviews Movement")
#Price data
df_calendar = pd.read_csv("datas/calendar_4.csv")
df_calendar['date'] = pd.to_datetime(df_calendar['date'], format='%Y-%m-%d')
df_calendar = df_calendar[(df_calendar['date']>= '2021-09-29') & (df_calendar['date']<= '2022-06-22')]
df_calendar['price'] = df_calendar['price'].apply(lambda x: str(x).replace('$',''))
df_calendar['price'] = pd.to_numeric(df_calendar['price'], errors = 'coerce')

#Reviews data
df_reviews = pd.read_csv("datas/reviews_1.csv")
df_reviews['date'] = pd.to_datetime(df_reviews['date'], format='%Y-%m-%d')
df_reviews = df_reviews[(df_reviews['date']>= '2021-09-29') & (df_reviews['date']<= '2022-06-22')]
df_reviews.head()

#Resample the data
p = df_calendar.resample('M', on='date')['price'].mean()
r = df_reviews.resample('M', on='date')['listing_id'].count()

movement = st.radio("Select The Information", ['Price History', 'Number of Reviews History'])
import plotly.graph_objects as go



if movement == 'Price History':
    fig = go.Figure(go.Scatter(
    mode= "lines+markers",
    y = p.values,
    x= p.index))

    fig.update_xaxes(
        title_text = 'Month',
    )

    fig.update_yaxes(
        title_text = "Price($)",
        range = [0,155]
    )

    fig.update_layout(
        title="Price History",
        font=dict(
            family="Arial",
            size=18)
    )
    st.plotly_chart(fig)
    st.metric("Price Movement", "$8", "5.517%" )
elif movement == 'Number of Reviews History':
    fig = go.Figure(go.Scatter(
    mode= "lines+markers",
    y = r.values,
    x= r.index))

    fig.update_xaxes(
        title_text = 'Month',
    )

    fig.update_yaxes(
        title_text = "Number of Reviews"
    )

    fig.update_layout(
        title="Number of Reviews History",
        font=dict(
            family="Arial",
            size=18)
    )
    st.plotly_chart(fig)

st.markdown('---')
col10, col11 = st.columns([2,1])
with col10:
    "Created by:    Yoel"
with col11:
    "Data Obtained From :   Inside Airbnb: Get the Data "
