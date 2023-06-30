
import numpy as np, pandas as pd

# Loading of the dataset via pandas
df = kc_data = pd.read_csv("data/King_County_House_prices_dataset.csv")

# We will drop this row
kc_data.drop(15856, axis=0, inplace=True)

# We replace "?" with Nan
#kc_data['sqft_basement'] = kc_data['sqft_basement'].replace('?', np.NaN).astype(float)

# We are calculating the "sqft_basement" by substracting sqft_above of sqft_living
kc_data.eval('sqft_basement = sqft_living - sqft_above', inplace=True)

# We replace Nan values in "view" with the most frequent expression (0)
kc_data['view'].fillna(0, inplace=True)

# We replace Nan values in "waterfront" with the most frequent expression (0)
kc_data.waterfront.fillna(0, inplace=True)

# We will create an empty list in which we will store values
last_known_change = []

# For each row in our data frame, we look at what is in the column "yr_renovated".
for idx, yr_re in kc_data.yr_renovated.items():
    # if "yr_renovated" is 0 or contains no value, we store the year of construction of the house in our empty listes ab
    if str(yr_re) == 'nan' or yr_re == 0.0:
        last_known_change.append(kc_data.yr_built[idx])
    # if there is a value other than 0 in the column "yr_renovated", we transfer this value into our new list
    else:
        last_known_change.append(int(yr_re))


# We create a new column and take over the values of our previously created list
kc_data['last_known_change'] = last_known_change

# We delete the "yr_renovated" and "yr_built" columns
kc_data.drop("yr_renovated", axis=1, inplace=True)
kc_data.drop("yr_built", axis=1, inplace=True)


# Absolute difference of latitude between centre and property
kc_data['delta_lat'] = np.absolute(47.62774 - kc_data['lat'])
# Absolute difference of longitude between centre and property
kc_data['delta_long'] = np.absolute(-122.24194 - kc_data['long'])
# Distance between centre and property
kc_data['center_distance']= ((kc_data['delta_long']* np.cos(np.radians(47.6219)))**2 
                                   + kc_data['delta_lat']**2)**(1/2)*2*np.pi*6378/360


# This function helps us to calculate the distance between the house overlooking the seafront and the other houses.
def dist(long, lat, ref_long, ref_lat):
    '''dist computes the distance in km to a reference location. Input: long and lat of 
    the location of interest and ref_long and ref_lat as the long and lat of the reference location'''
    delta_long = long - ref_long
    delta_lat = lat - ref_lat
    delta_long_corr = delta_long * np.cos(np.radians(ref_lat))
    return ((delta_long_corr)**2 +(delta_lat)**2)**(1/2)*2*np.pi*6378/360


# All houses with "waterfront" are added to the list
water_list= kc_data.query('waterfront == 1')


water_distance = []
# For each row in our data frame we now calculate the distance to the seafront
for idx, lat in kc_data.lat.items():
    ref_list = []
    for x,y in zip(list(water_list.long), list(water_list.lat)):
        ref_list.append(dist(kc_data.long[idx], kc_data.lat[idx],x,y).min())
    water_distance.append(min(ref_list))


# wir erstellen eine neue Spalte und übernehmen die Werte unserer vorher erstellten Liste
kc_data['water_distance'] = water_distance





