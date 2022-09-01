# analysis and visualization before machine learning
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil import relativedelta
from utils import load_dataset


crime_type = {'drugs': 'Drugs', 'weapons': 'Possession of weapons', 'violence': 'Violence and sexual offences'}

drugs = load_dataset(crime_type['drugs'])
weapons = load_dataset(crime_type['weapons'])
violence = load_dataset(crime_type['violence'])

# london boundary
borough = gpd.read_file('data/statistical-gis-boundaries-london/ESRI/London_Borough_Excluding_MHW.shp')
borough = borough.to_crs(epsg=27700)

polys = borough.geometry.unary_union
london_bound = gpd.GeoDataFrame(geometry=[polys], crs=borough.crs)

# sample 200 points from possession of weapons dataset, plot on the map
data_samples =  weapons.sample(200, replace=False)
sns.set_theme(style="whitegrid")
ax = london_bound.boundary.plot(zorder=1)
sns.scatterplot(data=data_samples, x='loc_x', y='loc_y', hue='label', s=10, palette=['green', 'red'], zorder=2)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.set(xlabel='', ylabel='')
# plt.savefig('fig/neg_samples.png', bbox_inches='tight', dpi=200)
plt.show()

# kde based on data from 02/2022 to 04/2022
# crime points from 02 to 04/2022
drugs_kde = drugs.loc[(drugs['label'] == 1) & (drugs['Month'] >= datetime(2022, 2, 1))]
weapons_kde = weapons.loc[(weapons['label'] == 1) & (weapons['Month'] >= datetime(2022, 2, 1))]
violence_kde = violence.loc[(violence['label'] == 1) & (violence['Month'] >= datetime(2022, 2, 1))]

# plot
fig, ax = plt.subplots(1,3, figsize=(24,6))
sns.set(style='whitegrid')
sns.despine(left=True, bottom=True)
sns.set_context('paper')

borough.boundary.plot(ax=ax[0], zorder=2)
borough.boundary.plot(ax=ax[1], zorder=2)
borough.boundary.plot(ax=ax[2], zorder=2)

sns.kdeplot(x=drugs_kde['loc_x'], y=drugs_kde['loc_y'], shade=True, bw_method='scott', bw_adjust=0.3, cmap='Reds', ax=ax[0], cbar=True, zorder=1)
sns.kdeplot(x=weapons_kde['loc_x'], y=weapons_kde['loc_y'], shade=True, bw_method='scott', bw_adjust=0.3, cmap='Blues', ax=ax[1], cbar=True, zorder=1)
sns.kdeplot(x=violence_kde['loc_x'], y=violence_kde['loc_y'], shade=True, bw_method='scott', bw_adjust=0.3, cmap='Greens', ax=ax[2], cbar=True, zorder=1)

ax[0].set_title('Drugs KDE', fontsize=16)
ax[0].set(xlabel='', ylabel='')
ax[1].set_title('Possession of weapons KDE', fontsize=16)
ax[1].set(xlabel='', ylabel='')
ax[2].set_title('Violence and sexual offences, KDE', fontsize=16)
ax[2].set(xlabel='', ylabel='')

plt.tight_layout()
# plt.savefig('fig/kde.png', bbox_inches='tight', dpi=200)
plt.show()

# number of crimes line plot
# count the number of crimes for each crime type each month
crime_num = []
base = datetime(2021, 1, 1)
while base < datetime(2022, 5, 1):
    num = drugs.loc[(drugs['label'] == 1) & (drugs['Month'] == base)].shape[0]
    crime_num.append(['Drugs', base, num])
    num = weapons.loc[(weapons['label'] == 1) & (weapons['Month'] == base)].shape[0]
    crime_num.append(['Possession of weapons', base, num])
    num = violence.loc[(violence['label'] == 1) & (violence['Month'] == base)].shape[0]
    crime_num.append(['Violence and sexual offences', base, num])
    base = base + relativedelta.relativedelta(months=1)
df = pd.DataFrame(crime_num, columns=['type', 'month', 'num'])

# plot
sns.set_theme(style='whitegrid')
g = sns.relplot(data=df, x='month', y='num', col='type', hue='type', kind='line',
                facet_kws={'sharey': False}, legend=False)
g.set_titles('{col_name}').set_axis_labels('', 'Number of crimes')
g.set_xticklabels(rotation=30)
plt.tight_layout()
# plt.savefig('fig/crime_num.png', bbox_inches='tight', dpi=200)
plt.show()
