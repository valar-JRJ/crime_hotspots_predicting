import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime
from dateutil import relativedelta
import shapely
import random
import os
import statsmodels.api as sm
from statsmodels.nonparametric.kernel_density import EstimatorSettings
from utils import get_logger


def cell_construct(bounds: gpd.GeoDataFrame, cell_len: int):
    # total bounds of the geographical area
    xmin, ymin, xmax, ymax = bounds.total_bounds
    # projection of the grid
    crs = bounds.crs
    # generating grid cells
    # referencing from https://james-brennan.github.io/posts/fast_gridding_geopandas/
    cells_lst = []
    for x0 in np.arange(xmin, xmax+cell_len, cell_len):
        for y0 in np.arange(ymin, ymax+cell_len, cell_len):
            # locating cells
            x1 = x0-cell_len
            y1 = y0+cell_len
            cells_lst.append(shapely.geometry.box(x0, y0, x1, y1))
    cells_gdf = gpd.GeoDataFrame(cells_lst, columns=['geometry'], crs=crs)
    # mapping the geometry boundary of London into the geometry of cells
    merged_cells = gpd.sjoin(left_df=cells_gdf, right_df=bounds, how='inner')
    # drop duplicate cells, reset index, unify crs
    merged_cells = merged_cells.drop_duplicates(subset=['geometry'])
    merged_cells = merged_cells.drop(columns=['index_right']).reset_index(drop=True)
    merged_cells = merged_cells.to_crs(epsg=27700)
    return merged_cells


def crime_extract(df: pd.DataFrame, ctype: str, current: datetime, last=None):
    if last is None:
        crime = df.loc[(df['Crime type'] == ctype) & (df['Month'] == current)].reset_index()
    else:
        crime = df.loc[(df['Crime type'] == ctype) & (df['Month'] < current) & (df['Month'] >= last)].reset_index()

    crime = crime[['Month', 'Longitude', 'Latitude', 'LSOA code']]
    crime.dropna(inplace=True)
    crime = gpd.GeoDataFrame(crime, geometry=gpd.points_from_xy(crime['Longitude'], crime['Latitude'], crs='epsg:4326'))
    crime = crime.to_crs(epsg=27700)
    return crime


def kernel_pdf(pre_crime: gpd.GeoDataFrame, cur_points: gpd.GeoDataFrame):
    pre_loc = np.vstack([pre_crime.geometry.x, pre_crime.geometry.y]).T
    cur_loc = np.vstack([cur_points.geometry.x, cur_points.geometry.y]).T
    bw = sm.nonparametric.bandwidths.bw_scott(pre_loc)
    print(f'bandwidth: {bw}')
    settings = EstimatorSettings(n_jobs=8)
    kde = sm.nonparametric.KDEMultivariate(data=pre_loc, var_type='uu', bw=bw, defaults=settings)
    pdf = kde.pdf(cur_loc)
    return pdf


def generate_random(number, polygon):
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < number:
        pnt = shapely.geometry.Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(pnt):
            points.append(pnt)
    return points


def negative_sampling(crimes: gpd.GeoDataFrame, lsoas: gpd.GeoDataFrame):
    london_cells = cell_construct(lsoas, 500)
    crime_cells = gpd.sjoin(left_df=london_cells, right_df=crimes, how='inner', op='contains')
    crime_cells = crime_cells.drop_duplicates(subset=['geometry'])
    negative_cells = london_cells.drop(crime_cells.index, axis=0)

    # creating buffer to ensure enough distance between crime points and negative points
    negative_cells.geometry = negative_cells.geometry.centroid.buffer(100, cap_style=3)

    # generating negative points
    num_per_cell = int(crimes.shape[0]/negative_cells.shape[0]*1.5)
    if num_per_cell == 0:
        num_per_cell =1
    neg_points = []
    negative_cells.geometry.apply(lambda x: neg_points.extend(generate_random(num_per_cell, x)))
    print(f'number of points per cell: {num_per_cell}')
    neg_points = gpd.GeoDataFrame(neg_points, columns=['geometry'], crs=lsoas.crs)
    neg_points = gpd.sjoin(left_df=neg_points, right_df=lsoa, how='left', op='within').dropna()

    # sampling
    num = random.randint(int(0.8*crimes.shape[0]), int(1.2*crimes.shape[0]))
    try:
        neg_samples = neg_points.sample(n=num, replace=False).reset_index(drop=True)
    except ValueError as e:
        print(e)
        neg_samples = neg_points
        # neg_samples = neg_points.sample(n=int(0.8*crimes.shape[0]), replace=False).reset_index(drop=True)
    return neg_samples.loc[:, ['LSOA11CD', 'LAD11CD', 'geometry']]


if __name__ == "__main__":
    imd = pd.read_csv('data/IoD2019_Transformed_Scores.csv')
    imd.columns = ['LSOACD', 'LSOANM', 'LADCD', 'LADNM', 'Income', 'Employment', 'Education', 'Health', 'Crime',
                   'Barriers', 'Living Environment']

    all_crime = pd.read_csv('data/London Crime/all_crime.csv')
    all_crime['Month'] = pd.to_datetime(all_crime['Month'], yearfirst=True)

    mobility = pd.read_csv('data/google_activity_by_London_Borough.csv')
    mobility.columns = ['date', 'area_name', 'area_code', 'retail', 'grocery', 'parks', 'stations', 'workplaces',
                        'residential']
    mobility['stations'] = pd.to_numeric(mobility['stations'], downcast='float')
    mobility['date'] = pd.to_datetime(mobility['date'], yearfirst=True)
    mobility.fillna(method='bfill', inplace=True)

    lsoa = gpd.read_file('data/statistical-gis-boundaries-london/ESRI/LSOA_2011_London_gen_MHW.shp')
    lsoacd = lsoa['LSOA11CD'].to_numpy()

    all_crime = all_crime.loc[all_crime['LSOA code'].isin(lsoacd)]
    crime_type = ['Violence and sexual offences', 'Drugs', 'Possession of weapons']
    # crime_type = ['Drugs', 'Possession of weapons']

    logger = get_logger('log/')

    for tp in crime_type:
        path = f'data/dataset/{tp}'
        if not os.path.isdir(path):
            print('Datasets directory is not present. Creating a new one..')
            os.mkdir(path)
        else:
            print('Datasets directory is present.')
        
        cur_month = datetime(2022, 1, 1)
        while cur_month <= datetime(2022, 4, 1):
            pre_quarter = cur_month - relativedelta.relativedelta(months=3)
            pre_month = cur_month - relativedelta.relativedelta(months=1)
            next_month = cur_month + relativedelta.relativedelta(months=1)

            pre_crimes = crime_extract(all_crime, tp, current=cur_month, last=pre_quarter)
            # pre_crimes = crime_extract(all_crime, tp, current=pre_month)
            cur_crimes = crime_extract(all_crime, tp, current=cur_month)

            # mobility of current month
            # mob_month = mobility.loc[(mobility['date'] >= cur_month) &
            #                          (mobility['date'] < next_month)].groupby('area_code').median()

            # mobility of previous month
            mob_month = mobility.loc[(mobility['date'] >= pre_month) &
                                     (mobility['date'] < cur_month)].groupby('area_code').median()

            cur_crimes = cur_crimes.merge(imd, left_on='LSOA code', right_on='LSOACD', how='left')
            cur_crimes = cur_crimes.merge(mob_month, left_on='LADCD', right_on='area_code', how='left')

            negative_samples = negative_sampling(cur_crimes, lsoa)
            negative_samples['Month'] = cur_month
            negative_samples = negative_samples.merge(imd, left_on='LSOA11CD', right_on='LSOACD', how='left')
            negative_samples = negative_samples.merge(mob_month, left_on='LAD11CD', right_on='area_code', how='left')

            negative_samples['label'] = 0
            cur_crimes['label'] = 1
            training_samples = pd.concat([cur_crimes, negative_samples], axis=0, join='inner', ignore_index=True)

            density = kernel_pdf(pre_crimes, training_samples)
            training_samples['density'] = density
            training_samples['loc_x'] = training_samples.geometry.x
            training_samples['loc_y'] = training_samples.geometry.y
            training_samples = pd.DataFrame(training_samples.drop(columns=['geometry']))

            logger.info(f'{cur_month} {tp} crimes, {cur_crimes.shape[0]} positive samples, '
                        f'{negative_samples.shape[0]} negative samples')

            training_samples.to_csv(os.path.join(path, f'{cur_month.year}-{cur_month.month}.csv'), index=False)
            cur_month = next_month

    print('feature extraction finished!')
