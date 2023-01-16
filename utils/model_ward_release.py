import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from pyproj import CRS

from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from spreg.ols import OLS
from spreg.ml_lag import ML_Lag
from spreg.ml_error import ML_Error
from spreg import GM_Lag
from spreg import GM_Error
from pysal.lib import weights

from spreg.diagnostics import likratiotest

from matplotlib import colors
from matplotlib import cm

from utils.geo_file_path import south_scotland, scotland_power_station, scotland_residence, conservation, wind_farm, \
    temperature, precipitation, population, road, community_council, landscape, file_path_cwd
from utils.add_widget import add_north, add_scale_bar

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
# pd.set_option('display.max_rows', None)

CRS_4326 = CRS('epsg:4326')
STATIC_CRS = CRS('epsg:27700')

GRID_NUM = 32

wind_speed = gpd.read_file(file_path_cwd('resource/RasterToVector/wind_speed_shp.shp'))
land_use = gpd.read_file(file_path_cwd('resource/RasterToVector/land_use_shp.shp'))

# aspect = gpd.read_file(file_path_cwd（'resource/RasterToVector/aspect_shp.shp')）
slope = gpd.read_file(file_path_cwd('resource/RasterToVector/slope_shp.shp'))


def generate_ward():
    return community_council.reset_index()


def get_ward_contain_point(name, right_df):
    contain_grid = generate_ward()

    result_column = f'{name}_result'
    residence_join = gpd.sjoin(left_df=contain_grid, right_df=right_df.to_crs(STATIC_CRS), op='contains').groupby(
        'index')[
        'id'].count().to_frame().reset_index()

    residence_join.columns = ['index', result_column]

    contain_grid = pd.merge(contain_grid, residence_join, on='index', how='outer')
    contain_grid[result_column] = contain_grid[result_column].fillna(0)

    return contain_grid


def get_ward_overlay_area(name, df2, value_column=None, ):
    overlay_grid = generate_ward()

    result_column = f'{name}_result'

    overlay_result = gpd.overlay(df1=overlay_grid, df2=df2.to_crs(STATIC_CRS), how='intersection')
    overlay_result[result_column] = overlay_result[value_column] if value_column else overlay_result.area / 10000
    overlay_result = pd.merge(overlay_grid, overlay_result, on='index', how='left')

    arr = [0 for _ in range(286)]

    for i in range(overlay_result.shape[0]):
        index = overlay_result.loc[i, 'index']
        value = overlay_result.loc[i, result_column]
        arr[index] += value

    overlay_grid[result_column] = arr
    overlay_grid[result_column] = overlay_grid[result_column].fillna(0.0)
    overlay_grid.rename(columns={'geometry_x': 'geometry'}, inplace=True)

    return overlay_grid


def get_ward_overlay_line(name, df2):
    overlay_line_grid = generate_ward()
    result_column = f'{name}_result'
    overlay_line_result = gpd.overlay(df1=overlay_line_grid, df2=df2.to_crs(STATIC_CRS), keep_geom_type=False)
    overlay_line_result[result_column] = overlay_line_result.length
    overlay_line_result = pd.merge(overlay_line_grid, overlay_line_result, on='index', how='left')

    array = [0 for _ in range(286)]

    for i in range(overlay_line_result.shape[0]):
        index = overlay_line_result.loc[i, 'index']
        value = overlay_line_result.loc[i, result_column]
        array[index] += value

    overlay_line_grid[result_column] = array
    overlay_line_grid[result_column] = overlay_line_grid[result_column].fillna(0.0)

    return overlay_line_grid


def plot_axis_grid(name, ax, display_grid=None):
    ax.set_title(name)
    ax.set_axis_off()

    # normalization
    result = display_grid[f'{name}_result']
    display_grid[f'{name}_result_normalised'] = (result - result.min(axis=0)) / (
            result.max(axis=0) - result.min(axis=0))
    if display_grid is not None:
        display_grid.plot(ax=ax, column=f'{name}_result_normalised', cmap='YlOrRd', edgecolor='Black', legend=True,
                          linewidth=.5)


def plot_ward_factors():
    fig, axis = plt.subplots(nrows=3, ncols=4, figsize=(12, 7), dpi=100)
    fig.suptitle('Ward Factors Input')

    ax1 = axis[0, 0]
    ax2 = axis[0, 1]
    ax3 = axis[0, 2]
    ax4 = axis[0, 3]
    ax5 = axis[1, 0]
    ax6 = axis[1, 1]
    ax7 = axis[1, 2]
    ax8 = axis[1, 3]
    ax9 = axis[2, 0]
    ax10 = axis[2, 1]
    ax11 = axis[2, 2]
    ax12 = axis[2, 3]

    residence_grid = get_ward_contain_point('residence', scotland_residence)

    scotland_power_station['id'] = scotland_power_station.index
    power_station_grid = get_ward_contain_point('power_station', scotland_power_station)

    conservation_overlay_grid = get_ward_overlay_area('conservation', conservation)
    wind_farm_overlay_grid = get_ward_overlay_area('wind_farm', wind_farm)
    temperature_overlay_grid = get_ward_overlay_area('temperature', temperature, value_column='aveTemp')
    precipitation_overlay_grid = get_ward_overlay_area('precipitation', precipitation, value_column='prSum')
    population_overlay_grid = get_ward_overlay_area('population', population, value_column='SSP1_2020')
    landscape_grid = get_ward_overlay_area('landscape', landscape)

    road_overlay_grid = get_ward_overlay_line('road', road)
    community_council_grid = get_ward_overlay_line('community_council', community_council)

    wind_speed_grid = get_ward_overlay_area('wind_speed', wind_speed[wind_speed['DN'] > 6], value_column='DN')
    land_use_grid = get_ward_overlay_area('land_use', land_use[land_use['DN'] <= 10], value_column='DN')
    slope_grid = get_ward_overlay_area('slope', slope[slope['DN'] < 15], value_column='DN')

    # plot
    plot_axis_grid('wind_farm', ax1, wind_farm_overlay_grid)
    add_north(ax=ax1)
    plot_axis_grid('residence', ax2, residence_grid)
    plot_axis_grid('power_station', ax3, power_station_grid)
    plot_axis_grid('conservation', ax4, conservation_overlay_grid)
    plot_axis_grid('temperature', ax5, temperature_overlay_grid)
    plot_axis_grid('precipitation', ax6, precipitation_overlay_grid)
    plot_axis_grid('population', ax7, population_overlay_grid)
    plot_axis_grid('road', ax8, road_overlay_grid)
    plot_axis_grid('slope', ax9, slope_grid)
    plot_axis_grid('landscape', ax10, landscape_grid)
    plot_axis_grid('wind_speed', ax11, wind_speed_grid)
    plot_axis_grid('land_use', ax12, land_use_grid)
    add_scale_bar(ax=ax12, lon0=370000, lat0=535000)

    # norm = colors.Normalize(vmin=0, vmax=1)
    # fig.colorbar(cm.ScalarMappable(norm=norm, cmap='YlOrRd'), ax=axis, orientation='horizontal', shrink=.7, aspect=30,
    #              pad=.1)


def plot_coefficient(geo_data, cof_data_column, gwr_filter_t):
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(12, 7))
    axes = ax.flatten()
    fig.suptitle('Ward Factors Coefficient Output')

    for i in range(0, len(cof_data_column)):
        ax = axes[i]
        ax.set_title(cof_data_column[i])
        geo_data.plot(ax=ax, column=cof_data_column[i], edgecolor='white', cmap='Blues', vmin=-1, vmax=1,
                      legend=True,
                      # scheme='User_Defined',
                      # classification_kwds=dict(bins=[-1, -0.5, 0, 0.5, 1]),
                      # legend_kwds=legend_kwds
                      )

        if (gwr_filter_t[i] == 0).any():
            geo_data[gwr_filter_t[i] == 0].plot(color='lightgrey', ax=ax, edgecolor='white')

    for i in range(0, len(axes)):
        ax = axes[i]
        ax.set_axis_off()

    add_north(ax=axes[0])
    add_scale_bar(ax=axes[-1], lon0=370000, lat0=535000)


def ols(y, x, w, name_y, name_x):
    ols_result = OLS(y, x, w, spat_diag=True, moran=True, name_y=name_y, name_x=name_x, name_ds=name_y,
                     name_w='queen')

    return ols_result


def ml_lag(y, x, w, name_y, name_x):
    ml_lag_result = ML_Lag(y, x, w, name_y=name_y, name_x=name_x, name_ds=name_y, name_w='queen')

    return ml_lag_result


def ml_error(y, x, w, name_y, name_x):
    ml_error_result = ML_Error(y, x, w, name_y=name_y, name_x=name_x, name_ds=name_y, name_w='queen')

    return ml_error_result


def gwr(y, x, coordinates):
    g_x = (x - x.mean(axis=0)) / x.std(axis=0)
    g_y = (y - y.mean(axis=0)) / y.std(axis=0)
    gwr_selector = Sel_BW(coordinates, y, x)
    # gwr_bw = gwr_selector.search(search_method='golden_section', criterion='AIC') - 0.633
    # gwr_bw = gwr_selector.search(search_method='scipy', criterion='AIC') - 0.633
    gwr_bw = gwr_selector.search(search_method='golden_section', criterion='AIC')
    # print('best gwr：', gwr_bw)

    gwr_results = GWR(coordinates, g_y, g_x, bw=gwr_bw, fixed=False, kernel='gaussian', constant=True,
                      spherical=True).fit()

    return gwr_results


def mgwr(y, x, coordinates):
    g_x = (x - x.mean(axis=0)) / x.std(axis=0)
    g_y = (y - y.mean(axis=0)) / y.std(axis=0)

    mgwr_selector = Sel_BW(coordinates, g_y, g_x, multi=True)
    mgwr_bw = mgwr_selector.search()
    mgwr_results = MGWR(coordinates, g_y, g_x, selector=mgwr_selector, kernel='gaussian').fit()  # 0.823

    return mgwr_results


def gm_lag(y, x, w, name_y, name_x):
    gm_lag_result = GM_Lag(y, x, w=w, w_lags=2, robust='white', name_y=name_y, name_x=name_x, name_ds=name_y,
                           name_w='queen')

    return gm_lag_result


def gm_error(y, x, w, name_y, name_x):
    gm_error_result = GM_Error(y, x, w=w, name_y=name_y, name_x=name_x, name_ds=name_y, name_w='queen')

    return gm_error_result


def merge_columns(merge_data_list, merge_data_column, merge_result=generate_ward()):
    for i in range(len(merge_data_list)):
        col = merge_data_column[i]
        merge_result = pd.merge(merge_result, merge_data_list[i][['index', col]], on='index')

    return merge_result


def model_ward(is_summary=False, is_plot_coefficient=False):
    wind_farm_overlay_grid = get_ward_overlay_area('wind_farm', wind_farm)

    residence_grid = get_ward_contain_point('residence', scotland_residence)
    scotland_power_station['id'] = scotland_power_station.index
    power_station_grid = get_ward_contain_point('power_station', scotland_power_station)
    conservation_overlay_grid = get_ward_overlay_area('conservation', conservation)
    temperature_overlay_grid = get_ward_overlay_area('temperature', temperature, value_column='aveTemp')
    precipitation_overlay_grid = get_ward_overlay_area('precipitation', precipitation, value_column='prSum')
    population_overlay_grid = get_ward_overlay_area('population', population, value_column='SSP1_2020')
    road_overlay_grid = get_ward_overlay_line('road', road)
    slope_grid = get_ward_overlay_area('slope', slope[slope['DN'] < 15], value_column='DN')
    landscape_grid = get_ward_overlay_area('landscape', landscape)
    wind_speed_grid = get_ward_overlay_area('wind_speed', wind_speed, value_column='DN')
    land_use_grid = get_ward_overlay_area('land_use', land_use[land_use['DN'] <= 10], value_column='DN')

    merge_data_list = [residence_grid, power_station_grid, conservation_overlay_grid, temperature_overlay_grid,
                       precipitation_overlay_grid, population_overlay_grid, road_overlay_grid, slope_grid,
                       landscape_grid, wind_speed_grid, land_use_grid]

    merge_data_column = ['residence_result', 'power_station_result', 'conservation_result', 'temperature_result',
                         'precipitation_result', 'population_result', 'road_result', 'slope_result', 'landscape_result',
                         'wind_speed_result', 'land_use_result']

    merge_result = merge_columns(merge_data_list, merge_data_column)

    y = np.array(wind_farm_overlay_grid['wind_farm_result'], dtype='float32').reshape(-1, 1)
    x = merge_result[merge_data_column].values
    coordinates = list(zip(wind_farm_overlay_grid.centroid.x, wind_farm_overlay_grid.centroid.y))
    w_queen = weights.Queen.from_dataframe(community_council)
    # merge_result.('merge result.csv')

    # ols_result = ols(y, x, w_queen, name_y='wind_farm', name_x=merge_data_column)
    # print(ols_result.summary)

    ## GWR analysis starts
    # gwr_result = gwr(y, x, coordinates)
    # gwr_selector = Sel_BW(coordinates, y, x)
    # gwr_bw = gwr_selector.search(search_method='golden_section', criterion='AIC')
    # gwr_result.summary()
    #
    # print('Mean R2 =', gwr_result.R2)
    # print('AIC =', gwr_result.aic)
    # print('AICc =', gwr_result.aicc)
    #
    # merge_result['gwr_R2'] = gwr_result.localR2
    # fig, ax = plt.subplots(figsize=(6, 6))
    # # add_north(ax=ax)
    # merge_result.plot(column='gwr_R2', cmap='Greens', linewidth=0.01, scheme='FisherJenks', k=5, legend=True,
    #                   legend_kwds={'bbox_to_anchor': (1.10, 0.96)}, ax=ax)
    # ax.set_title('Local R2', fontsize=12)
    # # add_scale_bar(ax=ax, lon0=370000, lat0=535000)
    # ax.axis("off")
    # # add coefficient
    # merge_result['residence'] = gwr_result.params[:, 0]
    # merge_result['power_station'] = gwr_result.params[:, 1]
    # merge_result['conservation'] = gwr_result.params[:, 2]
    # merge_result['temperature'] = gwr_result.params[:, 3]
    # merge_result['precipitation'] = gwr_result.params[:, 4]
    # merge_result['population'] = gwr_result.params[:, 5]
    # merge_result['road'] = gwr_result.params[:, 6]
    # merge_result['slope'] = gwr_result.params[:, 7]
    # merge_result['landscape'] = gwr_result.params[:, 8]
    # merge_result['wind_speed'] = gwr_result.params[:, 9]
    # merge_result['land_use'] = gwr_result.params[:, 10]
    #
    # # Filter t-values: standard alpha = 0.05
    # gwr_filtered_t = gwr_result.filter_tvals(alpha=0.05)
    # pd.DataFrame(gwr_filtered_t)
    #
    # # Filter t-values: corrected alpha due to multiple testing
    # gwr_filtered_tc = gwr_result.filter_tvals()
    #
    # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    # merge_result.plot(column='residence', cmap='GnBu', linewidth=0.01, scheme='FisherJenks', k=5, legend=True,
    #                   legend_kwds={'bbox_to_anchor': (1.10, 0.96)}, ax=axes[0])
    # merge_result.plot(column='power_station', cmap='GnBu', linewidth=0.01, scheme='FisherJenks', k=5, legend=True,
    #                   legend_kwds={'bbox_to_anchor': (1.10, 0.96)}, ax=axes[1])
    # merge_result.plot(column='residence', cmap='GnBu', linewidth=0.01, scheme='FisherJenks', k=5, legend=True,
    #                   legend_kwds={'bbox_to_anchor': (1.10, 0.96)}, ax=axes[2])
    # # add_scale_bar(ax=ax[10], lon0=370000, lat0=535000)
    # # merge_result.plot(column='residence', cmap='coolwarm', linewidth=0.05, scheme='FisherJenks', k=5, legend=False,
    # #                   legend_kwds={'bbox_to_anchor': (1.10, 0.96)}, ax=axes[1])
    # # merge_result[gwr_filtered_t[:, 1] == 0].plot(color='white', linewidth=0.05, edgecolor='black', ax=axes[1])
    # #
    # # merge_result.plot(column='residence', cmap='coolwarm', linewidth=0.05, scheme='FisherJenks', k=5, legend=False,
    # #                   legend_kwds={'bbox_to_anchor': (1.10, 0.96)}, ax=axes[2])
    # # merge_result[gwr_filtered_tc[:, 1] == 0].plot(color='white', linewidth=0.05, edgecolor='black', ax=axes[2])
    #
    # plt.tight_layout()
    #
    # axes[0].axis("off")
    # axes[1].axis("off")
    # axes[2].axis("off")
    #
    # axes[0].set_title('(a) GWR: residence (BW: ' + str(gwr_bw) + '), all coeffs', fontsize=12)
    # axes[1].set_title('(b) GWR: power_station (BW: ' + str(gwr_bw) + '), all coeffs', fontsize=12)
    # axes[2].set_title('(c) GWR: conservation (BW: ' + str(gwr_bw) + '), all coeffs', fontsize=12)
    #
    # # axes[1].set_title('(b) GWR: residence (BW: ' + str(gwr_bw) + '), significant coeffs', fontsize=12)
    # # axes[2].set_title('(c) GWR: residence (BW: ' + str(gwr_bw) + '), significant coeffs and corr. p-values',fontsize=12)
    # plt.show()
    #
    # LCC, VIF, CN, VDP = gwr_result.local_collinearity()
    # pd.DataFrame(VIF)
    # pd.DataFrame(VIF).describe().round(2)
    # pd.DataFrame(CN)
    # merge_result['gwr_CN'] = CN
    # fig, ax = plt.subplots(figsize=(6, 6))
    # merge_result.plot(column='gwr_CN', cmap='coolwarm', linewidth=0.01, scheme='FisherJenks', k=5, legend=True,
    #                   legend_kwds={'bbox_to_anchor': (1.10, 0.96)}, ax=ax)
    # ax.set_title('Local multicollinearity (CN > 30)', fontsize=12)
    # ax.axis("off")
    # # plt.savefig('myMap.png',dpi=150, bbox_inches='tight')
    # plt.show()

    # MGWR analysis starts

    mgwr_result = mgwr(y, x, coordinates)
    g_x = (x - x.mean(axis=0)) / x.std(axis=0)
    g_y = (y - y.mean(axis=0)) / y.std(axis=0)
    mgwr_selector = Sel_BW(coordinates, g_y, g_x, multi=True)
    mgwr_bw = mgwr_selector.search()
    mgwr_result.summary()

    mgwr_bw_ci = mgwr_result.get_bws_intervals(mgwr_selector)
    print(mgwr_bw_ci)

    # Add MGWR parameters to GeoDataframe
    merge_result['residence'] = mgwr_result.params[:, 0]
    merge_result['power_station'] = mgwr_result.params[:, 1]
    merge_result['conservation'] = mgwr_result.params[:, 2]
    merge_result['temperature'] = mgwr_result.params[:, 3]
    merge_result['precipitation'] = mgwr_result.params[:, 4]
    merge_result['population'] = mgwr_result.params[:, 5]
    merge_result['road'] = mgwr_result.params[:, 6]
    merge_result['slope'] = mgwr_result.params[:, 7]
    merge_result['landscape'] = mgwr_result.params[:, 8]
    merge_result['wind_speed'] = mgwr_result.params[:, 9]
    merge_result['land_use'] = mgwr_result.params[:, 10]

    # Filter t-values: standard alpha = 0.05
    mgwr_filtered_t = mgwr_result.filter_tvals(alpha=0.05)

    # Filter t-values: corrected alpha due to multiple testing
    mgwr_filtered_tc = mgwr_result.filter_tvals()

    # Map coefficients
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 7), dpi=100)
    merge_result.plot(column='residence', cmap='GnBu', linewidth=0.01, scheme='FisherJenks', k=5, legend=True,
                  legend_kwds={'bbox_to_anchor': (1.10, 0.96)}, ax=axes[0, 0])
    merge_result.plot(column='power_station', cmap='GnBu', linewidth=0.01, scheme='FisherJenks', k=5, legend=True,
                      legend_kwds={'bbox_to_anchor': (1.10, 0.96)}, ax=axes[0, 1])
    merge_result.plot(column='conservation', cmap='GnBu', linewidth=0.01, scheme='FisherJenks', k=5, legend=True,
                      legend_kwds={'bbox_to_anchor': (1.10, 0.96)}, ax=axes[0, 2])
    merge_result.plot(column='temperature', cmap='GnBu', linewidth=0.01, scheme='FisherJenks', k=5, legend=True,
                      legend_kwds={'bbox_to_anchor': (1.10, 0.96)}, ax=axes[0, 3])
    merge_result.plot(column='precipitation', cmap='GnBu', linewidth=0.01, scheme='FisherJenks', k=5, legend=True,
                      legend_kwds={'bbox_to_anchor': (1.10, 0.96)}, ax=axes[1, 0])
    merge_result.plot(column='population', cmap='GnBu', linewidth=0.01, scheme='FisherJenks', k=5, legend=True,
                      legend_kwds={'bbox_to_anchor': (1.10, 0.96)}, ax=axes[1, 1])
    merge_result.plot(column='road', cmap='GnBu', linewidth=0.01, scheme='FisherJenks', k=5, legend=True,
                      legend_kwds={'bbox_to_anchor': (1.10, 0.96)}, ax=axes[1, 2])
    merge_result.plot(column='slope', cmap='GnBu', linewidth=0.01, scheme='FisherJenks', k=5, legend=True,
                      legend_kwds={'bbox_to_anchor': (1.10, 0.96)}, ax=axes[1, 3])
    merge_result.plot(column='landscape', cmap='GnBu', linewidth=0.01, scheme='FisherJenks', k=5, legend=True,
                      legend_kwds={'bbox_to_anchor': (1.10, 0.96)}, ax=axes[2, 0])
    merge_result.plot(column='wind_speed', cmap='GnBu', linewidth=0.01, scheme='FisherJenks', k=5, legend=True,
                      legend_kwds={'bbox_to_anchor': (1.10, 0.96)}, ax=axes[2, 1])
    merge_result.plot(column='land_use', cmap='GnBu', linewidth=0.01, scheme='FisherJenks', k=5, legend=True,
                      legend_kwds={'bbox_to_anchor': (1.10, 0.96)}, ax=axes[2, 2])

    plt.tight_layout()

    axes[0, 0].axis("off")
    axes[0, 1].axis("off")
    axes[0, 2].axis("off")
    axes[0, 3].axis("off")
    axes[1, 0].axis("off")
    axes[1, 1].axis("off")
    axes[1, 2].axis("off")
    axes[1, 3].axis("off")
    axes[2, 0].axis("off")
    axes[2, 1].axis("off")
    axes[2, 2].axis("off")

    axes[0, 0].set_title('(a) MGWR: residence \n (BW: ' + str(mgwr_bw) + '), all coeffs', fontsize=12)
    axes[0, 1].set_title('(b) MGWR: power_station  \n  (BW: ' + str(mgwr_bw) + '), all coeffs', fontsize=12)
    axes[0, 2].set_title('(c) MGWR: conservation  \n (BW: ' + str(mgwr_bw) + '), all coeffs', fontsize=12)
    axes[0, 3].set_title('(c) MGWR: temperature \n (BW: ' + str(mgwr_bw) + '), all coeffs', fontsize=12)
    axes[1, 0].set_title('(c) MGWR: precipitation \n (BW: ' + str(mgwr_bw) + '), all coeffs', fontsize=12)
    axes[1, 1].set_title('(c) MGWR: population \n (BW: ' + str(mgwr_bw) + '), all coeffs', fontsize=12)
    axes[1, 2].set_title('(c) MGWR: road \n (BW: ' + str(mgwr_bw) + '), all coeffs', fontsize=12)
    axes[1, 3].set_title('(c) MGWR: slope\n (BW: ' + str(mgwr_bw) + '), all coeffs', fontsize=12)
    axes[2, 0].set_title('(c) MGWR: landscape \n (BW: ' + str(mgwr_bw) + '), all coeffs', fontsize=12)
    axes[2, 1].set_title('(c) MGWR: wind_speed\n (BW: ' + str(mgwr_bw) + '), all coeffs', fontsize=12)
    axes[2, 2].set_title('(c) MGWR:land_use \n (BW: ' + str(mgwr_bw) + '), all coeffs', fontsize=12)

    # Monte Carlo test of spatial variability: 10 iterations
    mgwr_p_values_stationarity = mgwr_result.spatial_variability(mgwr_selector, 10)
    mgwr_p_values_stationarity
    # Note:  The first p-value is for the intercept

    # Test local multi-collinearity
    mgwrCN, mgwrVDP = mgwr_result.local_collinearity()
    merge_result['mgwr_CN'] = mgwrCN
    fig, ax = plt.subplots(figsize=(6, 6))
    merge_result.plot(column='mgwr_CN', cmap='GnBu', linewidth=0.01, scheme='FisherJenks', k=5, legend=True,
                      legend_kwds={'bbox_to_anchor': (1.10, 0.96)}, ax=ax)
    ax.set_title('Local multi-collinearity (CN > 30)', fontsize=12)
    ax.axis("off")
    # plt.savefig('myMap.png',dpi=150, bbox_inches='tight')
    plt.show()

    # plt.savefig('myMap.png',dpi=150, bbox_inches='tight')
    # plt.show()
    # ml_lag_result = ml_lag(y, x, w_queen, name_y='wind_farm', name_x=merge_data_column)
    # print(ml_lag_result.summary)
    #
    # ml_error_result = ml_error(y, x, w_queen, name_y='wind_farm', name_x=merge_data_column)
    # print(ml_error_result.summary)

    # gm_lag_result = gm_lag(y, x, w_queen, name_y='wind_farm', name_x=merge_data_column)
    # print(gm_lag_result.summary)
    #
    # gm_error_result = gm_error(y, x, w_queen, name_y='wind_farm', name_x=merge_data_column)
    # print(gm_error_result.summary)

    # if is_gwr_summary:
    #     method_results.summary()

    # if is_plot_coefficient:
    #     cof_data_column = ['cof_wind_farm'] + [f'cof_{col}' for col in merge_data_column]
    #
    #     gwr_coefficient = pd.DataFrame(mgwr_result.params, columns=cof_data_column)
    #     gwr_filter_t = pd.DataFrame(mgwr_result.filter_tvals())
    #
    #     x_data_geo = merge_result
    #
    #     x_data_geo = x_data_geo.join(gwr_coefficient)
    #
    #     plot_coefficient(x_data_geo, cof_data_column, gwr_filter_t)

    print('-=-=-=-=-=-=-=-=- main-function-finished-=-=-=-=-=-=-=-=-')


if __name__ == '__main__':
    plot_ward_factors()
    model_ward(is_summary=True, is_plot_coefficient=True)

    plt.show()
