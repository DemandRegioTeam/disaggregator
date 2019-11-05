# -*- coding: utf-8 -*-
# Written by Fabian P. Gotzens, 2019.

# This file is part of disaggregator.

# disaggregator is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.

# disaggregator is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
# more details.

# You should have received a copy of the GNU General Public License
# along with disaggregator.  If not, see <https://www.gnu.org/licenses/>.
"""
Provides functions for temporal disaggregations.
"""

from .config import get_config, _data_out
from .data import (elc_consumption_HH, households_per_size, population,
                   standard_load_profile_elc, zve_percentages_applications,
                   zve_percentages_baseload, zve_application_profiles,
                   database_shapes)
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import logging
logger = logging.getLogger(__name__)
cfg = get_config()


def disagg_temporal(spat, temp, time_indexed=False, **kwargs):
    """
    Disagreggate spatial data temporally through one or a set of spatial time
    series.

    Parameters
    ----------
    spat : pd.DataFrame or pd.Series
        Container for the spatial data; NUTS-3-index'ed
    temp : pd.DataFrame or pd.Series
        Container for the time series containing the temporal data.
        If pd.Series: pd.DatetimeIndex'ed
        If pd.DataFrame: NUTS-3-index'ed
    """
    # Make sure spat is a pd.DataFrame and determine dimension
    if isinstance(spat, pd.DataFrame):
        spat = spat.unstack()
    if not isinstance(spat, pd.Series):
        raise ValueError('`spat` must be either a pd.Series or a '
                         'one-level-indexed (!) pd.DataFrame.')

    if isinstance(temp, pd.Series):
        # Simple time series to be applied to all regions
        temp /= temp.sum()  # Normalize temporal data
        if time_indexed:
            return pd.DataFrame(np.outer(temp, spat),
                                index=temp.index, columns=spat.index)
        else:
            return pd.DataFrame(np.outer(spat, temp),
                                index=spat.index, columns=temp.index)
    elif isinstance(temp, pd.DataFrame):
        # Exising time series for all regions
        temp = temp.div(temp.sum(axis=1), axis='index')  # Normalize
        if time_indexed:
            return temp.multiply(spat, axis=0).T
        else:
            return temp.multiply(spat, axis=0)
    else:
        raise ValueError('`temp` must be either a pd.Series or a '
                         'one-level-indexed (!) pd.DataFrame.')


def make_zve_load_profiles(return_basic_profile=False, **kwargs):
    """
    Make load profiles based on the ZVE (German: "Zeitverwendungserhebung").

    Parameters
    ----------
    nTsLP : int, default 96
        number of time steps in load profiles

    Returns
    -------
    pd.DataFrame
    """
    year = kwargs.get('year', cfg['base_year'])
    reg = kwargs.get('reg', None)
    # Load number of households per size
    df_HH = households_per_size()
    df_HH[5] = df_HH[5] + df_HH[6]
    df_HH.drop(labels=6, axis=1, inplace=True)
    n_HH = len(df_HH.columns)

    # Load Specific electricity consumption by household size
    df_elc_cons_HH = elc_consumption_HH(by_HH_size=True)[0:5] * 1e3

    # Determine percentage of load share of each size
    df_elc_HH = df_HH.multiply(df_elc_cons_HH)
    df_elc_HH_share = df_elc_HH.divide(df_elc_HH.sum(axis=1), axis='index')

    df_perc_app = zve_percentages_applications()
    df_app_prof = zve_application_profiles()
    df_perc_baseload = zve_percentages_baseload()
    df_perc_activityload = df_perc_baseload.apply(lambda x: 1-x)

    nTsLP = 96  # number of time steps in load profiles
    df_baseload = (df_perc_baseload * df_perc_app) / float(nTsLP)
    df_activityload = df_perc_activityload * df_perc_app
    n_app_activity = 9  # number of activity-based applications
    n_app_all = len(df_perc_baseload)  # number of all applications
    l_app_all = df_perc_baseload.index.tolist()
    l_app_activity = df_perc_baseload[:-3].index.tolist()
    l_app_baseload = df_perc_baseload[-3:].index.tolist()

    # TODO: Only valid for 2012. Make this dynamic for user-defined years,
    # e.g. by instantiating a class for each time slice!
    time_slices_to_days = {'WD_Win': 103,
                           'WD_Tra': 61,
                           'WD_Sum': 85,
                           'SA_Win': 24,
                           'SA_Tra': 13,
                           'SA_Sum': 17,
                           'SU_Win': 25,
                           'SU_Tra': 17,
                           'SU_Sum': 21}
    time_slices = ['WD_Win', 'WD_Tra', 'WD_Sum',
                   'SA_Win', 'SA_Tra', 'SA_Sum',
                   'SU_Win', 'SU_Tra', 'SU_Sum']
    n_ts = len(time_slices_to_days)  # number of time slices

    DE = database_shapes()
    assert(isinstance(DE, gpd.GeoDataFrame))
    # Derive lat/lon tuple as representative point for each shape
    DE['coords'] = DE.to_crs({'init': 'epsg:4326'}).geometry.apply(
            lambda x: x.representative_point().coords[:][0])
    idx = pd.date_range(start='{}-01-01'.format(year),
                        end='{}-12-31 23:00'.format(year), freq='1H')
    if reg is not None:
        DE = DE.loc[reg].to_frame().T
    DF = pd.DataFrame(index=idx, columns=DE.index)
    for reg, row in DE.iterrows():
        logger.info('Creating ZVE-profile for {} ({})'.format(row['gen'], reg))
        lat = row.coords[1]
        lon = row.coords[0]
        prob_night = probability_light_needed(lat=lat, lon=lon, nTsLP=nTsLP)

        time_idx = pd.date_range(start='2012-01-02 00:00:00',
                                 end='2012-01-02 23:45:00', freq='15Min')
        LP_15 = xr.DataArray(np.zeros((nTsLP, n_app_activity, n_ts, n_HH)),
                             dims=['Time', 'Application',
                                   'TimeSlice', 'HH_size'],
                             coords=[time_idx, df_perc_app.index[0:9],
                                     time_slices, df_HH.columns])
        for HH_size, df_HH_size in df_app_prof.groupby('HH_size'):
            for id_day, df_day in df_HH_size.groupby('Day'):
                for id_season, df_season in df_day.groupby('Season'):
                    id_ts = (id_day-1)*3 + id_season - 1
                    for id_app, df_app in df_season.groupby('Application'):
                        LP_15[:, id_app-1, id_ts, HH_size-1] = (
                                df_app.iloc[0, 4:100])

        LP_Fin = xr.DataArray(np.zeros((nTsLP, n_app_all, n_ts), dtype=float),
                              dims=['Time', 'Application', 'TimeSlice'],
                              coords=[time_idx, l_app_all, time_slices])

        LP_HHGr = xr.DataArray(np.zeros((nTsLP, n_ts, n_HH), dtype=float),
                               dims=['Time', 'TimeSlice', 'HH_size'],
                               coords=[time_idx, time_slices, df_HH.columns])

        for i_HH, HH_size in enumerate(df_elc_HH.columns):
            logger.info('Calc load profile for household-size: {}'
                        .format(HH_size))

            # Multiplication of a person's presence (activity_id = 0) with the
            # probability of night, which gives the probability of needed light
            year_sum = 0.0
            for i_ts, ts in enumerate(time_slices):
                j = i_ts % 3
                LP_15[:, 0, i_ts, i_HH] *= prob_night[:, j]
                year_sum += (LP_15[:, 0, i_ts, i_HH].sum(axis=0).item()
                             * time_slices_to_days[ts])
            # Light (i_app=0) needs to be normalized to a daily sum = 1.
            LP_15[:, 0, :, i_HH] *= 366.0 / year_sum

            # Normalize something.
            # TODO: Understand what is intended here - just copy&pasted yet.
            norm_factor_0 = (nTsLP / 24.) * 1000000. / 366.
            norm_factor = norm_factor_0 * df_elc_HH_share.loc[reg, HH_size]
            # Loop over activity-based applications
            for i_app, app in enumerate(l_app_activity):
                for i_ts, ts in enumerate(time_slices):
                    LP_tmp = (LP_15[:, i_app, i_ts, i_HH]
                              * df_activityload.loc[app, HH_size]
                              + df_baseload.loc[app, HH_size])
                    LP_Fin[:, i_app, i_ts] += LP_tmp.values * norm_factor
                    LP_HHGr[:, i_ts, i_HH] += LP_tmp.values * norm_factor_0
            # Loop over baseload applications
            for i_app, app in enumerate(l_app_baseload, start=n_app_activity):
                LP_Fin[:, i_app, :] += (df_baseload.loc[app, HH_size]
                                        * norm_factor)
                LP_HHGr[:, :, i_HH] += (df_baseload.loc[app, HH_size]
                                        * norm_factor_0)

        logger.info('...creating hourly load profile for entire year...')
        # relevant result
        df_erg = LP_HHGr.sum('HH_size').to_pandas()
        if return_basic_profile:
            return df_erg
        df_erg.columns = pd.MultiIndex.from_product([['WD', 'SA', 'SU'],
                                                     ['Win', 'Tra', 'Sum']])

        # Resample to hourly values:
        df_erg = df_erg.resample('1H').sum().reset_index(drop=True)
        # Create a 8760-hour time series out of these profiles for given year
        dic_month_to_season = {1: 'Win', 2: 'Win', 3: 'Win', 4: 'Tra',
                               5: 'Sum', 6: 'Sum', 7: 'Sum', 8: 'Sum',
                               9: 'Tra', 10: 'Tra', 11: 'Win', 12: 'Win'}
        dic_day_to_typeday = {0: 'WD', 1: 'WD', 2: 'WD', 3: 'WD', 4: 'WD',
                              5: 'SA', 6: 'SU'}
        df_new = (pd.DataFrame(index=idx)
                    .assign(Season=lambda x: x.index.month,
                            TypeDay=lambda x: x.index.dayofweek,
                            Hour=lambda x: x.index.hour)
                    .replace(dict(Season=dic_month_to_season))
                    .replace(dict(TypeDay=dic_day_to_typeday)))
        # assign values # TODO: this is comparatively slow, make more efficient
        for i in idx:
            ind = df_new.loc[i, 'Hour']
            col = (df_new.loc[i, 'TypeDay'], df_new.loc[i, 'Season'])
            df_new.loc[i, 'value'] = df_erg.loc[ind, col]
        # generate distribution keys
        df_new = df_new['value'] / df_new['value'].sum()
        DF.loc[:, reg] = df_new

    # Looping over all regions is done. Now save and return
    if reg is None:
        reg = 'AllRegions'
    f = 'ZVE_timeseries_{}_{}.csv'.format(reg, year)
    DF.to_csv(_data_out(f), encoding='utf-8')
    return DF


def create_projection(df, target_year, **kwargs):
    """
    Create a future projection for a given dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset that is going to be projected; NUTS-3-index'ed
    target_year : int
        The future year.

    Returns
    -------
    pd.DataFrame
    """
    year = kwargs.get('year', cfg['base_year'])
    if not isinstance(target_year, int):
        raise ValueError('`target_year` must be an int.')

    keys = population(year=target_year) / population(year=year)
    return df.multiply(keys, axis='index')


# --- Utility functions -------------------------------------------------------


def getSunsetSunrise(doy, lat, lon, UTC_diff):
    """
    Approximate the time of sunrise/sunset (±5 Min), based on day-of-year for
    given lat/lon-coordinates. Daylight saving is being respected.

    Source for astronomical background:
    https://lexikon.astronomie.info/zeitgleichung/index.html
    Jan Hartmann 2018 (mailto:jh1319@posteo.de) License:
    "This script may be used, passed on and modified without restrictions."

    Parameters
    ----------
    doy : int
        day of year
    lat : float
        latitute
    lon : float
        longitude
    UTC_diff : int
        difference to UTC

    Returns
    -------
    tuple
    """
    import math
    B = math.pi * lat / 180
    declination_sun = 0.4095 * math.sin(0.016906 * (doy-80.086))

    # elevation angle at which there is dawn (i.e. daylight available)
    # civil dawn:         -6° below horizon
    # nautical dawn:     -12° below horizon
    # astronomical dawn: -18° below horizon
    # Here, we choose +12 because we still/already need electric light in the
    # sunrise/sunset hours.
    light_h = 12.
    # Transform into radiants
    sunset_h = light_h/(180/math.pi)
    # time difference
    time_diff_arg = ((math.sin(sunset_h)-math.sin(B) *
                      math.sin(declination_sun)) /
                     (math.cos(B)*math.cos(declination_sun)))
    time_diff_arg = min(time_diff_arg, 1.)
    time_diff = 12 * math.acos(time_diff_arg) / math.pi
    time_equation = (-0.171 * math.sin(0.0337*doy+0.465) - 0.1299 *
                     math.sin(0.01787*doy-0.168))
    # Sunrise / sunset at central local time (MOZ)
    sunset_MOZ = 12 + time_diff - time_equation
    sunrise_MOZ = 12 - time_diff - time_equation
    # Difference between MOZ und MEZ/MESZ
    MOZ_MEZ_diff = -lon/15.0 + UTC_diff
    # Time in MEZ/MESZ
    sunset = sunset_MOZ + MOZ_MEZ_diff
    sunrise = sunrise_MOZ + MOZ_MEZ_diff
    return sunset, sunrise


def probability_light_needed(lat, lon, nTsLP=96):
    """
    Calculates the probability if artifical light is needed for each time step
    per day (0:00 - 23:45) during the months in `season_months`.

    Parameters
    ----------
    lat : float
        latitute
    lon : float
        longitude
    nTsLP : int, default 96
        number of time steps in load profiles

    Returns
    -------
    xr.DataArray
    """
    lat, lon = float(lat), float(lon)
    # -- Static values for the year 2012 in which evaluation was carried out --
    month_lengths = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_first_day = [0] + np.cumsum(month_lengths[:-1]).tolist()
    SummerTimeBegin = 87  # only valid for 2012
    WinterTimeBegin = 301  # only valid for 2012
    # -------------------------------------------------------------------------
    timezone = 1  # Germany: Winter time minus UTC
    hoursBeforeSunset = 0.0
    seasons = ['Winter', 'Transition', 'Summer']
    season_months = {'Winter':     [1, 2, 3, 11, 12],
                     'Transition': [4, 9, 10],
                     'Summer':     [5, 6, 7, 8]}

    time_idx = pd.date_range(start='2012-01-02 00:00:00',
                             end='2012-01-02 23:45:00', freq='15Min')
    p_night = xr.DataArray(np.zeros((nTsLP, len(seasons)), dtype=np.float32),
                           dims=['Time', 'Season'],
                           coords=[time_idx, list(range(3))])

    for season_id, season in enumerate(seasons):
        iadd = 0
        for m in season_months[season]:
            iadd += month_lengths[m-1]  # Count of days in the season
        add = 1./float(iadd)  # Percentage of one day in that season
        TsH = nTsLP / 24.  # Number of time steps per Hour
        for m in season_months[season]:
            doy = float(month_first_day[m-1])  # doy is 'day of year'
            for d in range(0, month_lengths[m-1]):
                doy += 1.
                # min(max(doy-SummerTimeBegin,0),1)-min(max(doy-WinterTimeBegin,0),1)
                # gives 1 during summer time, else 0
                UTC_diff = float(timezone + min(max(doy-SummerTimeBegin, 0), 1)
                                 - min(max(doy-WinterTimeBegin, 0), 1))
                sunset, sunrise = getSunsetSunrise(doy, lat, lon, UTC_diff)
                # NEnd, NBeg are integer indices of the nTsLP
                # sunrise + hoursBeforeSunset :
                # Light is needed until 'hoursBeforeSunset' after sunrise
                # sunset - hoursBeforeSunset :
                # Light is needed 'hoursBeforeSunset' until sunset
                NEnd = int((sunrise+hoursBeforeSunset)*TsH)  # round floor
                NBeg = int((sunset-hoursBeforeSunset)*TsH+0.999)  # round ceil
                p_night[0:NEnd, season_id] += add
                p_night[NBeg:nTsLP, season_id] += add

    return p_night
