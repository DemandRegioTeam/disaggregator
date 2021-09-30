# -*- coding: utf-8 -*-
# Written by Fabian P. Gotzens, Paul Verwiebe, Maike Held 2019/2020.

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
# %% Imports

from .config import (get_config, data_out, bl_dict, shift_profile_industry,
                     slp_branch_cts_gas as slp_wz_g, data_in,
                     slp_branch_cts_power as slp_wz_p, dict_region_code,
                     slp_household_gas, blp_branch_cts_power, blp_wz_list)
from .data import (elc_consumption_HH, households_per_size, population,
                   living_space, h_value, zve_percentages_applications,
                   zve_percentages_baseload, zve_application_profiles,
                   database_shapes, CTS_power_slp_generator, t_allo,
                   shift_load_profile_generator, gas_slp_weekday_params,
                   percentage_EFH_MFH, regional_branch_load_profiles)
from .spatial import (disagg_CTS_industry, disagg_households_power,
                      disagg_households_gas)
from datetime import timedelta
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import logging
logger = logging.getLogger(__name__)


# %% Generic functions


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
    time_indexed : bool, default False
        Option to return the timesteps in index and regions in columns

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


def make_zve_load_profiles(return_profile_by_typeday=False,
                           return_profile_by_application=False, **kwargs):
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
    cfg = kwargs.get('cfg', get_config())
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

    # WARNING: Only valid for 2012.
    # Future TODO: Make this dynamic for user-defined years,
    # e.g. by instantiating a class for each time slice
    # to achieve a little more accuracy.
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
    for region, row in DE.iterrows():
        logger.info('Creating ZVE-profile for {} ({})'.format(row['gen'],
                                                              region))
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
            norm_factor = norm_factor_0 * df_elc_HH_share.loc[region, HH_size]
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
        if return_profile_by_typeday:
            return df_erg
        if return_profile_by_application:
            app = kwargs.get('app', 'WD_Win')
            return LP_Fin.loc[:, :, app].to_pandas()

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
        DF.loc[:, region] = df_new

    # Looping over all regions is done. Now save and return
    if reg is None:
        reg = 'AllRegions'
    f = 'ZVE_timeseries_{}_{}.csv'.format(reg, year)
    DF.to_csv(data_out(f), encoding='utf-8')
    return DF


def create_projection(df, target_year, by, **kwargs):
    """
    Create a future projection for a given dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset that is going to be projected; NUTS-3-index'ed
    target_year : int
        The future year.
    by : str
        The parameter to base the projection on.

    Returns
    -------
    pd.DataFrame
    """
    cfg = kwargs.get('cfg', get_config())
    year = kwargs.get('year', cfg['base_year'])
    if not isinstance(target_year, int):
        raise ValueError('`target_year` must be an int.')

    if by == 'population':
        keys = population(year=target_year) / population(year=year)
    elif by == 'living_space':
        keys = living_space(year=target_year) / living_space(year=year)
    else:
        raise ValueError("`by` must be in ['population', 'living_space'.")
    return df.multiply(keys, axis='index')


# %% Utility functions


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
    time_diff_arg = ((math.sin(sunset_h)
                      - math.sin(B) * math.sin(declination_sun))
                     / (math.cos(B) * math.cos(declination_sun)))
    time_diff_arg = min(time_diff_arg, 1.)
    time_diff = 12 * math.acos(time_diff_arg) / math.pi
    time_equation = (-0.171 * math.sin(0.0337*doy + 0.465)
                     - 0.1299 * math.sin(0.01787*doy - 0.168))
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


def disagg_temporal_power_CTS_blp(detailed=False, use_nuts3code=False,
                                  **kwargs):
    """
    Disagreggate spatial data of CTS' power demand temporally.

    Parameters
    ----------
    detailed : bool, default False
        If True return 'per district and branch' else only 'per district'
    use_nuts3code : bool, default False
        If True use NUTS-3 codes as region identifiers.
    Returns
    -------
    pd.DataFrame
    """
    cfg = kwargs.get('cfg', get_config())
    year = kwargs.get('year', cfg['base_year'])
    if year not in range(2009, 2020):
        raise ValueError("to use blp, `year` must be between 2009 and 2019.")
    # Obtain yearly power consumption per WZ per LK
    sv_yearly = (disagg_CTS_industry('power', 'CTS'))
    # splitting yearly demand in two groups. slp group are disaggregated with
    # slp. blp group are disaggregated with blp (=profiles based on metered
    # load data from the demandregio project)
    blp_ghd = ([x for x in slp_wz_p().keys()
                if x in blp_wz_list()])
    slp_ghd = ([x for x in slp_wz_p().keys()
                if x not in blp_wz_list()])

    ec_slp = (sv_yearly.reindex(index=slp_ghd).transpose()
                       .assign(BL=lambda x: [bl_dict().get(int(i[: -3]))
                                             for i in x.index.astype(str)]))
    ec_blp = sv_yearly.reindex(index=blp_ghd)

    total_sum = sv_yearly.sum().sum()

    # start with temporal disaggregation of slp group
    # Create empty 15min-index'ed DataFrame for target year
    idx = pd.date_range(start=str(year), end=str(year+1), freq='15T')[:-1]
    DF = pd.DataFrame(index=idx)

    for state in bl_dict().values():
        logger.info('Working on state: {}.'.format(state))
        sv_lk_wz = (ec_slp
                    .loc[lambda x: x['BL'] == state]
                    .drop(columns=['BL'])
                    .fillna(value=0)
                    .transpose()
                    .assign(SLP=lambda x: [slp_wz_p()[i] for i in x.index]))

        logger.info('... creating state-specific load-profiles')
        slp_bl = CTS_power_slp_generator(state, year=year)
        # Plausibility check:
        assert slp_bl.index.equals(idx), "The time-indizes are not aligned"
        # Create 15min-index'ed DataFrames for current state
        if detailed:
            sv_lk_wz_ts = pd.DataFrame(index=idx)
        else:
            cols = sv_lk_wz.drop(columns=['SLP']).columns
            sv_lk_ts = pd.DataFrame(index=idx, columns=cols).fillna(0.0)

        logger.info('... assigning load-profiles to WZs')
        for slp in sv_lk_wz['SLP'].unique():
            sv_lk = (sv_lk_wz.loc[sv_lk_wz['SLP'] == slp]
                             .drop(columns=['SLP']).stack().reset_index())
            sv_dtl_df = sv_lk.groupby(by=['level_1'])[[0]].sum().transpose()
            sv_lk = (sv_lk.assign(LK_WZ=lambda x: x.level_1.astype(str) + '_'
                                  + x.level_0.astype(str))
                     .set_index('LK_WZ')
                     .drop(['level_0', 'level_1'], axis=1)
                     .loc[lambda x: x[0] >= 0]
                     .transpose())

            if detailed:  # Calculate load profile for each LK and WZ
                lp_lk_wz = (pd.DataFrame(np.multiply(slp_bl[[slp]].values,
                                                     sv_lk.values),
                                         index=slp_bl.index,
                                         columns=sv_lk.columns))
            else:  # Calculate load profile for each LK
                lp_lk = (pd.DataFrame(np.multiply(slp_bl[[slp]].values,
                                                  sv_dtl_df.values),
                                      index=slp_bl.index,
                                      columns=sv_dtl_df.columns))
            # Merge intermediate results
            if detailed:
                sv_lk_wz_ts = (sv_lk_wz_ts.merge(lp_lk_wz, left_index=True,
                                                 right_index=True,
                                                 suffixes=(False, False)))
            else:
                sv_lk_ts += lp_lk

        # Concatenate the state-wise results
        if detailed:  # restore MultiIndex as integer tuples
            sv_lk_wz_ts.columns =\
                pd.MultiIndex.from_tuples([(int(x), int(y)) for x, y in
                                           sv_lk_wz_ts.columns.str.split('_')])
            DF = pd.concat([DF, sv_lk_wz_ts], axis=1)
            DF.columns = pd.MultiIndex.from_tuples(DF.columns,
                                                   names=['LK', 'WZ'])
        else:
            DF = pd.concat([DF, sv_lk_ts], axis=1).dropna()

    # start with temporal disaggregation of blp group
    if detailed:
        DF2 = pd.DataFrame()
        logger.info('Start downloading BLP from Database now. This may take a'
                    ' while depending on the connection. ca. 250MB per industr'
                    'y will be downloaded and stored in your local folder ../d'
                    'ata_in/cache.')
        branch_model_64 = None
        branch_model_41 = None
        branch_model = None
        for branch in blp_ghd:
            if branch in [58, 59, 64, 65, 66, 67, 68, 69, 70, 71,
                          73, 74, 75, 78, 95, 96, 99]:
                # these branches use the office blp, which is wz=64
                logger.info('Working on WZ'+str(branch))
                if branch_model_64 is None:
                    branch_model_64 = regional_branch_load_profiles(wz=64,
                                                                    year=year)
                demand_branch = (np.multiply(branch_model_64,
                                             ec_blp.loc[ec_blp.index
                                                        == branch]))
            elif branch in [42, 43]:
                # these branches use the same blp, which is wz=41
                logger.info('Working on WZ' + str(branch))
                if branch_model_41 is None:
                    branch_model_41 = regional_branch_load_profiles(wz=41,
                                                                    year=year)
                demand_branch = (np.multiply(branch_model_41,
                                             ec_blp.loc[ec_blp.index
                                                        == branch]))
            else:
                # rest of the branches uses individual blp
                logger.info('Working on WZ' + str(branch))
                branch_model = regional_branch_load_profiles(wz=branch,
                                                             year=year)
                demand_branch = (np.multiply(branch_model,
                                             ec_blp.loc[ec_blp
                                                        .index == branch]))
            # concat demand_branch to DF2, DF2 has same index as DF
            DF2 = pd.concat([DF2, (demand_branch.T.assign(WZ=branch)
                                   .set_index('WZ', append=True).T)], axis=1)

        DF2 = DF2/4000  # convert from GW to MWh
        total_demand = DF.join(DF2)

    else:
        DF2 = (pd.DataFrame(0, index=pd.date_range(freq='15Min',
                                                   start='01/01/'+str(year),
                                                   periods=len(DF),
                                                   ),
                            columns=ec_blp.columns))
        branch_model_64 = None
        branch_model_41 = None
        branch_model = None
        logger.info('Start downloading BLP from Database now. This may take a '
                    'while depending on the connection. ca. 250MB per industry'
                    ' will be downloaded and stored in your local folder ../da'
                    'ta_in/cache.')
        for branch in blp_ghd:
            if branch in [58, 59, 64, 65, 66, 67, 68, 69, 70, 71,
                          73, 74, 75, 78, 95, 96, 99]:
                # these branches use the office blp, which is wz=64
                logger.info('Working on WZ'+str(branch))
                if branch_model_64 is None:
                    branch_model_64 = regional_branch_load_profiles(wz=64,
                                                                    year=year)
                demand_branch = (np.multiply(branch_model_64,
                                             ec_blp.loc[ec_blp.index
                                                        == branch]))
            elif branch in [42, 43]:
                # these branches use the same blp, which is wz=41
                logger.info('Working on WZ' + str(branch))
                if branch_model_41 is None:
                    branch_model_41 = regional_branch_load_profiles(wz=41,
                                                                    year=year)
                demand_branch = (np.multiply(branch_model_41,
                                             ec_blp.loc[ec_blp.index
                                                        == branch]))
            else:
                # rest of the branches uses individual blp
                logger.info('Working on WZ' + str(branch))
                branch_model = regional_branch_load_profiles(wz=branch,
                                                             year=year)
                demand_branch = (np.multiply(branch_model,
                                             ec_blp.loc[ec_blp.index
                                                        == branch]))
            DF2 = DF2.add(demand_branch, axis=1, fill_value=0)
        DF2 = DF2/4000  # convert from GW to MWh
        total_demand = DF.add(DF2, axis=1, fill_value=0)

    # Plausibility check:
    msg = ('The sum of yearly consumptions (={:.3f}) and the sum of disaggrega'
           'ted consumptions (={:.3f}) do not match! Please check algorithm!')
    disagg_sum = total_demand.sum().sum()
    assert np.isclose(total_sum, disagg_sum,
                      rtol=1.e-3), msg.format(total_sum, disagg_sum)

    if use_nuts3code:
        total_demand = total_demand.rename(columns=dict_region_code(
            level='lk', keys='ags_lk', values='natcode_nuts3'),
            level=(0 if detailed else None))
    return total_demand


def disagg_temporal_power_CTS(detailed=False, use_nuts3code=False, **kwargs):
    """
    Disagreggate spatial data of CTS' power demand temporally.

    Parameters
    ----------
    detailed : bool, default False
        If True return 'per district and branch' else only 'per district'
    use_nuts3code : bool, default False
        If True use NUTS-3 codes as region identifiers.
    Returns
    -------
    pd.DataFrame
    """
    cfg = kwargs.get('cfg', get_config())
    year = kwargs.get('year', cfg['base_year'])
    # Obtain yearly power consumption per WZ per LK
    sv_yearly = (disagg_CTS_industry('power', 'CTS')
                 .transpose()
                 .assign(BL=lambda x: [bl_dict().get(int(i[: -3]))
                                       for i in x.index.astype(str)]))
    total_sum = sv_yearly.drop('BL', axis=1).sum().sum()

    # Create empty 15min-index'ed DataFrame for target year
    idx = pd.date_range(start=str(year), end=str(year+1), freq='15T')[:-1]
    DF = pd.DataFrame(index=idx)

    for state in bl_dict().values():
        logger.info('Working on state: {}.'.format(state))
        sv_lk_wz = (sv_yearly
                    .loc[lambda x: x['BL'] == state]
                    .drop(columns=['BL'])
                    .transpose()
                    .assign(SLP=lambda x: [slp_wz_p()[i] for i in x.index]))

        logger.info('... creating state-specific load-profiles')
        slp_bl = CTS_power_slp_generator(state, year=year)
        # Plausibility check:
        assert slp_bl.index.equals(idx), "The time-indizes are not aligned"
        # Create 15min-index'ed DataFrames for current state
        if detailed:
            sv_lk_wz_ts = pd.DataFrame(index=idx)
        else:
            cols = sv_lk_wz.drop(columns=['SLP']).columns
            sv_lk_ts = pd.DataFrame(index=idx, columns=cols).fillna(0.0)

        logger.info('... assigning load-profiles to WZs')
        for slp in sv_lk_wz['SLP'].unique():
            sv_lk = (sv_lk_wz.loc[sv_lk_wz['SLP'] == slp]
                             .drop(columns=['SLP']).stack().reset_index())
            sv_dtl_df = sv_lk.groupby(by=['level_1'])[[0]].sum().transpose()
            sv_lk = (sv_lk.assign(LK_WZ=lambda x: x.level_1.astype(str) + '_'
                                  + x.level_0.astype(str))
                     .set_index('LK_WZ')
                     .drop(['level_0', 'level_1'], axis=1)
                     .loc[lambda x: x[0] >= 0]
                     .transpose())

            if detailed:  # Calculate load profile for each LK and WZ
                lp_lk_wz = (pd.DataFrame(np.multiply(slp_bl[[slp]].values,
                                                     sv_lk.values),
                                         index=slp_bl.index,
                                         columns=sv_lk.columns))
            else:  # Calculate load profile for each LK
                lp_lk = (pd.DataFrame(np.multiply(slp_bl[[slp]].values,
                                                  sv_dtl_df.values),
                                      index=slp_bl.index,
                                      columns=sv_dtl_df.columns))
            # Merge intermediate results
            if detailed:
                sv_lk_wz_ts = (sv_lk_wz_ts.merge(lp_lk_wz, left_index=True,
                                                 right_index=True,
                                                 suffixes=(False, False)))
            else:
                sv_lk_ts += lp_lk

        # Concatenate the state-wise results
        if detailed:  # restore MultiIndex as integer tuples
            sv_lk_wz_ts.columns =\
                pd.MultiIndex.from_tuples([(int(x), int(y)) for x, y in
                                           sv_lk_wz_ts.columns.str.split('_')])
            DF = pd.concat([DF, sv_lk_wz_ts], axis=1)
            DF.columns = pd.MultiIndex.from_tuples(DF.columns,
                                                   names=['LK', 'WZ'])
        else:
            DF = pd.concat([DF, sv_lk_ts], axis=1).dropna()

    # Plausibility check:
    msg = ('The sum of yearly consumptions (={:.3f}) and the sum of disaggrega'
           'ted consumptions (={:.3f}) do not match! Please check algorithm!')
    disagg_sum = DF.sum().sum()
    assert np.isclose(total_sum, disagg_sum), msg.format(total_sum, disagg_sum)

    if use_nuts3code:
        DF = DF.rename(columns=dict_region_code(level='lk', keys='ags_lk',
                                                values='natcode_nuts3'),
                       level=(0 if detailed else None))
    return DF


def disagg_temporal_power_housholds_slp(use_nuts3code=False,
                                        by='population',
                                        weight_by_income=False,
                                        **kwargs):
    """
    Disagreggate spatial data of households' power demand temporally.

    Parameters
    ----------
    use_nuts3code : bool, default False
        If True use NUTS-3 codes as region identifiers.
    dyn = TODO, bool, default False,
        If True use dynamic factors for household
    by : str, default 'population'
        throughput from function spatial.disagg_households_power(),
        must be one of ['households', 'population']
    weight_by_income : bool, optional, default False
        throughput from function spatial.disagg_households_power(),
        Flag if to weight the results by the regional income
    Returns
    -------
    pd.DataFrame
    """
    cfg = kwargs.get('cfg', get_config())
    year = kwargs.get('year', cfg['base_year'])
    # Obtain yearly power consumption per per district
    sv_yearly = ((disagg_households_power(by=by,
                                          weight_by_income=weight_by_income,
                                          year=year)
                  * 1e3)
                 .rename(index=dict_region_code(keys='natcode_nuts3',
                                                values='ags_lk'))
                 .to_frame()
                 .assign(BL=lambda x: [bl_dict().get(int(i[: -3]))
                                       for i in x.index.astype(str)]))

    total_sum = sv_yearly.value.sum()

    # Create empty 15min-index'ed DataFrame for target year
    idx = pd.date_range(start=str(year), end=str(year+1), freq='15T')[:-1]
    DF = pd.DataFrame(index=idx)

    for state in bl_dict().values():
        logger.info('Working on state: {}.'.format(state))
        sv_lk = (sv_yearly
                 .loc[lambda x: x['BL'] == state]
                 .drop(columns=['BL'])
                 .assign(SLP=lambda x: 'H0'))
        logger.info('... creating state-specific load-profiles')
        slp_bl = CTS_power_slp_generator(state, year=year)
        # Plausibility check:
        assert slp_bl.index.equals(idx), "The time-indizes are not aligned"
        # Create 15min-index'ed DataFrames for current state
        cols = sv_lk.drop(columns=['SLP']).columns
        sv_lk_ts = pd.DataFrame(index=idx, columns=cols).fillna(0.0)

        logger.info('... assigning load-profiles')
        # Calculate load profile for each LK
        slp = 'H0'
        lp_lk = (pd.DataFrame(np.multiply(slp_bl[[slp]].values,
                                          sv_lk.drop(columns=['SLP'])
                                          .transpose().values),
                              index=slp_bl.index,
                              columns=sv_lk.index))
        # save intermediate results
        sv_lk_ts = pd.concat([sv_lk_ts, lp_lk], axis=1).drop(columns=['value'])

        # Concatenate the state-wise results
        DF = pd.concat([DF, sv_lk_ts], axis=1).dropna()

    # Plausibility check:
    msg = ('The sum of yearly consumptions (={:.3f}) and the sum of disaggrega'
           'ted consumptions (={:.3f}) do not match! Please check algorithm!')
    disagg_sum = DF.sum().sum()
    assert np.isclose(total_sum, disagg_sum), msg.format(total_sum, disagg_sum)

    if use_nuts3code:
        DF = DF.rename(columns=dict_region_code(level='lk', keys='ags_lk',
                                                values='natcode_nuts3'))
    return DF


def disagg_daily_gas_slp_cts(state, temperatur_df, **kwargs):
    """
    Returns daily demand of gas with a given yearly demand in MWh
    per district and SLP.

    state: str
        must be one of ['BW','BY','BE','BB','HB','HH','HE','MV',
                        'NI','NW','RP','SL','SN','ST','SH','TH']
    Returns
    -------
    pd.DataFrame
    """
    cfg = kwargs.get('cfg', get_config())
    year = kwargs.get('year', cfg['base_year'])
    if ((year % 4 == 0)
            & (year % 100 != 0)
            | (year % 4 == 0)
            & (year % 100 == 0)
            & (year % 400 == 0)):
        days = 366
    else:
        days = 365
    gv_lk = disagg_CTS_industry('gas', 'CTS', year=year).transpose()
    gv_lk_return = gv_lk.copy()
    gv_lk = (gv_lk.assign(BL=[bl_dict().get(int(x[: -3]))
                          for x in gv_lk.index.astype(str)]))
    df = pd.DataFrame(index=range(days))
    gv_lk = gv_lk.loc[gv_lk['BL'] == state].drop(columns=['BL']).transpose()
    list_ags = gv_lk.columns.astype(str)
    gv_lk['SLP'] = [slp_wz_g()[x] for x in (gv_lk.index)]
    F_wd = (gas_slp_weekday_params(state, year=year)
            .drop(columns=['MO', 'DI', 'MI', 'DO', 'FR', 'SA', 'SO'])
            .set_index('Date'))
    tageswerte = pd.DataFrame(index=F_wd.index)
    logger.info('... creating state-specific load-profiles')
    for slp in gv_lk['SLP'].unique():
        F_wd_slp = F_wd[['FW_'+slp]]
        h_slp = h_value(slp, list_ags, temperatur_df)

        if (len(h_slp) != len(F_wd_slp)):
            raise KeyError('The chosen historical weather year and the chosen '
                           'projected year have mismatching lengths.'
                           'This could be due to gap years. Please change the '
                           'historical year in hist_weather_year() in '
                           'config.py to a year of matching length.')

        tw = pd.DataFrame(np.multiply(h_slp.values, F_wd_slp.values),
                          index=h_slp.index, columns=h_slp.columns)
        tw_norm = tw/tw.sum()
        gv_df = (gv_lk.loc[gv_lk['SLP'] == slp].drop(columns=['SLP'])
                      .stack().reset_index())
        tw_lk_wz = pd.DataFrame(index=h_slp.index)
        for lk in gv_df['level_1'].unique():
            gv_slp = (gv_df.loc[gv_df['level_1'] == lk]
                           .drop(columns=['level_1'])
                           .set_index('level_0').transpose()
                           .rename(columns=lambda x: str(lk) + '_' + str(x)))
            tw_lk_wz_slp = (pd.DataFrame(np.multiply(tw_norm[
                                                     [str(lk)]
                                                     * len(gv_slp.columns)]
                                                     .values, gv_slp.values),
                                         index=tw_norm.index,
                                         columns=gv_slp.columns))
            tw_lk_wz = pd.concat([tw_lk_wz, tw_lk_wz_slp], axis=1)
        tageswerte = pd.concat([tageswerte, tw_lk_wz], axis=1)
    df = pd.concat([df, tageswerte.iloc[:days]], axis=1)
    df = df.iloc[days:]
    df.columns =\
        pd.MultiIndex.from_tuples([(int(x), int(y)) for x, y in
                                   df.columns.str.split('_')])
    return [df, gv_lk_return]


def disagg_daily_gas_slp_households(state, temperatur_df, how='top-down',
                                    **kwargs):
    """
    Returns daily demand of gas with a given yearly demand in MWh
    per district and SLP.

    state: str
        must be one of ['BW','BY','BE','BB','HB','HH','HE','MV',
                        'NI','NW','RP','SL','SN','ST','SH','TH']
    how : str, optional, throughput from function disagg_households_gas
        must be one of ['top-down', 'bottom-up', 'bottom-up_2']

    Returns
    -------
    pd.DataFrame
    """
    cfg = kwargs.get('cfg', get_config())
    year = kwargs.get('year', cfg['base_year'])
    if ((year % 4 == 0)
            & (year % 100 != 0)
            | (year % 4 == 0)
            & (year % 100 == 0)
            & (year % 400 == 0)):
        days = 366
    else:
        days = 365
    # gas consumption is split in heating, cooking, hotwater.
    # will be summed up since SLP do not distinguish these.
    gv_lk_master = disagg_households_gas(how=how, year=year).sum(axis=1)
    # get ratio if single and multi familiy houses per district
    ratio_EFH = percentage_EFH_MFH(MFH=False)
    ratio_MFH = percentage_EFH_MFH(MFH=True)
    # multiply ratio with df
    df_gas_EFH = (gv_lk_master.multiply(ratio_EFH))
    df_gas_MFH = (gv_lk_master.multiply(ratio_MFH))
    # create new DF and rename nuts_3 labeled districts with german ags and
    # assign new column with federal state
    gv_lk = pd.DataFrame({'EFH': df_gas_EFH,
                          'MFH': df_gas_MFH})
    gv_lk_return = gv_lk.copy()
    gv_lk = (gv_lk.rename(index=dict_region_code(keys='natcode_nuts3',
                                                 values='ags_lk'))
             .assign(BL=lambda x: [bl_dict().get(int(i[: -3]))
                                   for i in x.index.astype(str)]))
    # drop all districts which are not in given state
    gv_lk = gv_lk.loc[gv_lk['BL'] == state].drop(columns=['BL']).transpose()
    # create new DF for later use
    df = pd.DataFrame(index=range(days))
    # create list of district id
    list_ags = gv_lk.columns.astype(str)
    # get slp name for MFH and EFH Heating as well as Cooking
    gv_lk['SLP'] = [slp_household_gas()[x] for x in (gv_lk.index)]
    # get slp weekday parameters
    F_wd = (gas_slp_weekday_params(state, year=year)
            .drop(columns=['MO', 'DI', 'MI', 'DO', 'FR', 'SA', 'SO'])
            .set_index('Date'))
    tageswerte = pd.DataFrame(index=F_wd.index)
    logger.info('... creating state-specific load-profiles')
    for slp in gv_lk['SLP'].unique():
        F_wd_slp = F_wd[['FW_'+slp]]
        h_slp = h_value(slp, list_ags, temperatur_df)

        if (len(h_slp) != len(F_wd_slp)):
            raise KeyError('The chosen historical weather year and the chosen '
                           'projected year have mismatching lengths.'
                           'This could be due to gap years. Please change the '
                           'historical year in hist_weather_year() in '
                           'config.py to a year of matching length.')

        tw = pd.DataFrame(np.multiply(h_slp.values, F_wd_slp.values),
                          index=h_slp.index, columns=h_slp.columns)
        tw_norm = tw/tw.sum()
        gv_df = (gv_lk.loc[gv_lk['SLP'] == slp].drop(columns=['SLP'])
                      .stack().reset_index())
        tw_lk_wz = pd.DataFrame(index=h_slp.index)
        for lk in gv_df['nuts3'].unique():
            gv_slp = (gv_df.loc[gv_df['nuts3'] == lk]
                           .drop(columns=['nuts3'])
                           .set_index('level_0').transpose()
                           .rename(columns=lambda x: str(lk) + '_' + str(x)))
            tw_lk_wz_slp = (pd.DataFrame(np.multiply(tw_norm[
                                                     [str(lk)]
                                                     * len(gv_slp.columns)]
                                                     .values, gv_slp.values),
                                         index=tw_norm.index,
                                         columns=gv_slp.columns))
            tw_lk_wz = pd.concat([tw_lk_wz, tw_lk_wz_slp], axis=1)
        tageswerte = pd.concat([tageswerte, tw_lk_wz], axis=1)
    df = pd.concat([df, tageswerte.iloc[:days]], axis=1)
    df = df.iloc[days:]
    df.columns =\
        pd.MultiIndex.from_tuples([(int(x), str(y)) for x, y in
                                   df.columns.str.split('_')])
    return [df, gv_lk_return]


def disagg_temporal_gas_CTS(detailed=False, use_nuts3code=False, **kwargs):
    """
    Disagreggate spatial data of CTS' gas demand temporally.

    detailed : bool, default False
        If True return 'per district and branch' else only 'per district'
    use_nuts3code : bool, default False
        If True use NUTS-3 codes as region identifiers.

    Returns
    -------
    pd.DataFrame
    """
    cfg = kwargs.get('cfg', get_config())
    year = kwargs.get('year', cfg['base_year'])
    if ((year % 4 == 0) & (year % 100 != 0) | (year % 4 == 0)
            & (year % 100 == 0) & (year % 400 == 0)):
        hours = 8784
    else:
        hours = 8760
    temperatur_df = t_allo(year=year)
    df = pd.DataFrame(0, columns=temperatur_df.columns,
                      index=pd.date_range((str(year) + '-01-01'),
                                          periods=hours, freq='H'))
    for state in bl_dict().values():
        logger.info('Working on state: {}.'.format(state))
        tw_df, gv_lk = disagg_daily_gas_slp_cts(state, temperatur_df,
                                                year=year)
        gv_lk = (gv_lk.assign(BL=[bl_dict().get(int(x[:-3]))
                                  for x in gv_lk.index.astype(str)]))
        t_allo_df = temperatur_df[gv_lk.loc[gv_lk['BL'] == state]
                                       .index.astype(str)]
        for col in t_allo_df.columns:
            t_allo_df[col].values[t_allo_df[col].values < -15] = -15
            t_allo_df[col].values[(t_allo_df[col].values > -15)
                                  & (t_allo_df[col].values < -10)] = -10
            t_allo_df[col].values[(t_allo_df[col].values > -10)
                                  & (t_allo_df[col].values < -5)] = -5
            t_allo_df[col].values[(t_allo_df[col].values > -5)
                                  & (t_allo_df[col].values < 0)] = 0
            t_allo_df[col].values[(t_allo_df[col].values > 0)
                                  & (t_allo_df[col].values < 5)] = 5
            t_allo_df[col].values[(t_allo_df[col].values > 5)
                                  & (t_allo_df[col].values < 10)] = 10
            t_allo_df[col].values[(t_allo_df[col].values > 10)
                                  & (t_allo_df[col].values < 15)] = 15
            t_allo_df[col].values[(t_allo_df[col].values > 15)
                                  & (t_allo_df[col].values < 20)] = 20
            t_allo_df[col].values[(t_allo_df[col].values > 20)
                                  & (t_allo_df[col].values < 25)] = 25
            t_allo_df[col].values[(t_allo_df[col].values > 25)] = 100
            t_allo_df = t_allo_df.astype('int32')
        f_wd = ['FW_BA', 'FW_BD', 'FW_BH', 'FW_GA', 'FW_GB', 'FW_HA', 'FW_KO',
                'FW_MF', 'FW_MK', 'FW_PD', 'FW_WA', 'FW_SpaceHeating-MFH',
                'FW_SpaceHeating-EFH',
                'FW_Cooking_HotWater-HKO']
        calender_df = (gas_slp_weekday_params(state, year=year)
                       .drop(columns=f_wd))
        temp_calender_df = (pd.concat([calender_df.reset_index(),
                                       t_allo_df.reset_index()], axis=1))

        if temp_calender_df.isnull().values.any():
            raise KeyError('The chosen historical weather year and the chosen '
                           'projected year have mismatching lengths.'
                           'This could be due to gap years. Please change the '
                           'historical year in hist_weather_year() in '
                           'config.py to a year of matching length.')

        temp_calender_df['Tagestyp'] = 'MO'
        for typ in ['DI', 'MI', 'DO', 'FR', 'SA', 'SO']:
            (temp_calender_df.loc[temp_calender_df[typ], 'Tagestyp']) = typ
        list_lk = gv_lk.loc[gv_lk['BL'] == state].index.astype(str)
        for lk in list_lk:
            lk_df = pd.DataFrame(index=pd.date_range((str(year) + '-01-01'),
                                                     periods=hours, freq='H'))
            tw_df_lk = tw_df.loc[:, int(lk)]
            tw_df_lk.index = pd.DatetimeIndex(tw_df_lk.index)
            last_hour = tw_df_lk.copy()[-1:]
            last_hour.index = last_hour.index + timedelta(1)
            tw_df_lk = tw_df_lk.append(last_hour)
            tw_df_lk = tw_df_lk.resample('H').pad()
            tw_df_lk = tw_df_lk[:-1]

            temp_cal = temp_calender_df.copy()
            temp_cal = temp_cal[['Date', 'Tagestyp', lk]].set_index("Date")
            last_hour = temp_cal.copy()[-1:]
            last_hour.index = last_hour.index + timedelta(1)
            temp_cal = temp_cal.append(last_hour)
            temp_cal = temp_cal.resample('H').pad()
            temp_cal = temp_cal[:-1]
            temp_cal['Stunde'] = pd.DatetimeIndex(temp_cal.index).time
            temp_cal = temp_cal.set_index(["Tagestyp", lk, 'Stunde'])

            for slp in list(dict.fromkeys(slp_wz_g().values())):
                f = ('Lastprofil_{}.xls'.format(slp))
                slp_profil = pd.read_excel(data_in('temporal',
                                                   'Gas Load Profiles', f))
                slp_profil = pd.DataFrame(slp_profil.set_index(['Tagestyp',
                                            'Temperatur\nin °C\nkleiner']))
                slp_profil.columns = pd.to_datetime(slp_profil.columns,
                                                    format='%H:%M:%S')
                slp_profil.columns = pd.DatetimeIndex(slp_profil.columns).time
                slp_profil = slp_profil.stack()
                temp_cal['Prozent'] = [slp_profil[x] for x in temp_cal.index]
                for wz in [k for k, v in slp_wz_g().items()
                           if v.startswith(slp)]:
                    lk_df[str(lk) + '_' + str(wz)] = (tw_df_lk[wz].values
                                                      * temp_cal['Prozent']
                                                      .values/100)
                    df[str(lk) + '_' + str(wz)] = (tw_df_lk[wz].values
                                                   * temp_cal['Prozent']
                                                   .values/100)
            df[str(lk)] = lk_df.sum(axis=1)
    if detailed:
        df = df.drop(columns=gv_lk.index.astype(str))
        df.columns =\
            pd.MultiIndex.from_tuples([(int(x), int(y)) for x, y in
                                       df.columns.str.split('_')])
    else:
        df = df[gv_lk.index.astype(str)]

    if use_nuts3code:
        df = df.rename(columns=dict_region_code(level='lk', keys='ags_lk',
                                                values='natcode_nuts3'),
                       level=(0 if detailed else None))
    return df


def disagg_temporal_gas_households(use_nuts3code=False, how='top-down',
                                   **kwargs):
    """
    Disagreggate spatial data of households' gas demand temporally.

    use_nuts3code : bool, default False
        If True use NUTS-3 codes as region identifiers.
     how : str, optional, throughput from function disagg_households_gas
        must be one of ['top-down', 'bottom-up', 'bottom-up_2']
    Returns
    -------
    pd.DataFrame
    """
    cfg = kwargs.get('cfg', get_config())
    year = kwargs.get('year', cfg['base_year'])
    if ((year % 4 == 0) & (year % 100 != 0) | (year % 4 == 0)
            & (year % 100 == 0) & (year % 400 == 0)):
        hours = 8784
    else:
        hours = 8760
    temperatur_df = t_allo(year=year)
    df = pd.DataFrame(0, columns=temperatur_df.columns,
                      index=pd.date_range((str(year) + '-01-01'),
                                          periods=hours, freq='H'))
    for state in bl_dict().values():
        logger.info('Working on state: {}.'.format(state))
        tw_df, gv_lk = disagg_daily_gas_slp_households(state, temperatur_df,
                                                       year=year, how=how)
        gv_lk = (gv_lk.rename(index=dict_region_code(keys='natcode_nuts3',
                                                     values='ags_lk'))
                 .assign(BL=lambda x: [bl_dict().get(int(i[: -3]))
                                       for i in x.index.astype(str)]))
        t_allo_df = temperatur_df[gv_lk.loc[gv_lk['BL'] == state]
                                       .index.astype(str)]
        for col in t_allo_df.columns:
            t_allo_df[col].values[t_allo_df[col].values < -15] = -15
            t_allo_df[col].values[(t_allo_df[col].values > -15)
                                  & (t_allo_df[col].values < -10)] = -10
            t_allo_df[col].values[(t_allo_df[col].values > -10)
                                  & (t_allo_df[col].values < -5)] = -5
            t_allo_df[col].values[(t_allo_df[col].values > -5)
                                  & (t_allo_df[col].values < 0)] = 0
            t_allo_df[col].values[(t_allo_df[col].values > 0)
                                  & (t_allo_df[col].values < 5)] = 5
            t_allo_df[col].values[(t_allo_df[col].values > 5)
                                  & (t_allo_df[col].values < 10)] = 10
            t_allo_df[col].values[(t_allo_df[col].values > 10)
                                  & (t_allo_df[col].values < 15)] = 15
            t_allo_df[col].values[(t_allo_df[col].values > 15)
                                  & (t_allo_df[col].values < 20)] = 20
            t_allo_df[col].values[(t_allo_df[col].values > 20)
                                  & (t_allo_df[col].values < 25)] = 25
            t_allo_df[col].values[(t_allo_df[col].values > 25)] = 100
            t_allo_df = t_allo_df.astype('int32')
        # all of these columns will be dropped
        f_wd = ['FW_BA', 'FW_BD', 'FW_BH', 'FW_GA', 'FW_GB', 'FW_HA', 'FW_KO',
                'FW_MF', 'FW_MK', 'FW_PD', 'FW_WA', 'FW_SpaceHeating-MFH',
                'FW_SpaceHeating-EFH',
                'FW_Cooking_HotWater-HKO']
        # create calender
        calender_df = (gas_slp_weekday_params(state, year=year)
                       .drop(columns=f_wd))
        temp_calender_df = (pd.concat([calender_df.reset_index(),
                                       t_allo_df.reset_index()], axis=1))

        if temp_calender_df.isnull().values.any():
            raise KeyError('The chosen historical weather year and the chosen '
                           'projected year have mismatching lengths.'
                           'This could be due to gap years. Please change the '
                           'historical year in hist_weather_year() in '
                           'config.py to a year of matching length.')

        temp_calender_df['Tagestyp'] = 'MO'
        for typ in ['DI', 'MI', 'DO', 'FR', 'SA', 'SO']:
            (temp_calender_df.loc[temp_calender_df[typ], 'Tagestyp']) = typ
        list_lk = gv_lk.loc[gv_lk['BL'] == state].index.astype(str)
        for lk in list_lk:
            lk_df = pd.DataFrame(index=pd.date_range((str(year) + '-01-01'),
                                                     periods=hours, freq='H'))
            tw_df_lk = tw_df.loc[:, int(lk)]
            tw_df_lk.index = pd.DatetimeIndex(tw_df_lk.index)
            last_hour = tw_df_lk.copy()[-1:]
            last_hour.index = last_hour.index + timedelta(1)
            tw_df_lk = tw_df_lk.append(last_hour)
            tw_df_lk = tw_df_lk.resample('H').pad()
            tw_df_lk = tw_df_lk[:-1]

            temp_cal = temp_calender_df.copy()
            temp_cal = temp_cal[['Date', 'Tagestyp', lk]].set_index("Date")
            last_hour = temp_cal.copy()[-1:]
            last_hour.index = last_hour.index + timedelta(1)
            temp_cal = temp_cal.append(last_hour)
            temp_cal = temp_cal.resample('H').pad()
            temp_cal = temp_cal[:-1]
            temp_cal['Stunde'] = pd.DatetimeIndex(temp_cal.index).time
            temp_cal = temp_cal.set_index(["Tagestyp", lk, 'Stunde'])

            for slp in [slp_household_gas()[x] for x in tw_df_lk.columns.values]:
                f = ('Lastprofil_{}.xls'.format(slp))
                slp_profil = pd.read_excel(data_in('temporal',
                                                   'Gas Load Profiles', f))
                slp_profil = pd.DataFrame(slp_profil.set_index(['Tagestyp',
                                            'Temperatur\nin °C\nkleiner']))
                slp_profil.columns = pd.to_datetime(slp_profil.columns,
                                                    format='%H:%M:%S')
                slp_profil.columns = pd.DatetimeIndex(slp_profil.columns).time
                slp_profil = slp_profil.stack()
                temp_cal['Prozent'] = [slp_profil[x] for x in temp_cal.index]
                for wz in [k for k, v in slp_household_gas().items()
                           if v.startswith(slp)]:
                    lk_df[str(lk) + '_' + str(wz)] = (tw_df_lk[wz].values
                                                      * temp_cal['Prozent']
                                                      .values/100)
                    df[str(lk) + '_' + str(wz)] = (tw_df_lk[wz].values
                                                   * temp_cal['Prozent']
                                                   .values/100)
            df[str(lk)] = lk_df.sum(axis=1)
    df = df[gv_lk.index.astype(str)]
    df.columns = df.columns.astype(int)

    if use_nuts3code:
        df = df.rename(columns=dict_region_code(level='lk', keys='ags_lk',
                                                values='natcode_nuts3'))
    return df


def disagg_temporal_industry_blp(source='power', detailed=False,
                                 use_nuts3code=False, low=0.4,
                                 no_self_gen=False, **kwargs):
    """
    Disagreggate spatial data of industrie's power or gas demand temporally.

    Parameters
    ----------
    source : str
        Must be either 'power' or 'gas'
    detailed : bool, default False
        choose depth of dissolution
        True: demand per district and detailed
        False: demand per district
    use_nuts3code : bool, default False
        If True use NUTS-3 codes as region identifiers.
    low : float
        throughput for data.shift_load_profile_generator(low)
    no_self_gen : bool, default False
        throughput for spatial.disagg_CTS_industry(no_self_gen=False)
        If True: returns specific power and gas consumption without self
                 generation, resulting energy consumption will be lower
    Returns
    -------
    pd.DataFrame or Tuple
    """
    cfg = kwargs.get('cfg', get_config())
    year = kwargs.get('year', cfg['base_year'])
    if year not in range(2009, 2020):
        raise ValueError("to use blp, `year` must be between 2009 and 2019.")
    if source != 'power':
        raise ValueError("blp can only used for electricity consumption."
                         'change source to "power".')
    # Obtain yearly power consumption per WZ per LK
    ec_yearly = (disagg_CTS_industry('power', 'industry', year=year,
                                     no_self_gen=no_self_gen))
    # splitting yearly demand in two groups. slp group are disaggregated with
    # shift load profiles. blp group are disaggregated with blp
    # (=profiles based on metered load data from the demandregio project)
    blp_i = ([x for x in shift_profile_industry().keys()
              if x in blp_wz_list()])
    slp_i = ([x for x in shift_profile_industry().keys()
              if x not in blp_wz_list()])
    # energy consumption 'slp'
    ec_slp = (ec_yearly.loc[slp_i].transpose()
              .assign(BL=lambda x: [bl_dict().get(int(i[: -3]))
                                    for i in x.index.astype(str)]))
    # energy consumption 'blp'
    ec_blp = ec_yearly.loc[blp_i]

    total_sum = ec_yearly.sum().sum()  # for later check

    # start with temporal disaggregation of slp group
    # Create empty 15min-index'ed DataFrame for target year
    idx = pd.date_range(start=str(year), end=str(year+1), freq='15T')[:-1]
    DF = pd.DataFrame(index=idx)

    for state in bl_dict().values():
        logger.info('Working on state: {}.'.format(state))
        sv_lk_wz = (ec_slp
                    .loc[lambda x: x['BL'] == state]
                    .drop(columns=['BL'])
                    .fillna(value=0)
                    .transpose()
                    .assign(SP=lambda x:
                            [shift_profile_industry()[i] for i in x.index]))
        logger.info('... creating state-specific load-profiles')
        sp_bl = shift_load_profile_generator(state, low, year=year)
        # Plausibility check:
        assert sp_bl.index.equals(idx), "The time-indizes are not aligned"
        # Create 15min-index'ed DataFrames for current state
        if detailed:
            sv_lk_wz_ts = pd.DataFrame(index=idx)

        else:
            cols = sv_lk_wz.drop(columns=['SP']).columns
            sv_lk_ts = pd.DataFrame(index=idx, columns=cols).fillna(0.0)

        logger.info('... assigning load-profiles to WZs')
        for sp in sv_lk_wz['SP'].unique():
            sv_lk = (sv_lk_wz.loc[sv_lk_wz['SP'] == sp]
                             .drop(columns=['SP']).stack().reset_index())
            sv_dtl_df = sv_lk.groupby(by=['level_1'])[[0]].sum().transpose()
            sv_lk = (sv_lk.assign(LK_WZ=lambda x: x.level_1.astype(str) + '_'
                                  + x.level_0.astype(str))
                     .set_index('LK_WZ')
                     .drop(['level_0', 'level_1'], axis=1)
                     .loc[lambda x: x[0] >= 0]
                     .transpose())

            if detailed:  # Calculate load profile for each LK and WZ
                lp_lk_wz = (pd.DataFrame(np.multiply(sp_bl[[sp]].values,
                                                     sv_lk.values),
                                         index=sp_bl.index,
                                         columns=sv_lk.columns))
            else:  # Calculate load profile for each LK
                lp_lk = (pd.DataFrame(np.multiply(sp_bl[[sp]].values,
                                                  sv_dtl_df.values),
                                      index=sp_bl.index,
                                      columns=sv_dtl_df.columns))
            # Merge intermediate results
            if detailed:
                sv_lk_wz_ts = (sv_lk_wz_ts.merge(lp_lk_wz, left_index=True,
                                                 right_index=True,
                                                 suffixes=(False, False)))
            else:
                sv_lk_ts += lp_lk

        # Concatenate the state-wise results
        # restore MultiIndex as integer tuples
        if detailed:
            sv_lk_wz_ts.columns =\
                pd.MultiIndex.from_tuples([(int(x), int(y)) for x, y in
                                           sv_lk_wz_ts.columns.str.split('_')])
            DF = pd.concat([DF, sv_lk_wz_ts], axis=1)
            DF.columns = pd.MultiIndex.from_tuples(DF.columns,
                                                   names=['LK', 'WZ'])
        else:
            DF = pd.concat([DF, sv_lk_ts], axis=1).dropna()

    # start with temporal disaggregation of blp group
    if detailed:
        DF2 = pd.DataFrame()
        logger.info('Start downloading BLP from Database now. This may take a'
                    ' while depending on the connection. ca. 250MB per industr'
                    'y will be downloaded and stored in your local folder ../d'
                    'ata_in/cache.')
        for branch in blp_i:
            logger.info('Working on WZ'+str(branch))
            branch_model = regional_branch_load_profiles(wz=branch, year=year)
            demand_branch = (np.multiply(branch_model,
                                         ec_blp.loc[ec_blp.index == branch]))
            DF2 = pd.concat([DF2, (demand_branch.T.assign(WZ=branch)
                                   .set_index('WZ', append=True).T)], axis=1)

        DF2 = DF2/4000  # convert from GW to MWh
        total_demand = DF.join(DF2)

    else:
        DF2 = (pd.DataFrame(0, index=pd.date_range(freq='15Min',
                                                   start='01/01/'+str(year),
                                                   periods=len(DF)),
                            columns=ec_blp.columns))
        logger.info('Start downloading BLP from Database now. This may take a '
                    'while depending on the connection. ca. 250MB per industry'
                    ' will be downloaded and stored in your local folder ../da'
                    'ta_in/cache.')
        for branch in blp_i:
            logger.info('Working on WZ'+str(branch))
            branch_model = regional_branch_load_profiles(wz=branch, year=year)
            demand_branch = (np.multiply(branch_model,
                                         ec_blp.loc[ec_blp.index == branch]))
            DF2 = DF2.add(demand_branch, axis=1, fill_value=0)

        DF2 = DF2/4000  # convert from GW to MWh
        total_demand = DF.add(DF2, axis=1, fill_value=0)

    # Plausibility check:
    msg = ('The sum of yearly consumptions (={:.3f}) and the sum of disaggrega'
           'ted consumptions (={:.3f}) do not match! Please check algorithm!')
    disagg_sum = total_demand.sum().sum()
    assert np.isclose(total_sum, disagg_sum,
                      rtol=1.e-3), msg.format(total_sum, disagg_sum)

    if use_nuts3code:
        total_demand = total_demand.rename(columns=dict_region_code(
            level='lk', keys='ags_lk', values='natcode_nuts3'),
            level=(0 if detailed else None))
    return total_demand


def disagg_temporal_industry(source, detailed=False, use_nuts3code=False,
                             low=0.4, no_self_gen=False, **kwargs):
    """
    Disagreggate spatial data of industrie's power or gas demand temporally.

    Parameters
    ----------
    source : str
        Must be either 'power' or 'gas'
    detailed : bool, default False
        choose depth of dissolution
        True: demand per district and detailed
        False: demand per district
    use_nuts3code : bool, default False
        If True use NUTS-3 codes as region identifiers.
    low : float
        throughput for data.shift_load_profile_generator(low)
    no_self_gen : bool, default False
        throughput for spatial.disagg_CTS_industry(no_self_gen=False)
        If True: returns specific power and gas consumption without self
                 generation, resulting energy consumption will be lower
    Returns
    -------
    pd.DataFrame or Tuple
    """
    cfg = kwargs.get('cfg', get_config())
    year = kwargs.get('year', cfg['base_year'])
    # Obtain yearly power consumption per WZ per LK
    ec_yearly = (disagg_CTS_industry(source, 'industry',
                                     no_self_gen=no_self_gen, year=year)
                 .transpose()
                 .assign(BL=lambda x: [bl_dict().get(int(i[: -3]))
                                       for i in x.index.astype(str)]))
    total_sum = ec_yearly.drop('BL', axis=1).sum().sum()  # for later check
    # Create empty 15min-index'ed DataFrame for target year
    idx = pd.date_range(start=str(year), end=str(year+1), freq='15T')[:-1]
    DF = pd.DataFrame(index=idx)

    for state in bl_dict().values():
        logger.info('Working on state: {}.'.format(state))
        sv_lk_wz = (ec_yearly
                    .loc[lambda x: x['BL'] == state]
                    .drop(columns=['BL'])
                    .fillna(value=0)
                    .transpose()
                    .assign(SP=lambda x:
                            [shift_profile_industry()[i] for i in x.index]))
        logger.info('... creating state-specific load-profiles')
        sp_bl = shift_load_profile_generator(state, low, year=year)
        # Plausibility check:
        assert sp_bl.index.equals(idx), "The time-indizes are not aligned"
        # Create 15min-index'ed DataFrames for current state
        if detailed:
            sv_lk_wz_ts = pd.DataFrame(index=idx)
        else:
            cols = sv_lk_wz.drop(columns=['SP']).columns
            sv_lk_ts = pd.DataFrame(index=idx, columns=cols).fillna(0.0)

        logger.info('... assigning load-profiles to WZs')
        for sp in sv_lk_wz['SP'].unique():
            sv_lk = (sv_lk_wz.loc[sv_lk_wz['SP'] == sp]
                             .drop(columns=['SP']).stack().reset_index())
            sv_dtl_df = sv_lk.groupby(by=['level_1'])[[0]].sum().transpose()
            sv_lk = (sv_lk.assign(LK_WZ=lambda x: x.level_1.astype(str) + '_'
                                  + x.level_0.astype(str))
                     .set_index('LK_WZ')
                     .drop(['level_0', 'level_1'], axis=1)
                     .loc[lambda x: x[0] >= 0]
                     .transpose())

            if detailed:  # Calculate load profile for each LK and WZ
                lp_lk_wz = (pd.DataFrame(np.multiply(sp_bl[[sp]].values,
                                                     sv_lk.values),
                                         index=sp_bl.index,
                                         columns=sv_lk.columns))
            else:  # Calculate load profile for each LK
                lp_lk = (pd.DataFrame(np.multiply(sp_bl[[sp]].values,
                                                  sv_dtl_df.values),
                                      index=sp_bl.index,
                                      columns=sv_dtl_df.columns))
            # Merge intermediate results
            if detailed:
                sv_lk_wz_ts = (sv_lk_wz_ts.merge(lp_lk_wz, left_index=True,
                                                 right_index=True,
                                                 suffixes=(False, False)))
            else:
                sv_lk_ts += lp_lk

        # Concatenate the state-wise results
        if detailed:  # restore MultiIndex as integer tuples
            sv_lk_wz_ts.columns =\
                pd.MultiIndex.from_tuples([(int(x), int(y)) for x, y in
                                           sv_lk_wz_ts.columns.str.split('_')])
            DF = pd.concat([DF, sv_lk_wz_ts], axis=1)
            DF.columns = pd.MultiIndex.from_tuples(DF.columns,
                                                   names=['LK', 'WZ'])
        else:
            DF = pd.concat([DF, sv_lk_ts], axis=1).dropna()

    # Plausibility check:
    msg = ('The sum of yearly consumptions (={:.3f}) and the sum of disaggrega'
           'ted consumptions (={:.3f}) do not match! Please check algorithm!')
    disagg_sum = DF.sum().sum()
    assert np.isclose(total_sum, disagg_sum), msg.format(total_sum, disagg_sum)

    if use_nuts3code:
        DF = DF.rename(columns=dict_region_code(level='lk', keys='ags_lk',
                                                values='natcode_nuts3'),
                       level=(0 if detailed else None))
    return DF
