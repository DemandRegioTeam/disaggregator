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
Provides functions to import all relevant data.
"""

import pandas as pd
import logging
from .config import (get_config, _data_in, database_raw, region_id_to_nuts3,
                     literal_converter)
logger = logging.getLogger(__name__)
cfg = get_config()


# --- Dimensionless data ------------------------------------------------------


def elc_consumption_HH(by_HH_size=False, **kwargs):
    """
    Read and return electricity consumption for households (HH) in [MWh/a].

    Parameters
    ----------
    by_HH_size : bool, optional
        If True, return scalar value accumulated per NUTS-0 area.
        If False, return specific values per household size.

    Returns
    -------
    float or pd.DataFrame
    """
    key = 'elc_cons_HH_by_size' if by_HH_size else 'elc_cons_HH_total'
    year = kwargs.get('year', cfg['base_year'])
    force_update = kwargs.get('force_update', False)
    source = kwargs.get('source', cfg[key]['source'])
    if source not in ['local', 'database']:
        raise KeyError("Wrong source key given in config.yaml - must be either"
                       "'local' or 'database' but is: {}".format(source))
    if by_HH_size:
        if source == 'local':
            df = read_local(_data_in('dimensionless', cfg[key]['filename']))
        elif source == 'database':
            df = database_get('spatial', table_id=cfg[key]['table_id'],
                              force_update=force_update)

        df = (df.assign(internal_id=lambda x: x.internal_id.astype(str))
                .assign(hh_size=lambda x: x.internal_id.str[1].astype(int))
                .loc[lambda x: x.hh_size != 0]
                .set_index('hh_size')
                .sort_index(axis=0)
                .loc[:, 'value'])
        # Quick & dirty: HH with >5 people = HH with 5 people
        df.loc[6] = df[5]
        # Transform kWh to MWh
        df /= 1e3
    else:
        if source == 'local':
            df = float(read_local(_data_in('dimensionless',
                                           cfg[key]['filename']),
                                  year=year)
                       .reset_index(drop=True)
                       .loc[0, 'value'])
        elif source == 'database':
            df = float(database_get('spatial', table_id=cfg[key]['table_id'],
                                    year=year, force_update=force_update)
                       .loc[0, 'value'])
    return df


def heat_consumption_HH(by='households', **kwargs):
    """
    Read and return heat consumption for households (HH) in [MWh/a] as
    specific values per household size.

    Parameters
    ----------
    by : str
        Either 'households' or 'buildings'

    Returns
    -------
    pd.DataFrame
    """
    source = kwargs.get('source', cfg['heat_consumption_HH']['source'])
    force_update = kwargs.get('force_update', False)

    if source == 'local':
        col = {'households': 'PersonsInHousehold',
               'buildings': 'BuildingType'}[by]
        df = (pd.read_csv(_data_in('dimensionless',
                                   cfg['heat_consumption_HH']['filename']))
                .pivot_table(index='Application', columns=col, values='Value'))
    elif source == 'database':
        raise NotImplementedError('Not here yet!')
    else:
        raise KeyError('Wrong source key given in config.yaml - must be either'
                       ' ``local'' or ``database'' but is: {}'.format(source))
    return df


def gas_consumption_HH(**kwargs):
    """
    Read and return gas consumption for households (HH) in [MWh/a] categorized
    by application

    Returns
    -------
    pd.DataFrame
    """
    id_to_application = {1: 'SpaceHeating',
                         2: 'HotWater',
                         3: 'Cooking'}
    key = 'gas_consumption_HH'
    year = kwargs.get('year', cfg['base_year'])
    source = kwargs.get('source', cfg[key]['source'])
    table_id = kwargs.get('table_id', cfg[key]['table_id'])
    force_update = kwargs.get('force_update', False)

    if source == 'local':
        df = read_local(_data_in('dimensionless', cfg[key]['filename']),
                        year=year)
    elif source == 'database':
        df = database_get('spatial', table_id=table_id, year=year,
                          force_update=force_update)
    else:
        raise KeyError("Wrong source key given in config.yaml - must be either"
                       "'local' or 'database' but is: {}".format(source))
    return (df.assign(application=lambda x:
                      x.internal_id.astype(str).str[1].astype(int))
              .replace(dict(application=id_to_application))
              .set_index('application'))['value']


def zve_percentages_applications():
    df = (pd.read_csv(_data_in('temporal', 'percentages_applications.csv'),
                      index_col='Application', engine='c')
            .drop(labels='all', axis=1))
    df.columns = df.columns.astype(int)
    return df


def zve_percentages_baseload():
    df = (pd.read_csv(_data_in('temporal', 'percentages_baseload.csv'),
                      index_col='Application', engine='c')
            .drop(labels='all', axis=1))
    df.columns = df.columns.astype(int)
    return df


def zve_application_profiles():
    return pd.read_csv(_data_in('temporal', 'application_profiles.csv'),
                       engine='c')


# --- Spatial data ------------------------------------------------------------


def population(**kwargs):
    """
    Read, transform and return the number of residents per NUTS-3 area.

    Returns
    -------
    pd.Series
        index: NUTS-3 codes
    """
    year = kwargs.get('year', cfg['base_year'])
    source = kwargs.get('source', cfg['population']['source'])
    table_id = kwargs.get('table_id', cfg['population']['table_id'])
    internal_id = kwargs.get('internal_id', None)
    force_update = kwargs.get('force_update', False)

    if source == 'local':
        if year >= 2018:
            logger.warn('open # TODO not yet working correctly!')
            fn = _data_in('regional', cfg['demographic_trend']['filename'])
            df = read_local(fn, year=year)
        else:
            fn = _data_in('regional', cfg['population']['filename'])
            df = read_local(fn, year=year)
    elif source == 'database':
        if year >= 2018:
            # In this case take demographic trend data
            df_spatial = database_description('spatial', short=False)
            table_id = kwargs.get('table_id',
                                  cfg['demographic_trend']['table_id'])
            internal_id = kwargs.get('internal_id',
                                     cfg['demographic_trend']['internal_id'])
            # if requested year is not available, take the closest one instead
            years = df_spatial.loc[table_id, 'years']
            year_closest = min(years, key=lambda x: abs(x-year))
            if year_closest != year:
                w = ('The requested year {} is not available. The closest '
                     'year {} is taken instead.')
                logger.warn(w.format(year, year_closest))
                year = year_closest
        df = database_get('spatial', table_id=table_id, year=year,
                          internal_id=internal_id, force_update=force_update)
    else:
        raise KeyError('Wrong source key given in config.yaml - must be either'
                       ' `local` or `database` but is: {}'.format(source))

    df = (df.assign(nuts3=lambda x: x.id_region.map(region_id_to_nuts3()))
            .set_index('nuts3').sort_index(axis=0))['value']
    plausibility_check_nuts3(df)
    return df


def elc_consumption_HH_spatial(**kwargs):
    """
    Read, transform and return a pd.Series with pre-calculated
    electricity consumption of households per NUTS-3 area.

    Returns
    -------
    pd.DataFrame
        index: NUTS-3 codes
    """
    year = kwargs.get('year', cfg['base_year'])
    source = kwargs.get('source', cfg['elc_cons_HH_spatial']['source'])
    table_id = kwargs.get('table_id', cfg['elc_cons_HH_spatial']['table_id'])
    force_update = kwargs.get('force_update', False)

    if source == 'local':
        fn = _data_in('regional', cfg['elc_cons_HH_spatial']['filename'])
        df = read_local(fn, year=year)
    elif source == 'database':
        df = database_get('spatial', table_id=table_id, year=year,
                          force_update=force_update)
    else:
        raise KeyError('Wrong source key given in config.yaml - must be either'
                       ' `local` or `database` but is: {}'.format(source))

    df = (df.assign(nuts3=lambda x: x.id_region.map(region_id_to_nuts3()))
            .set_index('nuts3').sort_index(axis=0))['value']
    plausibility_check_nuts3(df)
    return df


def households_per_size(original=False, **kwargs):
    """
    Read, transform and return the numbers of households for each household
    size per NUTS-3 area.

    Parameters
    ----------
    orignal : bool, optional
        A flag if the results should be left untouched and returned in
        original form for the year 2011 (True) or if they should be scaled to
        the given `year` by the population in that year (False).

    Returns
    -------
    pd.DataFrame
        index: NUTS-3 codes
    """
    year = kwargs.get('year', cfg['base_year'])
    source = kwargs.get('source', cfg['household_sizes']['source'])
    table_id = kwargs.get('table_id', cfg['household_sizes']['table_id'])
    force_update = kwargs.get('force_update', False)

    if source == 'local':
        fn = _data_in('regional', cfg['household_sizes']['filename'])
        df = read_local(fn)
    elif source == 'database':
        df = database_get('spatial', table_id=table_id, year=2011,
                          force_update=force_update)
    else:
        raise KeyError('Wrong source key given in config.yaml - must be either'
                       ' `local` or `database` but is: {}'.format(source))

    df = (df.assign(internal_id=lambda x: x.internal_id.astype(str))
            .assign(nuts3=lambda x: x.id_region.map(region_id_to_nuts3()),
                    hh_size=lambda x: x.internal_id.str[1].astype(int))
            .loc[lambda x: x.hh_size != 0]
            .pivot_table(values='value', index='nuts3', columns='hh_size',
                         aggfunc='sum'))

    if original:
        logger.warning('Orginal data is only available for the year 2011, so '
                       'passing `original=True` argument disables any scaling '
                       'and rounding to the given `year` based on the '
                       'newer population data (which is enabled by default).')
    else:
        # Create the percentages of persons living in each household size
        df_ratio = (df * df.columns)
        df_ratio = df_ratio.divide(df_ratio.sum(axis=1), axis='index')
        # Multiply with population, divide back into household counts and round
        df = df_ratio.multiply(population(year=year), axis='index')
        df = (df / df.columns).astype(int)

    plausibility_check_nuts3(df)
    return df


def living_space(aggregate=True, **kwargs):
    """
    Read, transform and return a DataFrame with the available living space
    in [m²] for each building type per NUTS-3 area.

    Parameters
    ----------
    aggregate : bool, optional
        A flag, if the results should only be grouped by the building
        type in columns (default is True)

    Returns
    -------
    pd.DataFrame
        index: NUTS-3 codes
    """
    building_nr_to_type = {1: '1FH',        # 1-family-house
                           2: '2FH',        # 2-family-house
                           3: 'MFH_03_06',  # multi-family-house (3-6)
                           4: 'MFH_07_12',  # multi-family-house (7-12)
                           5: 'MFH_13_99'}  # multi-family-house >13 families
    heating_system_to_name = {9: 'Steinkohle',
                              10: 'Braunkohle',
                              11: 'Erdgas',
                              34: 'Heizöl',
                              35: 'Biomasse (fest)',
                              68: 'Umgebungswärme',
                              69: 'Solarwärme',
                              71: 'Fernwärme',
                              72: 'Elektrische Energie',
                              218: 'Biomasse (außer Holz, Biogas)'}
    vc_to_yearstr = {1: 'A_<1900',
                     2: 'B_1900-1945',
                     3: 'C_1946-1960',
                     4: 'D_1961-1970',
                     5: 'E_1971-1980',
                     6: 'F_1981-1985',
                     7: 'G_1986-1995',
                     8: 'H_1996-2000',
                     9: 'I_2001-2005',
                     10: 'J_2006-2011',
                     2012: 'K_2012',
                     2013: 'L_2013',
                     2014: 'M_2014',
                     2015: 'N_2015',
                     2016: 'O_2016',
                     2017: 'P_2017',
                     2018: 'Q_2018'}

    year = kwargs.get('year', None)
    source = kwargs.get('source', cfg['living_space']['source'])
    table_id = kwargs.get('table_id', cfg['living_space']['table_id'])
    force_update = kwargs.get('force_update', False)
    bt = kwargs.get('internal_id_0', cfg['living_space']['internal_id'][0])
    vc = kwargs.get('internal_id_1', cfg['living_space']['internal_id'][1])
    hs = kwargs.get('internal_id_2', cfg['living_space']['internal_id'][2])
    ne = kwargs.get('internal_id_3', cfg['living_space']['internal_id'][3])
    if year is not None:
        logger.warning('No data for each year available, so passing the '
                       '`year` argument does not have any effect yet.')
    if source == 'local':
        df = read_local(_data_in('regional', cfg['living_space']['filename']))
    elif source == 'database':
        df = database_get('spatial', table_id=table_id,
                          force_update=force_update)
    else:
        raise KeyError('Wrong source key given in config.yaml - must be either'
                       ' `local` or `database` but is: {}'.format(source))

    df = (df.assign(nuts3=lambda x: x.id_region.map(region_id_to_nuts3()),
                    building_type=lambda x: x.internal_id.str[0],
                    vintage_class=lambda x: x.internal_id.str[1],
                    heating_system=lambda x: x.internal_id.str[2],
                    non_empty_building=lambda x:
                        x.internal_id.str[3].apply(bool))
          .drop(columns=['year', 'internal_id']))
    # Filter by possibly given internal_id
    if bt is not None and 1 <= bt <= 5:
        df = df.loc[lambda x: x.building_type == bt]
    if vc is not None and (1 <= vc <= 11 or vc >= 2000):
        df = df.loc[lambda x: x.vintage_class == vc]
    if hs is not None and (0 <= hs):
        df = df.loc[lambda x: x.heating_system == hs]
    if ne is not None and (0 <= ne <= 1):
        df = df.loc[lambda x: x.non_empty_building == ne]
    # Replace internal_ids by human readables of
    # - building_type
    # - heating_system
    # - vintage class
    df = (df.replace(dict(building_type=building_nr_to_type))
            .replace(dict(heating_system=heating_system_to_name))
            .replace(dict(vintage_class=vc_to_yearstr)))
    if aggregate:
        df = df.pivot_table(values='value', index='nuts3',
                            columns='building_type', aggfunc='sum')
        plausibility_check_nuts3(df)
    return df


def income(**kwargs):
    """
    Read, transform and return incomes in [Euro/cap] per NUTS-3 area.

    Returns
    -------
    pd.Series
        index: NUTS-3 codes
    """
    year = kwargs.get('year', cfg['base_year'])
    source = kwargs.get('source', cfg['income']['source'])
    table_id = kwargs.get('table_id', cfg['income']['table_id'])
    internal_id = kwargs.get('internal_id', cfg['income']['internal_id'])
    force_update = kwargs.get('force_update', False)

    if source == 'local':
        df = read_local(_data_in('regional', cfg['income']['filename']),
                        internal_id=internal_id, year=year)
    elif source == 'database':
        df = database_get('spatial', table_id=table_id, year=year,
                          internal_id=internal_id, force_update=force_update)
    else:
        raise KeyError('Wrong source key given in config.yaml - must be either'
                       ' `local` or `database` but is: {}'.format(source))

    df = (df.assign(nuts3=lambda x: x.id_region.map(region_id_to_nuts3()))
            .set_index('nuts3', drop=True)
            .sort_index())['value']
    plausibility_check_nuts3(df)
    return df


def stove_assumptions(**kwargs):
    """
    Return assumptions of shares and efficencies of gas and electric stoves
    for each nuts1 area.

    Returns
    -------
    pd.DataFrame
    """
    source = kwargs.get('source', cfg['stove_assumptions']['source'])
    if source == 'local':
        df = (pd.read_csv(_data_in('regional',
                                   cfg['stove_assumptions']['filename']),
                          index_col='natcode_nuts1', encoding='utf-8')
                .drop(labels='name_nuts1', axis=1))
    elif source == 'database':
        raise NotImplementedError('Not here yet!')
    else:
        raise KeyError('Wrong source key given in config.yaml!')
    return df


def hotwater_shares(**kwargs):
    """
    Return assumptions of shares and efficencies of gas and electric stoves
    for each nuts1 area.

    Returns
    -------
    pd.DataFrame
    """
    source = kwargs.get('source', cfg['hotwater_shares']['source'])
    if source == 'local':
        df = (pd.read_csv(_data_in('regional',
                                   cfg['hotwater_shares']['filename']),
                          index_col='natcode_nuts1', encoding='utf-8')
                .drop(labels='name_nuts1', axis=1))
    elif source == 'database':
        raise NotImplementedError('Not here yet!')
    else:
        raise KeyError('Wrong source key given in config.yaml!')
    return df


# --- Temporal data -----------------------------------------------------------


def elc_consumption_HH_temporal(**kwargs):
    """
    Return the electricity consumption of households in [MW] in hourly
    resolution based on the standard load profile H0.

    Returns
    -------
    pd.Series
        index: pd.DatetimeIndex
    """
    return (reshape_temporal(freq='1H', key='elc_cons_HH_temporal', **kwargs)
            * elc_consumption_HH(year=kwargs.get('year', cfg['base_year'])))


def reshape_temporal(freq=None, key=None, **kwargs):
    """
    Query temporal data w/o spatial resolution and return as DatetimeIndex'ed
    pd.Series.

    Returns
    -------
    pd.Series
        index: pd.DatetimeIndex
    """
    year = kwargs.get('year', cfg['base_year'])
    source = kwargs.get('source', cfg[key]['source'])
    table_id = kwargs.get('table_id', cfg[key]['table_id'])
    force_update = kwargs.get('force_update', False)
    if freq is None:
        if key is None:
            raise ValueError('You must pass either `freq` or `key`!')
        else:
            freq = cfg[key]['freq']

    if source == 'local':
        raise NotImplementedError('Not here yet!')
#        fn = _data_in('temporal', cfg[key]['filename'])
#        df = read_local(fn, year=year)
    elif source == 'database':
        values = literal_converter(
                database_get('temporal', table_id=table_id, year=year,
                             force_update=force_update).loc[0, 'values'])
        idx = pd.date_range(start=str(year), periods=len(values), freq=freq)
        df_exp = pd.Series(values, index=idx).astype(float)
    else:
        raise KeyError('Wrong source key given in config.yaml - must be either'
                       ' `local` or `database` but is: {}'.format(source))
    return df_exp


# --- Spatiotemporal data -----------------------------------------------------


def standard_load_profile(which='H0', freq='1H', **kwargs):
    """
    Return the standard load profile H0 in normalized units ('normalized' means
    here that the sum over all time steps equals one).
    """
    if which == 'H0':
        if freq.lower() == '1h':
            return reshape_spatiotemporal(key='slp_H0_1H', **kwargs)
        elif freq.lower() == '15min':
            return reshape_spatiotemporal(key='slp_H0_15min', **kwargs)
        else:
            raise NotImplementedError('`freq` must be either `1H` or `15min`.')
    else:
        raise NotImplementedError('Not here yet!')


def ambient_T(**kwargs):
    """
    Return the ambient temperature in [°C] per NUTS-3-region and time step.
    """
    return reshape_spatiotemporal(key='ambient_T', **kwargs)


def solar_irradiation(**kwargs):
    """
    Return the solar irradiation in [Wh/m²] per NUTS-3-region and time step.
    """
    return reshape_spatiotemporal(key='solar_irradiation', **kwargs)


def elc_consumption_HH_spatiotemporal(**kwargs):
    """
    Return the electricity consumption of private households in [MW]
    per NUTS-3-region and time step.
    """
    return reshape_spatiotemporal(key='elc_cons_HH_spatiotemporal', **kwargs)


def reshape_spatiotemporal(freq=None, key=None, **kwargs):
    """
    Query spatiotemporal data and shape into a 2-dimensional pd.DataFrame.

    Returns
    -------
    pd.DataFrame
        index:   NUTS-3 codes
        columns: time step
    """
    year = kwargs.get('year', cfg['base_year'])
    source = kwargs.get('source', cfg[key]['source'])
    table_id = kwargs.get('table_id', cfg[key]['table_id'])
    internal_id = kwargs.get('internal_id', cfg.get(key).get('internal_id'))
    force_update = kwargs.get('force_update', False)
    if freq is None:
        if key is None:
            raise ValueError('You must pass either `freq` or `key`!')
        else:
            freq = cfg[key]['freq']

    if source == 'local':
        raise NotImplementedError('Not here yet!')
    elif source == 'database':
        df = (database_get('temporal', table_id=table_id, year=year,
                           internal_id=internal_id, force_update=force_update)
              .assign(nuts3=lambda x: x.id_region.map(region_id_to_nuts3()))
              .set_index('nuts3').sort_index(axis=0)
              .loc[:, 'values']
              .apply(literal_converter))

        df_exp = (pd.DataFrame(df.values.tolist(), index=df.index)
                    .astype(float))
    else:
        raise KeyError('Wrong source key given in config.yaml - must be either'
                       ' `local` or `database` but is: {}'.format(source))
    return df_exp


# --- Utility functions -------------------------------------------------------


def database_description(dimension='spatial', short=True, only_active=True,
                         force_update=False):
    """
    Get a description table from the demandregio database.

    Parameters
    ----------
    dimension : str
        Either 'spatial' or 'temporal'.
    short : bool, default True
        If True, show a shortend version of the description table.
    only_active : bool, default True
        If True, show only active datasets in the description table.
    force_update : bool, default False
        If True, perform a fresh database query and overwrite data in cache.

    Returns
    -------
    pd.DataFrame
    """
    if dimension in ['spatial', 'temporal']:
        id_name = 'id_' + dimension
    else:
        raise ValueError("``which'' must be either 'spatial' or 'temporal' but"
                         "given was: {}".format(dimension))

    query = 'demandregio_{}_description'.format(dimension)
    df = database_raw(query, force_update=force_update).drop('sql', axis=1)
    if only_active:
        df = df.loc[lambda x: x.aktiv == 1]
    if short:
        cols = [id_name, 'title', 'description_short', 'region_types',
                'internal_id_description', 'units']
        if dimension == 'spatial':
            cols += ['years']
        else:
            cols += ['time_step', 'years_step', 'years_weather']
        df = df.reindex(cols, axis=1)
    return df.set_index(id_name).sort_index()


def database_get(dimension, table_id, internal_id=None, year=None,
                 allow_zero_negative=None, force_update=False):
    """
    Get data from the demandregio database.

    Parameters
    ----------
    dimension : str
        Either 'spatial' or 'temporal'.
    table_id : int or str, optional
        ID of table to be queried.
    internal_id : int or str, optional
        internal ID belonging to table_id to be queried (default: NoneType)
    year : int or str, optional
        Either the data year (spatial) or weather year (temporal)
    allow_zero_negative : bool, default NoneType
        Filter all spatial data (only!) that is zero or negative in query.
    force_update : bool, default False
        If True, perform a fresh database query and overwrite data in cache.

    Returns
    -------
    pd.DataFrame
    """
    if dimension in ['spatial', 'temporal']:
        id_name = 'id_' + dimension
        if dimension == 'spatial':
            if cfg['use_nuts_2016'] and table_id in cfg['nuts3_tables']:
                table = 'v_demandregio_spatial_lk401'
            else:
                table = 'demandregio_spatial'
        else:
            table = 'demandregio_temporal'
    else:
        raise ValueError("``which'' must be either 'spatial' or 'temporal' but"
                         "given was: {}".format(dimension))

    if not isinstance(allow_zero_negative, bool):
        allow_zero_negative = True if dimension == 'temporal' else False
    # Go through each variable and append to query str if needed.
    query = table + '?' + id_name + '=eq.' + str(table_id)
    if year is not None:
        year_var = 'year' if dimension == 'spatial' else 'year_weather'
        query += '&&' + year_var + '=eq.' + str(year)
    if internal_id is not None:
        if isinstance(internal_id, list):
            internal_id = ','.join([str(s) for s in internal_id])
        query += '&&' + 'internal_id' + '=eq.{' + str(internal_id) + '}'
    if allow_zero_negative is False:
        if dimension == 'spatial':
            query += '&&value=gt.0.0'
        else:
            logger.warning("Attention: `allow_zero_negative` is set to "
                           "False. In some timeseries (e.g. temperatures) "
                           "negative values make sense. Therefore they "
                           "are not truncated here. If really needed, "
                           "please do it later.")
    if table_id in cfg['load_sectionwise'][dimension]:
        logger.info('The requested dataframe is so huge, that it is '
                    'necessary to split the queries into the 16 federal '
                    'states. This process may take a while...')
        df = pd.DataFrame()
        for bl in range(1, 17):
            ll = int(bl * 1e6)      # lower limit
            ul = int((bl+1) * 1e6)  # upper limit
            q_reg = '&&id_region=gte.{}&&id_region=lt.{}'.format(ll, ul)
            df_bl = database_raw(query + q_reg, force_update=force_update)
            df = pd.concat([df, df_bl], axis=0, join='outer')
        return df
    else:
        return database_raw(query, force_update=force_update)


def database_shapes():
    """
    Perform a query through RESTful-API for the NUTS-3 shapefiles.

    Returns
    -------
    geopandas.DataFrame
    """
    import shapely.wkt as wkt  # needed for converting strings to MultiPolygons
    import geopandas as gpd

    df = database_raw('v_vg250_krs_simple?toleranz=eq.1000&&stand=eq.%2720'
                      '16-12-31%27&&select=id_ags,gen,geom_as_text,fl_km2')
    geom = [wkt.loads(mp_str) for mp_str in df.geom_as_text]
    return (gpd.GeoDataFrame(df.drop('geom_as_text', axis=1),
                             crs={'init': 'epsg:3857'}, geometry=geom)
               .assign(nuts3=lambda x: x.id_ags.map(region_id_to_nuts3()))
               .set_index('nuts3').sort_index(axis=0))


def plausibility_check_nuts3(df):
    """
    Check a given pd.DataFrame
    - if all nuts3 regions are available and
    - if all contained values are greater zero.

    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        Holding the values (required index: NUTS-3 codes)
    """
    A_db = set(region_id_to_nuts3().values())
    B_check = set(df.index)
    C_diff = B_check - A_db
    if len(C_diff) > 0:
        logger.warn('The nuts3-codes of the checked DataFrame are not '
                    'congruent with those in the database. These here are not '
                    'in the database: {}'.format(C_diff))
    if isinstance(df, pd.Series):
        if df.loc[lambda x: x <= 0.0].any():
            logger.warn('There are values less or equal to zero.')
    elif isinstance(df, pd.DataFrame):
        if df[df <= 0.0].any().any():
            logger.warn('There are values less or equal to zero.')
    else:
        raise NotImplementedError('Check for given type! Other than pd.Series '
                                  'or pd.DataFrame are not yet possible.')


def read_local(file, internal_id=None, year=None):
    df = pd.read_csv(file, index_col='idx', encoding='utf-8', engine='c',
                     converters={'internal_id': literal_converter,
                                 'region_types': literal_converter,
                                 'values': literal_converter,
                                 'years': literal_converter,
                                 'years_step': literal_converter,
                                 'years_weather': literal_converter})
    if year is not None:
        df = df.loc[lambda x: x.year == year]
    if internal_id is not None:
        df = df.loc[lambda x: x.internal_id.str[0] == internal_id]
    return df


def append_region_name(df):
    """
    Append the region name as additional column to a DataFrame or Series.

    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        required index: NUTS-3 codes

    Returns
    -------
    pd.DataFrame
        with additional column
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.assign(region_name=lambda x:
                     x.index.map(region_id_to_nuts3(nuts3_to_name=True)))


def transpone_spatiotemporal(df, freq='1H', **kwargs):
    """
    Transpone a spatiotemporal pd.DataFrame and set/reset the pd.DateTimeIndex.

    Parameters
    ----------
    df : pd.DataFrame
        required index: either NUTS-3 codes or pd.DatetimeIndex
    freq : str
        frequency of time series as offset alias. Examples:
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

    Returns
    -------
    pd.DataFrame
    """
    if isinstance(df.index, pd.DatetimeIndex):
        # put timesteps in columns and regions in index
        return df.reset_index(drop=True).T
    else:
        # put regions in columns and timesteps in index
        year = kwargs.get('year', cfg['base_year'])
        idx = pd.date_range(start=str(year), periods=len(df.columns),
                            freq=freq)
        return df.T.set_index(idx)
