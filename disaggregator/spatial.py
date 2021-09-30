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
Provides functions for spatial disaggregation
"""
# %% Imports

from .data import (elc_consumption_HH, heat_consumption_HH, gas_consumption_HH,
                   population, households_per_size, income, stove_assumptions,
                   living_space, hotwater_shares, heat_demand_buildings,
                   employees_per_branch, efficiency_enhancement,
                   generate_specific_consumption_per_branch_and_district)
from .config import (dict_region_code, get_config)
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)


# %%  Generic functions


def disagg_topdown(total, keys1, keys2=None, names=None):
    """
    Disaggregate DataFrame `df` by regional `keys1` and optionally by `keys2`.

    Example `keys1` could be federal states (BL) and `keys2` districts (LK).

    Parameters
    ----------
    total : float or list or pd.Series
        The total value(s) to be disaggregated
    keys1 : pd.Series or dict
        The distribution keys
    keys2 : pd.Series or dict
        The distribution keys
    names : list, optional

    Returns
    -------
        pd.DataFrame
    """
    # Cleanup and prepare data
    if isinstance(total, float) or isinstance(total, list):
        tot = pd.Series(data=total, index=names)
    elif isinstance(total, pd.Series) or isinstance(total, dict):
        tot = pd.Series(total)
    else:
        raise ValueError('`total` must be float, list, dict or pd.Series!')
    if isinstance(keys1, pd.Series) or isinstance(keys1, dict):
        keys1 = pd.Series(keys1)
    else:
        raise ValueError('`keys1` must be dict or pd.Series!')
    if keys2 is not None:
        if isinstance(keys1, pd.Series) or isinstance(keys1, dict):
            keys1 = pd.Series(keys1)
        else:
            raise ValueError('`keys1` must be dict or pd.Series!')

    # Disaggregate by keys1
    ser_k1 = keys1 / keys1.sum()
    df = pd.DataFrame(data=np.outer(ser_k1, tot),
                      index=ser_k1.index, columns=tot.index)
    cols_orig = df.columns
    df.columns = [str(c) for c in df.columns]

    # Disaggregate by keys2
    if keys2 is not None:
        keys2.name = 'original'
        df_k2 = (keys2.reset_index().rename(columns={'index': 'nuts3'})
                      .assign(nuts1=lambda x: x.nuts3.str[0:3]))
        # If in one nuts1 region all are zero, then replace all zeros by one,
        # in order to avoid division by zero and achieve equal distribution
        zero_regs = list(df_k2.groupby('nuts1').sum()
                              .loc[lambda x: x.original == 0.0].index)
        df_k2.loc[df_k2['nuts1'].isin(zero_regs), 'original'] = 1

        nuts1_sums = (df_k2.groupby('nuts1').agg(sum).reset_index()
                           .rename(columns={'original': 'sums'}))
        df_k2 = (df_k2.merge(nuts1_sums, how='left', on='nuts1')
                      .assign(ratio=lambda x: x['original']/x['sums']))
        for col, ser in df.iteritems():
            df_k2 = df_k2.assign(**{col: lambda x: x.nuts1.map(ser) * x.ratio})
        df = df_k2.set_index('nuts3').reindex(df.columns, axis=1)

    # restore original column namings:
    df.columns = cols_orig
    return df


# %%  Sector-specific functions


def disagg_households_power(by, weight_by_income=False, scale_by_pop=False,
                            **kwargs):
    """
    Spatial disaggregation of elc. power in [GWh/a] by key (weighted by income)

    Parameters
    ----------
    by : str
        must be one of ['households', 'population']
    weight_by_income : bool, optional
        Flag if to weight the results by the regional income (default False)
    orignal : bool, optional
        Throughput to function households_per_size,
        A flag if the results should be left untouched and returned in
        original form for the year 2011 (True) or if they should be scaled to
        the given `year` by the population in that year (False).

    Returns
    -------
    pd.DataFrame or pd.Series
    """
    cfg = kwargs.get('cfg', get_config())
    year = kwargs.get('year', cfg['base_year'])
    if by == 'households':
        # Bottom-Up: Power demand by household sizes in [GWh/a]
        power_per_HH = elc_consumption_HH(by_HH_size=True, year=year) / 1e3
        df = (households_per_size(original=scale_by_pop, year=year)
              * power_per_HH)
    elif by == 'population':
        # Top-Down: Power demand for entire country in [GWh/a]
        power_nuts0 = kwargs.get('power_nuts0', False)
        if power_nuts0 is False:
            power_nuts0 = elc_consumption_HH(year=year) / 1e3
        distribution_keys = population(year=year) / population(year=year).sum()
        df = distribution_keys * power_nuts0
    else:
        raise ValueError("`by` must be in ['households', 'population']")

    if weight_by_income:
        df = adjust_by_income(df=df)

    return df


def disagg_households_heat(by, weight_by_income=False, **kwargs):
    """
    Perform spatial disaggregation of heat demand in [MWh/a] by key.

    Parameters
    ----------
    by : str
        must be one of ['households', 'buildings']

    Returns
    -------
    pd.DataFrame
    """
    cfg = kwargs.get('cfg', get_config())
    year = kwargs.get('year', cfg['base_year'])
    if by not in ['households', 'buildings']:
        raise ValueError('The heating demand of households depends mainly on '
                         'the different household sizes and the building '
                         'types but not on the absolute population. Thus, '
                         'please pass `by=` either as "households" or '
                         '"buildings".')

    # Bottom-Up: Heat demand by household sizes in [MWh/a]
    df_heat_specific = heat_consumption_HH(by=by, year=year).T.unstack()
    df = pd.DataFrame(columns=df_heat_specific.index)
    base = households_per_size() if by == 'households' else living_space()
    for col, ser in base.iteritems():
        for idx in heat_consumption_HH(by=by, year=year).index:
            df.loc[:, (idx, col)] = ser
    df *= df_heat_specific
    return df


def disagg_households_gas(how='top-down', weight_by_income=False,
                          original=False, **kwargs):
    """
    Return spatial disaggregation of gas demand (possibly adjusted by income).

    Parameters
    ----------
    how : str, optional
        must be one of ['top-down', 'bottom-up', 'bottom-up_2']
    adjust_by_income : bool, optional
        Flag if to weight the results by the regional income (default False)
    orignal : bool, optional
        Throughput to function households_per_size,
        A flag if the results should be left untouched and returned in
        original form for the year 2011 (True) or if they should be scaled to
        the given `year` by the population in that year (False).

    Returns
    -------
    pd.DataFrame or pd.Series
    """
    cfg = kwargs.get('cfg', get_config())
    year = kwargs.get('year', cfg['base_year'])
    gas_nuts0 = gas_consumption_HH(year=year)
    # Derive distribution keys
    df_ls_gas = living_space(aggregate=True,
                             internal_id=[None, None, 11, 1]).sum(axis=1)
    df_pop = population(year=year)
    df_HH = households_per_size(year=year, original=original).sum(axis=1)
    d_keys_hotwater = df_pop / df_pop.sum()
    d_keys_cook = df_HH / df_HH.sum()

    if how == 'top-down':
        logger.info('Calculating regional gas demands top-down.')
        d_keys_space = df_ls_gas / df_ls_gas.sum()
        # Calculate
        df = (pd.DataFrame(index=df_ls_gas.index)
                .assign(Cooking=d_keys_cook * gas_nuts0['Cooking'],
                        HotWater=d_keys_hotwater * gas_nuts0['HotWater'],
                        SpaceHeating=d_keys_space * gas_nuts0['SpaceHeating']))

    elif how == 'bottom-up':
        logger.info('Calculating regional gas demands bottom-up.')
        # uniform non-matching vintage sections
        new_m2_vintages = {'A_<1900':     'A_<1948',
                           'B_1900-1945': 'A_<1948',
                           'C_1946-1960': 'B_1949-1968',
                           'D_1961-1970': 'B_1949-1968',
                           'E_1971-1980': 'C_1969-1985',
                           'F_1981-1985': 'C_1969-1985',
                           'G_1986-1995': 'D_1986-1995',
                           'H_1996-2000': 'E_1996-2000',
                           'I_2001-2005': 'F_>2000',
                           'J_2006-2011': 'F_>2000',
                           'K_2012': 'F_>2000',
                           'L_2013': 'F_>2000',
                           'M_2014': 'F_>2000',
                           'N_2015': 'F_>2000',
                           'O_2016': 'F_>2000',
                           'P_2017': 'F_>2000',
                           'Q_2018': 'F_>2000',
                           'R_2019': 'F_>2000'}
        new_dem_vintages = {'A_<1859':     'A_<1948',
                            'B_1860-1918': 'A_<1948',
                            'C_1919-1948': 'A_<1948',
                            'D_1949-1957': 'B_1949-1968',
                            'E_1958-1968': 'B_1949-1968',
                            'F_1969-1978': 'C_1969-1985',
                            'G_1979-1983': 'C_1969-1985',
                            'H_1984-1994': 'D_1986-1995',
                            'I_1995-2001': 'E_1996-2000',
                            'J_2002-2009': 'F_>2000'}

        # load the to-be-heated m² in spatial resolution
        df_ls_gas = (living_space(aggregate=False, year=2018,
                                  internal_id=[None, None, 11, 1])
                     .drop(['heating_system', 'non_empty_building'], axis=1)
                     .replace(dict(vintage_class=new_m2_vintages)))

        # load the specific heating demands
        # df1 = not refurbished buildings
        df1 = (heat_demand_buildings(table_id=56, year=2018,
                                     internal_id=[None, None, 1, 1])
               .replace(dict(vintage_class=new_dem_vintages))
               .loc[lambda x: x.vintage_class != 'A_<1948'])
        # df2 = refurbished buildings
        df2 = (heat_demand_buildings(table_id=56, year=2018,
                                     internal_id=[None, None, 1, 2])
               .replace(dict(vintage_class=new_dem_vintages))
               .loc[lambda x: x.vintage_class == 'A_<1948'])
        df_heat_dem = pd.concat([df1, df2])

        piv_dem = df_heat_dem.pivot_table(values='value', index='nuts3',
                                          columns='vintage_class',
                                          aggfunc='mean') / 1e3  # kWh -> MWh
        piv_m2 = df_ls_gas.pivot_table(values='value', index='nuts3',
                                       columns='vintage_class',
                                       aggfunc='sum')
        df_erg = piv_dem.multiply(piv_m2) / 0.99  # eta-boiler assumption
        df_spaceheat = df_erg.sum(axis=1)
        # Calculate
        df = (pd.DataFrame(index=df_spaceheat.index)
                .assign(Cooking=d_keys_cook * gas_nuts0['Cooking'],
                        HotWater=d_keys_hotwater * gas_nuts0['HotWater'],
                        SpaceHeating=df_spaceheat))

    elif how == 'bottom-up_2':
        logger.warning("This feature is currently experimental and should not "
                       "be used as long as you don't know what you're doing!")
        # The bottom-up_2 logic requires the heat demand of households
        df_heat_dem = disagg_households_heat(by='households', year=year)

        logger.info('Calculating regional gas demands bottom-up:')
        logger.info('1. Cooking based on household sizes in [MWh/a]')
        df_stove = stove_assumptions()
        df_heat_cook = pd.DataFrame(df_heat_dem['Cooking'])
        for idx, row in df_stove.iterrows():
            df_heat_cook.update(df_heat_cook.filter(axis=0, regex=idx)
                                * row.stoves_percentage_gas
                                / row.stoves_efficiency_gas)

        logger.info('2. Hot water (decentralised) based on household sizes'
                    ' in [MWh/a]')
        df_WW_shares = hotwater_shares()
        df_WW = pd.DataFrame(df_heat_dem['HotWater'])
        for idx, row in df_WW_shares.iterrows():
            df_WW.update(df_WW.filter(axis=0, regex=idx)
                         * row.share_decentralised_gas
                         / 0.95)  # efficiency assumption gas boilers

        logger.info('3. Space heating + hot water (centralised) based on '
                    'living space in [MWh/a]')

        df_hc_only = (heat_consumption_HH(by='buildings')
                      .T.loc[:, 'SpaceHeatingOnly'])
        df_hc_HW = (heat_consumption_HH(by='buildings')
                    .T.loc[:, 'SpaceHeatingPlusHotWater'])
        df_spaceheat = df_ls_gas.multiply(df_hc_only)
        df_spaceheat_HW = df_ls_gas.multiply(df_hc_HW)
        for idx, row in df_WW_shares.iterrows():
            df_spaceheat.update(df_spaceheat.filter(axis=0, regex=idx)
                                * (1.0 - row.share_centralised))
            df_spaceheat_HW.update(df_spaceheat_HW.filter(axis=0, regex=idx)
                                   * row.share_centralised)

        logger.info('4. Merging results')
        df = (pd.DataFrame(index=df_heat_dem.index)
                .assign(Cooking=df_heat_cook.sum(axis=1),
                        HotWaterDecentral=df_WW.sum(axis=1),
                        SpaceHeatingOnly=df_spaceheat.sum(axis=1),
                        SpaceHeatingPlusHotWater=df_spaceheat_HW.sum(axis=1)))

    if weight_by_income:
        df = adjust_by_income(df=df)

    return df


def disagg_CTS_industry(source, sector,
                        use_nuts3code=False, no_self_gen=False, **kwargs):
    """
    Perform spatial disaggregation of electric power or gas in [MWh/a]

    Parameters
    ----------
    source : str
        must be one of ['power', 'gas']
    sector : str
        must be one of ['CTS', 'industry']
    use_nuts3code : bool, default False
        If True use NUTS-3 codes as region identifiers.
    no_self_gen : bool, default False
        throughput for
        data.generate_specific_consumption_per_branch_and_district(
                                                            no_self_gen=False)
        If True: returns specific power and gas consumption without self
                 generation, resulting energy consumption will be lower

    Returns
    -------
    pd.DataFrame
        index = Branches
        columns = Districts
    """
    assert (source in ['power', 'gas']), "`source` must be in ['power', 'gas']"
    assert (sector in ['CTS', 'industry']),\
        "`sector` must be in ['CTS', 'industry']"

    # generate specific consumptions
    cfg = kwargs.get('cfg', get_config())
    year = kwargs.get('year', cfg['base_year'])
    [spez_sv, spez_gv] = generate_specific_consumption_per_branch_and_district(
        8, 8, no_self_gen, year=year)
    if source == 'power':
        spez_vb = spez_sv
    else:
        spez_vb = spez_gv
    spez_vb.columns = spez_vb.columns.astype(int)

    if sector == 'industry':
        wz = list(range(5, 34))  # = [5, 6, ..., 33]
    if sector == 'CTS':
        wz = [1, 2, 3, 36, 37, 38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 51, 52,
              53, 55, 56, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68, 69, 70, 71,
              72, 73, 74, 75, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90,
              91, 92, 93, 94, 95, 96, 97, 98, 99]

    spez_vb = spez_vb.loc[wz]
    df = (pd.DataFrame(
        data=(employees_per_branch(year=year).loc[spez_vb.index].values
              * spez_vb.values),
        index=spez_vb.index,
        columns=spez_vb.columns))
    df = df.multiply(efficiency_enhancement(source, year=year).transpose()
                     .loc[df.index], axis=0)
    if use_nuts3code:
        df = df.rename(columns=dict_region_code(keys='ags_lk',
                                                values='natcode_nuts3'))
    return df


# %% Utility functions


def adjust_by_income(df, **kwargs):
    cfg = kwargs.get('cfg', get_config())
    year = kwargs.get('year', cfg['base_year'])
    income_keys = income(year=year) / income(year=year).mean()
    return df.multiply(income_keys, axis=0)


def aggregate_to_nuts1(df, agg='sum'):
    """
    Re-aggregate to NUTS-1 level from NUTS-3 level data.

    Parameters
    ----------
    df : pd.DataFrame
        index: nuts3-codes
    agg : str
        The aggregation function key. The default is 'sum'.

    Returns
    -------
    pd.DataFrame
    """
    # Check if passed df is actually a Series and convert to DataFrame
    was_series = False
    if isinstance(df, pd.Series):
        name = df.name
        df = df.to_frame()
        was_series = True
    # Aggregate
    df = df.assign(nuts1=lambda x: x.index.str[0:3]).groupby('nuts1').agg(agg)
    # Restore Series if it was one.
    if was_series:
        df = df[name]
    return df
