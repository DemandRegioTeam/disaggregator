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

from .data import (elc_consumption_HH, heat_consumption_HH, gas_consumption_HH,
                   population, households_per_size, income, stove_assumptions,
                   living_space, hotwater_shares, heat_demand_buildings,
                   employees_per_branch_district, efficiency_enhancement,
                   generate_specific_consumption_per_branch_and_district)
from .config import (data_in, dict_region_code, get_config)

import pandas as pd
import os
import datetime
import logging
logger = logging.getLogger(__name__)
cfg = get_config()


def disagg_households_power(by, weight_by_income=False):
    """
    Perform spatial disaggregation of electric power in [GWh/a] by key and
    possibly weight by income.

    Parameters
    ----------
    by : str
        must be one of ['households', 'population']
    weight_by_income : bool, optional
        Flag if to weight the results by the regional income (default False)

    Returns
    -------
    pd.DataFrame or pd.Series
    """
    if by == 'households':
        # Bottom-Up: Power demand by household sizes in [GWh/a]
        power_per_HH = elc_consumption_HH(by_HH_size=True) / 1e3
        df = households_per_size() * power_per_HH
    elif by == 'population':
        # Top-Down: Power demand for entire country in [GWh/a]
        power_nuts0 = elc_consumption_HH() / 1e3
        distribution_keys = population() / population().sum()
        df = distribution_keys * power_nuts0
    else:
        raise ValueError("`by` must be in ['households', 'population']")

    if weight_by_income:
        df = adjust_by_income(df=df)

    return df


def disagg_households_heat(by, weight_by_income=False):
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
    if by not in ['households', 'buildings']:
        raise ValueError('The heating demand of households depends mainly on '
                         'the different household sizes and the building '
                         'types but not on the absolute population. Thus, '
                         'please pass `by=` either as "households" or '
                         '"buildings".')

    # Bottom-Up: Heat demand by household sizes in [MWh/a]
    df_heat_specific = heat_consumption_HH(by=by).T.unstack()
    df = pd.DataFrame(columns=df_heat_specific.index)
    base = households_per_size() if by == 'households' else living_space()
    for col, ser in base.iteritems():
        for idx in heat_consumption_HH(by=by).index:
            df.loc[:, (idx, col)] = ser
    df *= df_heat_specific
    return df


def disagg_households_gas(how='top-down', weight_by_income=False):
    """
    Perform spatial disaggregation of gas demand and possibly adjust
    by income.

    Parameters
    ----------
    how : str, optional
        must be one of ['top-down', 'bottom-up', 'bottom-up_2']
    adjust_by_income : bool, optional
        Flag if to weight the results by the regional income (default False)

    Returns
    -------
    pd.DataFrame or pd.Series
    """
    gas_nuts0 = gas_consumption_HH()
    # Derive distribution keys
    df_ls_gas = living_space(aggregate=True,
                             internal_id=[None, None, 11, 1]).sum(axis=1)
    df_pop = population()
    df_HH = households_per_size().sum(axis=1)
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
        new_m2_vintages = {'A_<1900': 'A_<1948',
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
        new_dem_vintages = {'A_<1859': 'A_<1948',
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
        df_heat_dem = disagg_households_heat(by='households')

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

def disagg_households_heatload_DB(how='top-down', weight_by_income=False):
    """
    Perform spatial disaggregation of gas demand and possibly adjust
    by income.

    Parameters
    ----------
    how : str, optional
        must be one of ['top-down', 'bottom-up', 'bottom-up_2']
    adjust_by_income : bool, optional
        Flag if to weight the results by the regional income (default False)

    Returns
    -------
    pd.DataFrame or pd.Series
    """
    # Multiply gas consumption to match overall heatload
    gas_nuts0 = gas_consumption_HH()
    gas_nuts0['HotWater'] = gas_nuts0['HotWater'] / 0.47
    gas_nuts0['SpaceHeating'] = gas_nuts0['SpaceHeating'] / 0.47

    # Derive distribution keys
    df_ls_all = living_space(aggregate=True,
                             internal_id=[None, None, None, 1]).sum(axis=1)
    df_pop = population()
    df_HH = households_per_size().sum(axis=1)
    d_keys_hotwater = df_pop / df_pop.sum()
    d_keys_cook = df_HH / df_HH.sum()

    if how == 'top-down':
        logger.info('Calculating regional gas demands top-down.')
        #d_keys_space = df_ls_gas / df_ls_gas.sum()
        d_keys_space = df_ls_all / df_ls_all.sum()
        # Calculate
        df = (pd.DataFrame(index=df_ls_all.index)
                .assign(Cooking=d_keys_cook * gas_nuts0['Cooking'],
                        HotWater=d_keys_hotwater * gas_nuts0['HotWater'],
                        SpaceHeating=d_keys_space * gas_nuts0['SpaceHeating']))

    elif how == 'bottom-up':
        logger.info('Calculating regional gas demands bottom-up.')
        # uniform non-matching vintage sections
        new_m2_vintages = {'A_<1900': 'A_<1948',
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
        new_dem_vintages = {'A_<1859': 'A_<1948',
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
                                  internal_id=[None, None, None, 1])
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
        df_erg = piv_dem.multiply(piv_m2) # No Boiler existing
        df_spaceheat = df_erg.sum(axis=1)
        # Calculate
        df = (pd.DataFrame(index=df_spaceheat.index)
                .assign(Cooking=d_keys_cook * gas_nuts0['Cooking'],
                        HotWater=d_keys_hotwater * gas_nuts0['HotWater'],
                        SpaceHeating=df_spaceheat))

    if weight_by_income:
        df = adjust_by_income(df=df)

    return df

def disagg_households_heatload(year, weight_by_income=False):
    """
    Perform spatial disaggregation of gas demand and possibly adjust
    by income.

    Parameters
    ----------
    adjust_by_income : bool, optional
        Flag if to weight the results by the regional income (default False)

    Returns
    -------
    pd.DataFrame or pd.Series
    """
    # Derive distribution keys
    gas_nuts0 = gas_consumption_HH()
    df_ls = living_space(aggregate=True, internal_id_3=1).sum(axis=1)
    df_pop = population()
    df_HH = households_per_size().sum(axis=1)
    d_keys_hotwater = df_pop / df_pop.sum()
    d_keys_cook = df_HH / df_HH.sum()

    logger.info('Calculating regional gas demands bottom-up.')
    # uniform non-matching vintage sections
    new_m2_vintages = {'A_<1900': 'A_<1948',
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
    new_dem_vintages = {'A_<1859': 'A_<1948',
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
    df_ls = (living_space(aggregate=False, year=2018)   #internal_id_3=1
                 .drop(['heating_system', 'non_empty_building'],
                       axis=1)
                 .replace(dict(vintage_class=new_m2_vintages)))

    # load the specific heating demands
    # df1 = not refurbished buildings
    df1 = (heat_demand_buildings(table_id=56, year=year,
                                 internal_id_2=1, internal_id_3=1)
           .replace(dict(vintage_class=new_dem_vintages))
           .loc[lambda x: x.vintage_class != 'A_<1948'])
    # df2 = refurbished buildings
    df2 = (heat_demand_buildings(table_id=56, year=year,
                                 internal_id_2=1, internal_id_3=2)
           .replace(dict(vintage_class=new_dem_vintages))
           .loc[lambda x: x.vintage_class == 'A_<1948'])
    #df_heat_dem = pd.concat([df1, df2])
    df_heat_dem = (heat_demand_buildings(table_id=56, year=year,
                                 internal_id_3=1)
           .replace(dict(vintage_class=new_dem_vintages))
           .loc[lambda x: x.vintage_class != 'A_<1948'])

    piv_dem = df_heat_dem.pivot_table(values='value', index='nuts3',
                                      columns='vintage_class',
                                      aggfunc='mean') / 1e3  # kWh -> MWh
    piv_m2 = df_ls.pivot_table(values='value', index='nuts3',
                                   columns='vintage_class',
                                   aggfunc='sum')
    df_erg = piv_dem.multiply(piv_m2) / 0.99  # eta-boiler assumption
    df_spaceheat = df_erg.sum(axis=1)
    # Calculate
    df = (pd.DataFrame(index=df_spaceheat.index)
            .assign(Cooking=d_keys_cook * gas_nuts0['Cooking'],
                    HotWater=d_keys_hotwater * gas_nuts0['HotWater'],
                    SpaceHeating=df_spaceheat))

    if weight_by_income:
        df = adjust_by_income(df=df)

    return df

def disagg_CTS_industry(source, sector,
                        use_nuts3code=False, no_self_gen=False):
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
    [spez_sv, spez_gv] = generate_specific_consumption_per_branch_and_district(
                                                              8, 8,no_self_gen)
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
        data=(employees_per_branch_district().loc[spez_vb.index].values
              * spez_vb.values),
        index=spez_vb.index,
        columns=spez_vb.columns))
    df = df.multiply(efficiency_enhancement(source).transpose().loc[df.index],
                     axis=0)
    if use_nuts3code:
        df = df.rename(columns=dict_region_code(keys='ags_lk',
                                                values='natcode_nuts3'))
    return df


# --- Utility functions -------------------------------------------------------


def adjust_by_income(df):
    income_keys = income() / income().mean()
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
