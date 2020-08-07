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
Provides functions to import all relevant data.
"""

import pandas as pd
import numpy as np
import logging
import holidays
import datetime
from collections import OrderedDict
from collections.abc import Iterable
from .config import (get_config, data_in, data_out, database_raw,
                     dict_region_code, literal_converter, wz_dict,
                     hist_weather_year, gas_load_profile_parameters_dict)
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
            df = read_local(data_in('dimensionless', cfg[key]['filename']))
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
            df = float(read_local(data_in('dimensionless',
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

    if source == 'local':
        col = {'households': 'PersonsInHousehold',
               'buildings': 'BuildingType'}[by]
        df = (pd.read_csv(data_in('dimensionless',
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
        df = read_local(data_in('dimensionless', cfg[key]['filename']),
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
    df = (pd.read_csv(data_in('temporal', 'percentages_applications.csv'),
                      index_col='Application', engine='c')
            .drop(labels='all', axis=1))
    df.columns = df.columns.astype(int)
    return df


def zve_percentages_baseload():
    df = (pd.read_csv(data_in('temporal', 'percentages_baseload.csv'),
                      index_col='Application', engine='c')
            .drop(labels='all', axis=1))
    df.columns = df.columns.astype(int)
    return df


def zve_application_profiles():
    return pd.read_csv(data_in('temporal', 'application_profiles.csv'),
                       engine='c')


def t_allo(**kwargs):
    """
    Returns allocation temperature from weather data for (historical) year

    Returns
    -------
    pd.DataFrame
    """
    year = kwargs.get('year', cfg['base_year'])
    hist_year = hist_weather_year().get(year)
    dic_nuts3 = (dict_region_code(raw=True)[['natcode_nuts3', 'ags_lk']]
                 .set_index('natcode_nuts3'))
    dic_nuts3['ags_lk'] = dic_nuts3['ags_lk'].astype(str).str.zfill(5)
    if ((hist_year % 4 == 0)
            & (hist_year % 100 != 0)
            | (hist_year % 4 == 0)
            & (hist_year % 100 == 0)
            & (hist_year % 400 == 0)):
        periods = 35136
    else:
        periods = 35040
    df = ambient_T(year=hist_year, internal_id=2)
    df = (df.assign(date=pd.date_range((str(hist_year) + '-01-01'),
                    periods=periods / 4, freq='H'))
            .set_index('date').resample('D').mean())
    df = (pd.merge(df.transpose(), dic_nuts3, how='right',
                   left_index=True, right_index=True)
            .set_index('ags_lk').transpose())
    # dropping districts '03152' and '03156' which do not exist anymore
    df = df.dropna(axis='columns')
    # Workaround for dropping leading Zeros
    df.columns = df.columns.astype(int).astype(str)
    for district in df.columns:
        te = df[district].values
        for i in range(len(te)-1, -1, -1):
            if (i >= 3):
                te[i] = ((te[i] + 0.5 * te[i - 1] + 0.25 * te[i - 2]
                         + te[i - 3] * 0.125) / 1.875)
        df[district] = te
    return df


def h_value(slp, districts, temperatur_df):
    """
    Returns h-values depending on allocation temperature  for every
    district.

    Parameter
    -------
    slp : str
        Must be one of ['BA', 'BD', 'BH', 'GA', 'GB', 'HA',
                        'KO', 'MF', 'MK', 'PD', 'WA']
    districts : list of district keys in state e.g. ['11000'] for Berlin

    Returns
    -------
    pd.DataFrame
    """
    temp_df = temperatur_df.copy()[[x for x in districts]]
    par = gas_load_profile_parameters_dict()
    A = par['A'][slp]
    B = par['B'][slp]
    C = par['C'][slp]
    D = par['D'][slp]
    mH = par['mH'][slp]
    bH = par['bH'][slp]
    mW = par['mW'][slp]
    bW = par['bW'][slp]
    for landkreis in districts:
        te = temp_df[landkreis].values
        for i in range(len(te)):
            temp_df[landkreis][i] = ((A / (1 + pow(B / (te[i] - 40), C)) + D)
                                     + max(mH * te[i] + bH, mW * te[i] + bW))
    return temp_df


def generate_specific_consumption_per_branch(**kwargs):
    """
    Returns specific power and gas consumption per branch. Also returns total
    power and gas consumption per branch and also the amount of workers per
    branch and district.

    Returns
    ------------
    Tuple that contains six pd.DataFrames
    """
    year = kwargs.get('year', cfg['base_year'])
    # get electricity and gas consumption from database
    x = True
    year1 = year
    if year1 not in range(2000, 2036):
        raise ValueError("`year` must be between 2000 and 2035")
    while(x):
        try:
            vb_wz = database_get('spatial', table_id=71, year=year1)
            x = False
        except ValueError:
            year1 -= 1
    vb_wz = (vb_wz.assign(WZ=[x[0] for x in vb_wz['internal_id']],
                          ET=[x[1] for x in vb_wz['internal_id']]))
    vb_wz = (vb_wz[(vb_wz['ET'] == 12)
                   | (vb_wz['ET'] == 18)])[['value', 'WZ', 'ET']]
    vb_wz = vb_wz.loc[vb_wz['WZ']
                      .isin(list(wz_dict().keys()))]
    vb_wz = vb_wz.replace({'WZ': wz_dict()})
    vb_wz['value'] = vb_wz['value'] * 1000 / 3.6
    sv_wz_real = (vb_wz.loc[vb_wz['ET'] == 18][['WZ', 'value']]
                       .groupby(by='WZ')[['value']].sum()
                       .rename(columns={'value': 'SV WZ [MWh]'}))
    gv_wz_real = (vb_wz.loc[vb_wz['ET'] == 12][['WZ', 'value']]
                       .groupby(by='WZ')[['value']].sum()
                       .rename(columns={'value': 'GV WZ [MWh]'}))
    # get number of employees (bze) from database
    bze_je_lk_wz = pd.DataFrame(employees_per_branch_district(year=year))
    bze_lk_wz = (pd.DataFrame(0.0, index=bze_je_lk_wz.columns,
                              columns=wz_dict().values()))
    # arrange employees DataFrame accordingly to energy consumption statistics
    for i in [1, 2, 3, 5, 6]:
        bze_lk_wz[str(i)] = bze_je_lk_wz.transpose()[i]
    for i in [7, 8, 9]:
        bze_lk_wz['7-9'] = bze_lk_wz['7-9'] + bze_je_lk_wz.transpose()[i]
    for i in [10, 11, 12]:
        bze_lk_wz['10-12'] = bze_lk_wz['10-12'] + bze_je_lk_wz.transpose()[i]
    for i in [13, 14, 15]:
        bze_lk_wz['13-15'] = bze_lk_wz['13-15'] + bze_je_lk_wz.transpose()[i]
    for i in [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
        bze_lk_wz[str(i)] = bze_je_lk_wz.transpose()[i]
    bze_lk_wz['31-32'] = (bze_je_lk_wz.transpose()[31]
                          + bze_je_lk_wz.transpose()[32])
    for i in [33, 35, 36, 37]:
        bze_lk_wz[str(i)] = bze_je_lk_wz.transpose()[i]
    bze_lk_wz['38-39'] = (bze_je_lk_wz.transpose()[38]
                          + bze_je_lk_wz.transpose()[39])
    bze_lk_wz['41-42'] = (bze_je_lk_wz.transpose()[41]
                          + bze_je_lk_wz.transpose()[42])
    for i in [43, 45, 46, 47, 49, 50, 51, 52, 53]:
        bze_lk_wz[str(i)] = bze_je_lk_wz.transpose()[i]
    bze_lk_wz['55-56'] = (bze_je_lk_wz.transpose()[55]
                          + bze_je_lk_wz.transpose()[56])
    for i in [58, 59, 60, 61, 62, 63]:
        bze_lk_wz['58-63'] = bze_lk_wz['58-63'] + bze_je_lk_wz.transpose()[i]
    for i in [64, 65, 66]:
        bze_lk_wz['64-66'] = bze_lk_wz['64-66'] + bze_je_lk_wz.transpose()[i]
    bze_lk_wz[str(68)] = bze_je_lk_wz.transpose()[68]
    for i in [69, 70, 71, 72, 73, 74, 75]:
        bze_lk_wz['69-75'] = bze_lk_wz['69-75'] + bze_je_lk_wz.transpose()[i]
    for i in [77, 78, 79, 80, 81, 82]:
        bze_lk_wz['77-82'] = bze_lk_wz['77-82'] + bze_je_lk_wz.transpose()[i]
    for i in [84, 85]:
        bze_lk_wz[str(i)] = bze_je_lk_wz.transpose()[i]
    bze_lk_wz['86-88'] = (bze_je_lk_wz.transpose()[86]
                          + bze_je_lk_wz.transpose()[87]
                          + bze_je_lk_wz.transpose()[88])
    for i in [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]:
        bze_lk_wz['90-99'] = bze_lk_wz['90-99'] + bze_je_lk_wz.transpose()[i]
    # calculate specific consumption
    spez_gv = (pd.DataFrame(bze_lk_wz.transpose().drop_duplicates()
                            .sum(axis=1))
                 .merge(gv_wz_real, left_index=True, right_index=True))
    spez_gv.loc[:, 'spez. GV'] = ((spez_gv['GV WZ [MWh]'] / spez_gv[0])
                                  .transpose())
    spez_gv = spez_gv[['spez. GV']].transpose()
    spez_sv = (pd.DataFrame(bze_lk_wz.transpose().drop_duplicates()
                            .sum(axis=1))
                 .merge(sv_wz_real, left_index=True, right_index=True))
    spez_sv.loc[:, 'spez. SV'] = spez_sv['SV WZ [MWh]'] / spez_sv[0]
    spez_sv = spez_sv[['spez. SV']].transpose()
    # assign specific consumption of grouped industry branches to each branch
    for item in [[7, 8, 9], [10, 11, 12], [13, 14, 15], [31, 32], [38, 39],
                 [41, 42], [55, 56], [58, 59, 60, 61, 62, 63], [64, 65, 66],
                 [69, 70, 71, 72, 73, 74, 75], [77, 78, 79, 80, 81, 82],
                 [86, 87, 88], [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]]:
        for i in item:
            spez_gv[i] = spez_gv[str(item[0]) + "-" + str(item[-1])]
            spez_sv[i] = spez_sv[str(item[0]) + "-" + str(item[-1])]
    spez_gv = spez_gv.drop(columns=['7-9', '10-12', '13-15', '31-32', '38-39',
                                    '41-42', '55-56', '58-63', '64-66',
                                    '69-75', '77-82', '86-88',
                                    '90-99']).transpose()
    spez_gv.index = spez_gv.index.astype(int)
    spez_sv = spez_sv.drop(columns=['7-9', '10-12', '13-15', '31-32', '38-39',
                                    '41-42', '55-56', '58-63', '64-66',
                                    '69-75', '77-82', '86-88',
                                    '90-99']).transpose()
    spez_sv.index = spez_sv.index.astype(int)

    # original source (table_id = 38) gives sum of natural gas and other gases
    # use factor from sheet to decompose energy consumption
    f = ('Decomposition Factors Industrial Energy Demand.xlsx')
    df_decom = pd.read_excel(data_in('dimensionless', f),
                               sheet_name='Tabelle1')
    df_decom.set_index('WZ', inplace=True)
    df_decom = spez_gv.merge(df_decom, how='left',
                             left_index=True, right_index=True)
    df_decom['Strom Eigenerzeugung'].fillna(0, inplace=True)
    df_decom.fillna(1, inplace=True)
    spez_gv['spez. GV'] = spez_gv['spez. GV'] * df_decom['Anteil Erdgas']

    # original source (table_id = 38) does not include gas consumption for
    # self generation in industrial sector
    # get gas consumption for self_generation from German energy balance
    x = True
    year1 = year
    while(x):
        try:
            df_balance = pd.read_excel(data_in('dimensionless',
                                               'bilanz'+str(year1)[-2:]+'d.xlsx'),
                                       sheet_name='nat', skiprows=3)
            x = False
        except FileNotFoundError:
            year1 -= 1
    # tbd import energy balance from database
    df_balance.rename(columns={"Unnamed: 1": "Zeile",
                               "Unnamed: 24": "Grubengas",
                               "Naturgase": "Erdgas in Mio kWh"},
                      inplace=True)
    df_balance.drop([0, 1, 2], inplace=True)
    df_balance.set_index('Zeile', inplace=True)
    # locate natural gas consumption for self generation in energy balance
    # Unit is in GWh (Mio kWh) is transformed to MWh
    GV_slf_gen_global = df_balance['Erdgas in Mio kWh'].loc[12]*1000
    # assign global gas consumption fo self gen to industry branches
    # according to their electricity generation from industrial powerplants
    df_help_sv = spez_sv.assign(BZE=bze_je_lk_wz.sum(axis=1),
                                SV_WZ_MWh=lambda x: x['spez. SV'] * x['BZE'],
                                f_SV_self_gen=df_decom['Strom Eigenerzeugung'],
                                SV_self_gen=lambda x:
                                    x.f_SV_self_gen * x.SV_WZ_MWh,
                                f_GV_self_gen=lambda x:
                                    x.SV_self_gen / x.SV_self_gen.sum(),
                                GV_self_gen=lambda x:
                                    x.f_GV_self_gen * GV_slf_gen_global,
                                spez_GV_self_gen=lambda x:
                                    x.GV_self_gen / x.BZE)
    df_help_gv = spez_gv.assign(spez_GV_self_gen=df_help_sv.spez_GV_self_gen,
                                spez_GV_final=lambda x:
                                    x['spez. GV'] + x.spez_GV_self_gen,
                                f_GV_WZ_no_self_gen=lambda x:
                                    x['spez. GV'] / (x['spez_GV_final']))

    spez_gv['spez. GV'] = df_help_gv.spez_GV_final
    df_f_sv_no_self_gen = df_decom['Strom Netzbezug']
    df_f_gv_no_self_gen = df_help_gv.f_GV_WZ_no_self_gen

    return [spez_sv.sort_index(), spez_gv.sort_index(), vb_wz, bze_je_lk_wz,
            df_f_sv_no_self_gen, df_f_gv_no_self_gen]


def generate_specific_consumption_per_branch_and_district(iterations_power=8,
                                                          iterations_gas=8,
                                                          no_self_gen=False,
                                                          **kwargs):
    """
    Returns specific power and gas consumption per branch and district.
    This function adjusts the specific consumption of all branches from the
    industrial sector.
    One iteration is one adjustment of the results from the function
    generate_specific_consumption_per_branch() regarding the publications
    from the Federal Statistical Office (table_id = 15).

    Parameters
    ----------
    iteration_power: int
        The amount of iterations to generate specific power consumption per
        branch and district, 8 recommended.
    iteration_gas: int
        The amount of iterations to generate specific gas consumption per
        branch and district, 8 recommended.
    no_self_gen : bool, optional, default = False,
        If True: returns specific power consumption without self generation,
                 resulting energy consumption will be lower
    Returns
    ------------
    Tuple that contains two pd.DataFrames
    """
    year = kwargs.get('year', cfg['base_year'])
    [spez_sv, spez_gv, vb_wz, bze_je_lk_wz, df_f_sv_no_self_gen,
     df_f_gv_no_self_gen] = (generate_specific_consumption_per_branch())
    # get latest "Regionalstatistik" from Database
    x = True
    year1 = year
    while(x):
        try:
            vb_LK = database_get('spatial', table_id=15, year=year1)
            x = False
        except ValueError:
            year1 -= 1
    vb_LK.loc[:, 'Verbrauch in MWh'] = vb_LK['value'] / 3.6
    vb_LK.loc[:, 'id_region'] = vb_LK['id_region'].astype(str)
    vb_LK = (vb_LK.assign(ags=[int(x[:-3]) for x in vb_LK['id_region']],
                          ET=[x[0] for x in vb_LK['internal_id']]))
    vb_LK = (vb_LK.loc[((vb_LK['ET'] == 2) | (vb_LK['ET'] == 4))]
                      [['ags', 'Verbrauch in MWh', 'ET']]
                  # HACK: Due to a merge problem in Landkreis-area naming
                  .replace(to_replace=[3152, 3156], value=3159))
    sv_LK_real = (vb_LK
                  .loc[vb_LK['ET'] == 2]
                  .groupby(by=['ags'])[['Verbrauch in MWh']]
                  .sum())
    gv_LK_real = (vb_LK.loc[vb_LK['ET'] == 4].groupby(by=['ags'])
                           [['Verbrauch in MWh']].sum())
    lk_ags = (vb_LK.groupby(by=['ags', 'ET'])[['Verbrauch in MWh']].sum()
                   .reset_index()['ags'].unique())
    # build dataframe with absolute elec and gas demand per district
    spez_gv_lk = pd.DataFrame(index=spez_gv.index, columns=lk_ags)
    spez_sv_lk = pd.DataFrame(index=spez_sv.index, columns=lk_ags)
    for lk in lk_ags:
        spez_gv_lk[lk] = spez_gv['spez. GV']
        spez_sv_lk[lk] = spez_sv['spez. SV']
    sv_lk_wz = bze_je_lk_wz * spez_sv_lk  # absolute electricty demand per dis
    gv_lk_wz = bze_je_lk_wz * spez_gv_lk  # absolute gas demand per district
    # get absolute industrial demands for grouped industry branches as in
    # publication of UGR (Umweltökonomische Gesamtrechnung)
    sv_ind_branches_grp = ['5', '6', '7-9', '10-12', '13-15', '16', '17', '18',
                           '19', '20', '22', '23', '24', '25', '27', '28',
                           '29', '33']
    sv_wz_e_int = (vb_wz.loc[(vb_wz['WZ'].isin(sv_ind_branches_grp)
                              & (vb_wz['ET'] == 18))]
                   .drop(columns=['ET'])
                   .set_index('WZ'))
    gv_ind_branches_grp = ['5', '6', '7-9', '10-12', '13-15', '16', '17',
                           '18', '19', '20', '21', '22', '23', '24', '25',
                           '30']
    gv_wz_e_int = (vb_wz.loc[(vb_wz['WZ'].isin(gv_ind_branches_grp)
                              & (vb_wz['ET'] == 12))]
                   .drop(columns=['ET'])
                   .set_index('WZ'))
    # get energy intensive industrial demand and number of workers per LK
    # energy intensive means a specific consumption >= 10 MWh/worker
    sv_ind_branches = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                       20, 22, 23, 24, 25, 27, 28, 29, 33]
    sv_lk_wz_e_int = sv_lk_wz.loc[sv_ind_branches]
    bze_sv_e_int = bze_je_lk_wz.loc[sv_ind_branches]

    gv_ind_branches = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                       20, 21, 22, 23, 24, 25, 30]
    gv_lk_wz_e_int = gv_lk_wz.loc[gv_ind_branches]
    bze_gv_e_int = bze_je_lk_wz.loc[gv_ind_branches]
    # get industry branches with energy intensity < 10 MWh/worker
    sv_LK_real.loc[:, 'Verbrauch e-arme WZ'] = (sv_lk_wz
                                                .loc[[21, 26, 30, 31, 32]]
                                                .sum())
    sv_LK_real.loc[:, 'Verbrauch e-int WZ'] = (sv_LK_real['Verbrauch in MWh']
                                               - sv_LK_real
                                                   ['Verbrauch e-arme WZ'])
    gv_LK_real.loc[:, 'Verbrauch e-arme WZ'] = (gv_lk_wz
                                                .loc[[26, 27, 28, 31, 32, 33]]
                                                .sum())
    gv_LK_real.loc[:, 'Verbrauch e-int WZ'] = (gv_LK_real['Verbrauch in MWh']
                                               - gv_LK_real
                                                   ['Verbrauch e-arme WZ'])
    # get specific demand per WZ and district
    spez_sv_e_int = spez_sv_lk.loc[sv_ind_branches]
    spez_gv_e_int = spez_gv_lk.loc[gv_ind_branches]
    # start of iterations to adjust regional specific demand of energy
    # energy intensive industries
    ET = [2, 4]
    for et in ET:
        if (et == 2):
            sv_LK = pd.DataFrame(sv_LK_real.loc[:, 'Verbrauch e-int WZ'])
            mean_value = sv_LK['Verbrauch e-int WZ'].sum() / len(sv_LK)
            spez_sv_angepasst = spez_sv_e_int.copy()
            # spez_sv_angepasst.columns = spez_sv_angepasst.columns
            # start loop for adjusting specific power consumption
            while(iterations_power > 0):
                iterations_power -= 1
                y = True
                i = 0
                while(y):
                    # adjust specific demand according to Regionalstatistik
                    i += 1
                    sv_LK.loc[:, 'SV Modell e-int [MWh]'] = (
                        sv_lk_wz_e_int.sum())
                    sv_LK.loc[:, 'Normierter relativer Fehler'] = (
                        (sv_LK['Verbrauch e-int WZ']
                         - sv_LK['SV Modell e-int [MWh]']) / mean_value)
                    sv_LK.loc[:, 'Anpassungsfaktor'] = 1
                    sv_LK.loc[lambda x:
                              abs(x['Normierter relativer Fehler']) > 0.1,
                              'Anpassungsfaktor'] = (
                                  sv_LK['Verbrauch e-int WZ']
                                  / sv_LK['SV Modell e-int [MWh]'])
                    if(sv_LK['Anpassungsfaktor'].sum() == 401):
                        y = False
                    elif(i < 10):
                        spez_sv_angepasst = (spez_sv_angepasst
                                             * sv_LK['Anpassungsfaktor']
                                             .transpose())
                        spez_sv_angepasst[spez_sv_angepasst < 10] = 10
                        spez_sv_angepasst = (spez_sv_angepasst
                                             * sv_LK['Verbrauch e-int WZ']
                                               .sum()
                                             / sv_LK['SV Modell e-int [MWh]']
                                               .sum())
                        sv_lk_wz_e_int = bze_sv_e_int * spez_sv_angepasst
                    else:
                        y = False
                sv_wz = (pd.DataFrame(sv_lk_wz_e_int.sum(axis=1),
                                      columns=['SV WZ Modell [MWh]']))
                k = 0
                z = True
                while(z):
                    # compare adjusted demand to source from UGR
                    # adjust specific demands for energy intensive industries
                    k = k + 1
                    sv_wz_t51 = (pd.DataFrame(index=sv_ind_branches_grp,
                                              columns=['SV WZ Modell [MWh]']))
                    sv_wz_t51.loc[:, 'SV WZ Modell [MWh]'] = 0.0
                    WZe = [7, 8, 9]
                    for i in WZe:
                        sv_wz_t51['SV WZ Modell [MWh]']['7-9'] = (
                            sv_wz_t51['SV WZ Modell [MWh]']['7-9']
                            + sv_wz['SV WZ Modell [MWh]'][i])
                    WZe = [10, 11, 12]
                    for i in WZe:
                        sv_wz_t51['SV WZ Modell [MWh]']['10-12'] = (
                            sv_wz_t51['SV WZ Modell [MWh]']['10-12']
                            + sv_wz['SV WZ Modell [MWh]'][i])
                    WZe = [13, 14, 15]
                    for i in WZe:
                        sv_wz_t51['SV WZ Modell [MWh]']['13-15'] = (
                            sv_wz_t51['SV WZ Modell [MWh]']['13-15']
                            + sv_wz['SV WZ Modell [MWh]'][i])
                    WZe = [5, 6, 16, 17, 18, 19, 20, 22, 23, 24, 25, 27, 28,
                           29, 33]
                    for i in WZe:
                        sv_wz_t51['SV WZ Modell [MWh]'][str(i)] = (
                            sv_wz['SV WZ Modell [MWh]'][i])
                    sv_wz_t51 = (sv_wz_t51.merge(sv_wz_e_int, left_index=True,
                                                 right_index=True))
                    mean_value2 = sv_wz_t51['value'].sum()/len(sv_wz_t51)
                    sv_wz_t51.loc[:, 'Normierter relativer Fehler'] = (
                        (sv_wz_t51['value']
                         - sv_wz_t51['SV WZ Modell [MWh]'])/mean_value2)
                    sv_wz_t51.loc[:, 'Anpassungsfaktor'] = 1
                    sv_wz_t51.loc[lambda x:
                                  abs(x['Normierter relativer Fehler']) > 0.01,
                                  'Anpassungsfaktor'] = (
                                      sv_wz_t51['value']
                                      / sv_wz_t51['SV WZ Modell [MWh]'])
                    sv_wz.loc[:, 'Anpassungsfaktor'] = 0.0
                    for wz in sv_wz.index:
                        if((wz == 7) | (wz == 8) | (wz == 9)):
                            sv_wz['Anpassungsfaktor'][wz] = (
                                sv_wz_t51['Anpassungsfaktor']['7-9'])
                        elif((wz == 10) | (wz == 11) | (wz == 12)):
                            sv_wz['Anpassungsfaktor'][wz] = (
                                sv_wz_t51['Anpassungsfaktor']['10-12'])
                        elif((wz == 13) | (wz == 14) | (wz == 15)):
                            sv_wz['Anpassungsfaktor'][wz] = (
                                sv_wz_t51['Anpassungsfaktor']['13-15'])
                        elif((wz == 31) | (wz == 32)):
                            sv_wz['Anpassungsfaktor'][wz] = (
                                sv_wz_t51['Anpassungsfaktor']['31-32'])
                        else:
                            sv_wz['Anpassungsfaktor'][wz] = (
                                sv_wz_t51['Anpassungsfaktor'][str(wz)])
                    if(sv_wz['Anpassungsfaktor'].sum() == len(sv_wz)):
                        z = False
                    elif(k < 10):
                        spez_sv_angepasst = (spez_sv_angepasst
                                             .multiply(sv_wz
                                                       ['Anpassungsfaktor'],
                                                       axis=0))
                        spez_sv_angepasst[spez_sv_angepasst < 10] = 10
                        sv_lk_wz_e_int = bze_sv_e_int * spez_sv_angepasst
                        sv_wz = pd.DataFrame(sv_lk_wz_e_int.sum(axis=1),
                                             columns=['SV WZ Modell [MWh]'])
                    else:
                        z = False

        elif (et == 4):  # start adjusting loop for gas
            gv_LK = pd.DataFrame(gv_LK_real.loc[:, 'Verbrauch e-int WZ'])
            mean_value = gv_LK['Verbrauch e-int WZ'].sum() / len(gv_LK)
            spez_gv_angepasst = spez_gv_e_int.copy()
            while(iterations_gas > 0):
                iterations_gas -= 1
                y = True
                i = 0
                while(y):
                    i += 1
                    gv_LK.loc[:, 'GV Modell e-int [MWh]'] = (
                        gv_lk_wz_e_int.sum())
                    gv_LK.loc[:, 'Normierter relativer Fehler'] = (
                        (gv_LK['Verbrauch e-int WZ']
                         - gv_LK['GV Modell e-int [MWh]']) / mean_value)
                    gv_LK.loc[:, 'Anpassungsfaktor'] = 1
                    gv_LK.loc[lambda x:
                              abs(x['Normierter relativer Fehler']) > 0.1,
                              'Anpassungsfaktor'] = (
                                  gv_LK['Verbrauch e-int WZ']
                                  / gv_LK['GV Modell e-int [MWh]'])
                    if(gv_LK['Anpassungsfaktor'].sum() == 400):
                        y = False
                    elif(i < 10):
                        spez_gv_angepasst = (spez_gv_angepasst
                                             * gv_LK['Anpassungsfaktor']
                                             .transpose())
                        spez_gv_angepasst[spez_gv_angepasst < 10] = 10
                        spez_gv_angepasst = (spez_gv_angepasst
                                             * gv_LK['Verbrauch e-int WZ']
                                               .sum()
                                             / gv_LK['GV Modell e-int [MWh]']
                                               .sum())
                        gv_lk_wz_e_int = bze_gv_e_int * spez_gv_angepasst
                    else:
                        y = False
                gv_wz = pd.DataFrame(gv_lk_wz_e_int.sum(axis=1),
                                     columns=['GV WZ Modell [MWh]'])
                k = 0
                z = True
                while(z):
                    k = k + 1
                    gv_wz_t51 = (pd.DataFrame(index=gv_ind_branches_grp,
                                              columns=['GV WZ Modell [MWh]']))
                    gv_wz_t51.loc[:, 'GV WZ Modell [MWh]'] = 0.0
                    WZe = [7, 8, 9]
                    for i in WZe:
                        gv_wz_t51['GV WZ Modell [MWh]']['7-9'] = (
                            gv_wz_t51['GV WZ Modell [MWh]']['7-9']
                            + gv_wz['GV WZ Modell [MWh]'][i])
                    WZe = [10, 11, 12]
                    for i in WZe:
                        gv_wz_t51['GV WZ Modell [MWh]']['10-12'] = (
                            gv_wz_t51['GV WZ Modell [MWh]']['10-12']
                            + gv_wz['GV WZ Modell [MWh]'][i])
                    WZe = [13, 14, 15]
                    for i in WZe:
                        gv_wz_t51['GV WZ Modell [MWh]']['13-15'] = (
                            gv_wz_t51['GV WZ Modell [MWh]']['13-15']
                            + gv_wz['GV WZ Modell [MWh]'][i])
                    WZe = [5, 6, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30]
                    for i in WZe:
                        gv_wz_t51['GV WZ Modell [MWh]'][str(i)] = (
                            gv_wz['GV WZ Modell [MWh]'][i])
                    gv_wz_t51 = gv_wz_t51.merge(gv_wz_e_int, left_index=True,
                                                right_index=True)
                    mean_value2 = gv_wz_t51['value'].sum() / len(gv_wz_t51)
                    gv_wz_t51.loc[:, 'Normierter relativer Fehler'] = (
                        (gv_wz_t51['value']
                             - gv_wz_t51['GV WZ Modell [MWh]']) / mean_value2)
                    gv_wz_t51.loc[:, 'Anpassungsfaktor'] = 1
                    gv_wz_t51.loc[lambda x:
                                  abs(x['Normierter relativer Fehler']) > 0.01,
                                  'Anpassungsfaktor'] = (
                                      gv_wz_t51['value']
                                      / gv_wz_t51['GV WZ Modell [MWh]'])
                    gv_wz.loc[:, 'Anpassungsfaktor'] = 0.0
                    for wz in gv_wz.index:
                        if((wz == 7) | (wz == 8) | (wz == 9)):
                            gv_wz['Anpassungsfaktor'][wz] = (
                                gv_wz_t51['Anpassungsfaktor']['7-9'])
                        elif((wz == 10) | (wz == 11) | (wz == 12)):
                            gv_wz['Anpassungsfaktor'][wz] = (
                                gv_wz_t51['Anpassungsfaktor']['10-12'])
                        elif((wz == 13) | (wz == 14) | (wz == 15)):
                            gv_wz['Anpassungsfaktor'][wz] = (
                                gv_wz_t51['Anpassungsfaktor']['13-15'])
                        else:
                            gv_wz['Anpassungsfaktor'][wz] = (
                                gv_wz_t51['Anpassungsfaktor'][str(wz)])
                    if(gv_wz['Anpassungsfaktor'].sum() == len(gv_wz)):
                        z = False
                    elif(k < 10):
                        spez_gv_angepasst = (spez_gv_angepasst
                                             .multiply(
                                                 gv_wz['Anpassungsfaktor'],
                                                 axis=0))
                        spez_gv_angepasst[spez_gv_angepasst < 10] = 10
                        gv_lk_wz_e_int = bze_gv_e_int * spez_gv_angepasst
                        gv_wz = pd.DataFrame(gv_lk_wz_e_int.sum(axis=1),
                                             columns=['GV WZ Modell [MWh]'])
                    else:
                        z = False
    spez_sv_lk.loc[list(spez_sv_angepasst.index)] = spez_sv_angepasst.values
    spez_gv_lk.loc[list(spez_gv_angepasst.index)] = spez_gv_angepasst.values
    #  HACK for Wolfsburg: There is no energy demand available Wolfsburg in the
    #  Regionalstatistik. Therefore, specific demand is set on the average.
    spez_gv_lk[3103] = spez_gv['spez. GV']
    # if no_self_gen==True adjust spezific consumption (both power and gas)
    # by multiplying with no_self_gen-factor determined in function
    # 'generate_specific_consumption_per_branch()'
    if(no_self_gen):
        spez_sv_lk = spez_sv_lk.multiply(df_f_sv_no_self_gen, axis=0)
        spez_gv_lk = spez_gv_lk.multiply(df_f_gv_no_self_gen, axis=0)

    return spez_sv_lk.sort_index(axis=1), spez_gv_lk.sort_index(axis=1)

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
            fn = data_in('regional', cfg['demographic_trend']['filename'])
            df = read_local(fn, year=year)
        else:
            fn = data_in('regional', cfg['population']['filename'])
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

    df = (df.assign(nuts3=lambda x: x.id_region.map(dict_region_code()))
            .set_index('nuts3').sort_index(axis=0))['value']
    df = plausibility_check_nuts3(df)
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
        fn = data_in('regional', cfg['elc_cons_HH_spatial']['filename'])
        df = read_local(fn, year=year)
    elif source == 'database':
        df = database_get('spatial', table_id=table_id, year=year,
                          force_update=force_update)
    else:
        raise KeyError('Wrong source key given in config.yaml - must be either'
                       ' `local` or `database` but is: {}'.format(source))

    df = (df.assign(nuts3=lambda x: x.id_region.map(dict_region_code()))
            .set_index('nuts3').sort_index(axis=0))['value']
    df = plausibility_check_nuts3(df)
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
        fn = data_in('regional', cfg['household_sizes']['filename'])
        df = read_local(fn)
    elif source == 'database':
        df = database_get('spatial', table_id=table_id, year=2011,
                          force_update=force_update)
    else:
        raise KeyError('Wrong source key given in config.yaml - must be either'
                       ' `local` or `database` but is: {}'.format(source))

    df = (df.assign(internal_id=lambda x: x.internal_id.astype(str))
            .assign(nuts3=lambda x: x.id_region.map(dict_region_code()),
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

    df = plausibility_check_nuts3(df)
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
    building_nr_to_type = {-1: 'All',       # All building types
                           1: '1FH',        # 1-family-house
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
                     2018: 'Q_2018',
                     2019: 'R_2019'}

    year = kwargs.get('year', 2018)
    source = kwargs.get('source', cfg['living_space']['source'])
    table_id = kwargs.get('table_id', cfg['living_space']['table_id'])
    force_update = kwargs.get('force_update', False)
    bt = kwargs.get('internal_id', cfg['living_space']['internal_id'])[0]
    vc = kwargs.get('internal_id', cfg['living_space']['internal_id'])[1]
    hs = kwargs.get('internal_id', cfg['living_space']['internal_id'])[2]
    ne = kwargs.get('internal_id', cfg['living_space']['internal_id'])[3]

    if source == 'local':
        df = read_local(data_in('regional', cfg['living_space']['filename']))
    elif source == 'database':
        df = database_get('spatial', table_id=table_id, year=year,
                          force_update=force_update)
    else:
        raise KeyError('Wrong source key given in config.yaml - must be either'
                       ' `local` or `database` but is: {}'.format(source))

    df = (df.assign(nuts3=lambda x: x.id_region.map(dict_region_code()),
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
        df = plausibility_check_nuts3(df)
    else:
        df = df.drop(['id_spatial', 'id_region_type', 'id_region'], axis=1)
    return df


def percentage_EFH_MFH(MFH=False, **kwargs):
    """
    Return either the percentages of single family houses (EFH) or those of
    multi family houses (MFH) for each region.

    Parameters
    ----------
    MFH : bool, default False
        If True: return MFH values
        If False: return EFH values

    Returns
    -------
    pd.Series
    """
    year = kwargs.get('year', 2011)
    source = kwargs.get('source', cfg['percentage_EFH_MFH']['source'])
    table_id = kwargs.get('table_id', cfg['percentage_EFH_MFH']['table_id'])
    force_update = kwargs.get('force_update', False)

    if year != 2011:
        logger.warn('Currently, there is only data for year 2011 available!')

    if source == 'local':
        fn = data_in('regional', cfg['percentage_EFH_MFH']['filename'])
        df = read_local(fn, year=year)
    elif source == 'database':
        df = database_get('spatial', table_id=table_id, year=year,
                          force_update=force_update)
    else:
        raise KeyError('Wrong source key given in config.yaml - must be either'
                       ' `local` or `database` but is: {}'.format(source))

    df = (df.assign(nuts3=lambda x: x.id_region.map(dict_region_code()))
            .set_index('nuts3').sort_index(axis=0))['value']
    df = plausibility_check_nuts3(df)
    if MFH:
        return 1 - df
    else:
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
        df = read_local(data_in('regional', cfg['income']['filename']),
                        internal_id=internal_id, year=year)
    elif source == 'database':
        df = database_get('spatial', table_id=table_id, year=year,
                          internal_id=internal_id, force_update=force_update)
    else:
        raise KeyError('Wrong source key given in config.yaml - must be either'
                       ' `local` or `database` but is: {}'.format(source))

    df = (df.assign(nuts3=lambda x: x.id_region.map(dict_region_code()))
            .set_index('nuts3', drop=True)
            .sort_index())['value']
    df = plausibility_check_nuts3(df)
    return df


def energy_balance_households_power(**kwargs):
    """
    Currently, DE1 and DE2 do not report specific values only for households,
    but aggregated for households+CTS in their energy balances. Therefore, we
    estimate the percentage of households based on the mean of the relation in
    the other regions.

    Returns
    -------
    pd.Series
        index: NUTS-1 codes
    """
    df_HH = energy_balance_values(internal_id=[52, 29], **kwargs)
    df_HH_CTS = energy_balance_values(internal_id=[54, 29], **kwargs)
    factor = df_HH.div(df_HH_CTS).mean()
    df_HH_DE1_DE2 = df_HH_CTS.loc[['DE1', 'DE2']].multiply(factor)
    return pd.concat([df_HH_DE1_DE2, df_HH], axis=0)


def energy_balance_households_gas(**kwargs):
    """
    Currently, DE1 and DE2 do not report specific values only for households,
    but aggregated for households+CTS in their energy balances. Therefore, we
    estimate the percentage of households based on the mean of the relation in
    the other regions.

    Returns
    -------
    pd.Series
        index: NUTS-1 codes
    """
    df_HH = energy_balance_values(internal_id=[52, 21], **kwargs)
    df_HH_CTS = energy_balance_values(internal_id=[54, 21], **kwargs)
    factor = df_HH.div(df_HH_CTS).mean()
    df_HH_DE1_DE2 = df_HH_CTS.loc[['DE1', 'DE2']].multiply(factor)
    return pd.concat([df_HH_DE1_DE2, df_HH], axis=0)


def energy_balance_values(**kwargs):
    """
    Read, transform and return energy balance values in [TWh] per NUTS-1 area.

    Returns
    -------
    pd.Series
        index: NUTS-1 codes
    """
    year = kwargs.get('year', cfg['base_year'])
    source = kwargs.get('source', cfg['energy_balance_values']['source'])
    table_id = kwargs.get('table_id', cfg['energy_balance_values']['table_id'])
    internal_id = kwargs.get('internal_id',
                             cfg['energy_balance_values']['internal_id'])
    force_update = kwargs.get('force_update', False)

    # handle the multiple internal_id's
    if internal_id is not None:
        if not is_real_iterable(internal_id):
            raise TypeError('The passed `internal_id` must be an iterable (a '
                            'list, a dict or a tuple), but is a {}'
                            .format(type(internal_id)))
        if isinstance(internal_id, dict):
            internal_id = list(OrderedDict(internal_id).values())
        assert sum(x <= 0 for x in internal_id) == 0, (
            "There are non-positive values given as internal_id's. Please "
            "pass or set in config.yaml correct line and column id's for the "
            "energy balances. An explanation can be found in the "
            "`internal_id_description` column in data.database_description().")

    if source == 'local':
        df = read_local(data_in('regional',
                                cfg['energy_balance_values']['filename']),
                        internal_id=internal_id, year=year)
    elif source == 'database':
        df = database_get('spatial', table_id=table_id, year=year,
                          internal_id=internal_id, force_update=force_update)
    else:
        raise KeyError('Wrong source key given in config.yaml - must be either'
                       ' `local` or `database` but is: {}'.format(source))

    df = (df.assign(nuts1=lambda x: x.id_region.map(
                    dict_region_code(values='natcode_nuts1', level='bl')))
            .set_index('nuts1', drop=True)
            .sort_index())['value'] / 3600  # convert TJ -> TWh
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
        df = (pd.read_csv(data_in('regional',
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
        df = (pd.read_csv(data_in('regional',
                                   cfg['hotwater_shares']['filename']),
                          index_col='natcode_nuts1', encoding='utf-8')
                .drop(labels='name_nuts1', axis=1))
    elif source == 'database':
        raise NotImplementedError('Not here yet!')
    else:
        raise KeyError('Wrong source key given in config.yaml!')
    return df


def heat_demand_buildings(**kwargs):
    """
    Read, transform and return heat_demand in [kWh/(m2a)] per NUTS-3 area.

    Returns
    -------
    pd.Series
        index: NUTS-3 codes
    """
    bt_to_type = {1: '1FH',  # 1-family-house
                  2: 'TH',   # Terraced house
                  3: 'MFH',  # multi-family-house (3-6)
                  4: 'BB'}   # building block
    hp_to_name = {1: 'net heat demand',
                  2: 'hot water (final energy)',
                  3: 'space heating',
                  4: 'hot water (generation energy)',
                  5: 'fossil fuels',
                  6: 'wood/biomass',
                  7: 'electricity (incl. side energy)',
                  8: 'electricity generation',
                  9: 'primary energy consumption (total)',
                  10: 'primary energy consumption (non-renewable)'}
    vc_to_yearstr = {1: 'A_<1859',
                     2: 'B_1860-1918',
                     3: 'C_1919-1948',
                     4: 'D_1949-1957',
                     5: 'E_1958-1968',
                     6: 'F_1969-1978',
                     7: 'G_1979-1983',
                     8: 'H_1984-1994',
                     9: 'I_1995-2001',
                     10: 'J_2002-2009'}
    va_to_variant = {1: 'Status-Quo',
                     2: 'Modernisation conventional',
                     3: 'Modernisation future'}

    year = kwargs.get('year', 2014)
    source = kwargs.get('source', cfg['heat_dem_bld']['source'])
    table_id = kwargs.get('table_id', cfg['heat_dem_bld']['table_id'])
    force_update = kwargs.get('force_update', False)
    bt = kwargs.get('internal_id', cfg['heat_dem_bld']['internal_id'])[0]
    vc = kwargs.get('internal_id', cfg['heat_dem_bld']['internal_id'])[1]
    hp = kwargs.get('internal_id', cfg['heat_dem_bld']['internal_id'])[2]
    va = kwargs.get('internal_id', cfg['heat_dem_bld']['internal_id'])[3]

    if source == 'local':
        raise NotImplementedError('Not here yet!')
    elif source == 'database':
        df = database_get('spatial', table_id=table_id, year=year,
                          force_update=force_update)
    else:
        raise KeyError('Wrong source key given in config.yaml - must be either'
                       ' `local` or `database` but is: {}'.format(source))

    df = (df.assign(nuts3=lambda x: x.id_region.map(dict_region_code()),
                    building_type=lambda x: x.internal_id.str[0],
                    vintage_class=lambda x: x.internal_id.str[1],
                    heat_parameter=lambda x: x.internal_id.str[2],
                    variant=lambda x: x.internal_id.str[3])
          .drop(columns=['year', 'internal_id', 'id_spatial', 'id_region_type',
                         'id_region'])
          .dropna(subset=['nuts3'])
          .loc[lambda x: ~(x.nuts3.isin(['DE915', 'DE919']))])
    # Filter by possibly given internal_id
    if bt is not None and 1 <= bt <= 4:
        df = df.loc[lambda x: x.building_type == bt]
    if vc is not None and (1 <= vc <= 10):
        df = df.loc[lambda x: x.vintage_class == vc]
    if hp is not None and (1 <= hp <= 10):
        df = df.loc[lambda x: x.heat_parameter == hp]
    if va is not None and (1 <= va <= 3):
        df = df.loc[lambda x: x.variant == va]
    # Replace internal_ids by human readables
    df = (df.replace(dict(building_type=bt_to_type))
            .replace(dict(heat_parameter=hp_to_name))
            .replace(dict(vintage_class=vc_to_yearstr))
            .replace(dict(variant=va_to_variant)))

#    df = plausibility_check_nuts3(df, check_zero=False)
    return df


def efficiency_enhancement(source, **kwargs):
    """
    Read and return the efficienicy enhancement for power or gas consumption
    per branch.

    Parameters
    ----------
    source : str
        must be one of ['power', 'gas']
    Returns
    -------
    pd.Series
        index: Branches
    """
    year = kwargs.get('year', cfg['base_year'])
    es_rate = (pd.read_excel(data_in('temporal',
                                     'Efficiency_Enhancement_Rates.xlsx'))
                 .set_index('WZ'))
    df = pow((-es_rate + 1), (year - 2015))
    if source == 'power':
        df = df['Effizienzsteigerungsrate Strom']
    elif source == 'gas':
        df = df['Effizienzsteigerungsrate Gas']
    else:
        raise ValueError("`source` must be in ['power', 'gas']")
    return df


def employees_per_branch_district(**kwargs):
    """
    Read, transform and return the number of employees per NUTS-3 area
    and branch.
    The variable 'scenario' is used only as of 2019!

    Returns
    -------
    pd.Dataframe
        index: Branches
        columns: District keys (Landkreisschlüssel)
    """

    year = kwargs.get('year', cfg['base_year'])
    scenario = kwargs.get('scenario', cfg['scenario'])

    if year in range(2000, 2008):
        df = database_get('spatial', table_id=18, year=2008)
        df = (df.assign(ags=[int(x[:-3]) for x in
                             df['id_region'].astype(str)],
                        WZ=[x[1] for x in df['internal_id']]))
        bool_list = np.array(df['id_region'].astype(str))
        for i in range(0, len(df)):
            bool_list[i] = (df['internal_id'][i][0] == 9)
        df = (df[((bool_list) & (df['WZ'] > 0))][['ags', 'value', 'WZ']]
              .rename(columns={'value': 'BZE'}))
        df = (pd.pivot_table(df, values='BZE', index='WZ',
                             columns='ags', fill_value=0, dropna=False))
        print("number of employees was taken from 2008, as there is no earlier\
               data available")
    elif year in range(2008, 2019):
        df = database_get('spatial', table_id=18, year=year)
        df = (df.assign(ags=[int(x[:-3]) for x in
                             df['id_region'].astype(str)],
                        WZ=[x[1] for x in df['internal_id']]))
        bool_list = np.array(df['id_region'].astype(str))
        for i in range(0, len(df)):
            bool_list[i] = (df['internal_id'][i][0] == 9)
        df = (df[((bool_list) & (df['WZ'] > 0))][['ags', 'value', 'WZ']]
              .rename(columns={'value': 'BZE'}))
        df = (pd.pivot_table(df, values='BZE', index='WZ',
                             columns='ags', fill_value=0, dropna=False))
    elif year in range(2019, 2036):
        if scenario == 'Basis':
            df = database_get('spatial', table_id=27, year=year)
        elif scenario == 'Digital':
            df = database_get('spatial', table_id=28, year=year)
        else:
            raise ValueError("`scenario` must be in ['Basis', 'Digital']")

        df = (df.assign(ags=[int(x[:-3]) for x in
                             df['id_region'].astype(str)],
                        WZ=[x[0] for x in df['internal_id']]))
        df = (pd.pivot_table(df, values='value', index='WZ',
                             columns='ags', fill_value=0, dropna=False))
    else:
        raise ValueError("`year` must be between 2000 and 2035")

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


def standard_load_profile_elc(which='H0', freq='1H', **kwargs):
    """
    Return the electric standard load profile H0 in normalized units
    ('normalized' means here that the sum over all time steps equals one).
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


def standard_load_profile_gas(which='H0', typ='EFH', normalized=True,
                              **kwargs):
    """
    Return the gas standard load profile H0 in normalized units
    ('normalized' means here that the sum over all time steps equals one).
    """
    if which != 'H0':
        raise NotImplementedError('Not here yet!')

    if typ == 'EFH':
        kwargs['internal_id'] = 0
    elif typ == 'MFH':
        kwargs['internal_id'] = 1
    else:
        raise ValueError("`typ` must be in ['EFH', 'MFH']")

    df_slp = reshape_spatiotemporal(key='slp_H0_1H_gas', **kwargs)
    if normalized:
        df_slp /= df_slp.sum(axis=0)

    return df_slp


def zve_load_profile_elc(region='AllRegions', year=2015, **kwargs):
    """
    Return the ZVE load profile, which was generated in
        temporal.make_zve_load_profiles()

    Since the underlying input data cannot be made public, not everyone can run
    this on their own. Therefore we publish the result for all regions in 2015.
    """
    fn = data_out('ZVE_timeseries_{}_{}.csv'.format(region, year))
    return pd.read_csv(fn, index_col=0, encoding='utf-8', parse_dates=True,
                       infer_datetime_format=True, engine='c')


def shift_load_profile_generator(state, low=0.35, **kwargs):
    """
    Return shift load profiles in normalized units
    ('normalized' means that the sum over all time steps equals to one).

    Parameters
    ----------
    state : str
        Must be one of ['BW','BY','BE','BB','HB','HH','HE','MV',
                        'NI','NW','RP','SL','SN','ST','SH','TH']
    low : float
        Load level during "low" loads. Industry loads have two levels:
            "low" outside of working hours and "high" during working hours.
        Default is set to 0.35 for low, which was deduced from real load data.

    Returns
    -------
    pd.DataFrame
    """
    year = kwargs.get('year', cfg['base_year'])
    validity_check_nuts1(state)
    idx = pd.date_range(start=str(year), end=str(year+1), freq='15T')[:-1]
    df = (pd.DataFrame(data={'Date': idx})
            .assign(Day=lambda x: pd.DatetimeIndex(x['Date']).date)
            .assign(Hour=lambda x: pd.DatetimeIndex(x['Date']).time)
            .assign(DayOfYear=lambda x:
                    pd.DatetimeIndex(x['Date']).dayofyear.astype(int)))
    periods = len(df)
    mask_holiday = []
    for i in range(0, len(holidays.DE(state=state, years=year))):
        mask_holiday.append('Null')
        mask_holiday[i] = ((df['Day'] == [x for x in holidays.DE(state=state,
                            years=year).items()][i][0]))
    hd = mask_holiday[0]
    for i in range(1, len(holidays.DE(state=state, years=year))):
        hd = hd | mask_holiday[i]
    df['WT'] = df['Date'].apply(lambda x: x.weekday() < 5)
    df['WT'] = df['WT'] & (~hd)
    df['SA'] = df['Date'].apply(lambda x: x.weekday() == 5)
    df['SA'] = df['SA'] & (~hd)
    df['SO'] = df['Date'].apply(lambda x: x.weekday() == 6)
    df['SO'] = df['SO'] | hd
    # 24th and 31st of december are treated like a saturday
    hld = [datetime.date(year, 12, 24), datetime.date(year, 12, 31)]
    mask = df['Day'].isin(hld)
    df.loc[mask, ['WT', 'SO']] = False
    df.loc[mask, 'SA'] = True
    for sp in ['S1_WT', 'S1_WT_SA', 'S1_WT_SA_SO', 'S2_WT',
               'S2_WT_SA', 'S2_WT_SA_SO', 'S3_WT', 'S3_WT_SA',
               'S3_WT_SA_SO']:
        if(sp == 'S1_WT'):
            anzahl_wz = 17 / 48 * len(df[df['WT']])
            anzahl_nwz = (31 / 48 * len(df[df['WT']]) + len(df[df['SO']])
                          + len(df[df['SA']]))
            anteil = 1 / (anzahl_wz + low * anzahl_nwz)
            df[sp] = anteil
            mask = (df['SO'] | df['SA'])
            df.loc[mask, sp] = low * anteil
            mask = ((df['WT'])
                    & ((df['Hour'] < pd.to_datetime('08:00:00').time())
                    | (df['Hour'] >= pd.to_datetime('16:30:00').time())))
            df.loc[mask, sp] = low * anteil
        elif(sp == 'S1_WT_SA'):
            anzahl_wz = (17 / 48 * len(df[df['WT']])
                         + 17 / 48 * len(df[df['SA']]))
            anzahl_nwz = (31 / 48 * len(df[df['WT']]) + len(df[df['SO']])
                          + 31/48 * len(df[df['SA']]))
            anteil = 1 / (anzahl_wz + low * anzahl_nwz)
            df[sp] = anteil
            mask = df['SO']
            df.loc[mask, sp] = low * anteil
            mask = ((df['WT']) & ((df['Hour'] < pd.to_datetime('08:00:00')
                    .time()) | (df['Hour'] >= pd.to_datetime('16:30:00')
                                .time())))
            df.loc[mask, sp] = low * anteil
            mask = ((df['SA']) & ((df['Hour'] < pd.to_datetime('08:00:00')
                    .time()) | (df['Hour'] >= pd.to_datetime('16:30:00')
                                .time())))
            df.loc[mask, sp] = low * anteil
        elif(sp == 'S1_WT_SA_SO'):
            anzahl_wz = (17 / 48 * (len(df[df['WT']]) + len(df[df['SO']])
                         + len(df[df['SA']])))
            anzahl_nwz = (31 / 48 * (len(df[df['WT']]) + len(df[df['SO']])
                          + len(df[df['SA']])))
            anteil = 1 / (anzahl_wz + low * anzahl_nwz)
            df[sp] = anteil
            mask = ((df['Hour'] < pd.to_datetime('08:00:00').time())
                    | (df['Hour'] >= pd.to_datetime('16:30:00').time()))
            df.loc[mask, sp] = low * anteil
        elif(sp == 'S2_WT'):
            anzahl_wz = 17/24 * len(df[df['WT']])
            anzahl_nwz = (7/24 * len(df[df['WT']]) + len(df[df['SO']])
                          + len(df[df['SA']]))
            anteil = 1 / (anzahl_wz + low * anzahl_nwz)
            df[sp] = anteil
            mask = (df['SO'] | df['SA'])
            df.loc[mask, sp] = low * anteil
            mask = ((df['WT'])
                    & ((df['Hour'] < pd.to_datetime('06:00:00').time())
                    | (df['Hour'] >= pd.to_datetime('23:00:00').time())))
            df.loc[mask, sp] = low * anteil
        elif(sp == 'S2_WT_SA'):
            anzahl_wz = 17/24 * (len(df[df['WT']]) + len(df[df['SA']]))
            anzahl_nwz = (7/24 * len(df[df['WT']]) + len(df[df['SO']])
                          + 7/24 * len(df[df['SA']]))
            anteil = 1 / (anzahl_wz + low * anzahl_nwz)
            df[sp] = anteil
            mask = df['SO']
            df.loc[mask, sp] = low * anteil
            mask = (((df['WT']) | (df['SA']))
                    & ((df['Hour'] < pd.to_datetime('06:00:00').time())
                    | (df['Hour'] >= pd.to_datetime('23:00:00').time())))
            df.loc[mask, sp] = low * anteil
        elif(sp == 'S2_WT_SA_SO'):
            anzahl_wz = (17/24 * (len(df[df['WT']]) + len(df[df['SA']])
                                  + len(df[df['SO']])))
            anzahl_nwz = (7/24 * (len(df[df['WT']]) + len(df[df['SO']])
                                  + len(df[df['SA']])))
            anteil = 1 / (anzahl_wz + low * anzahl_nwz)
            df[sp] = anteil
            mask = (((df['Hour'] < pd.to_datetime('06:00:00').time())
                    | (df['Hour'] >= pd.to_datetime('23:00:00').time())))
            df.loc[mask, sp] = low * anteil
        elif(sp == 'S3_WT_SA_SO'):
            anteil = 1 / periods
            df[sp] = anteil
        elif(sp == 'S3_WT'):
            anzahl_wz = len(df[df['WT']])
            anzahl_nwz = len(df[df['SO']]) + len(df[df['SA']])
            anteil = 1 / (anzahl_wz + low * anzahl_nwz)
            df[sp] = anteil
            mask = (df['SO'] | df['SA'])
            df.loc[mask, sp] = low * anteil
        elif(sp == 'S3_WT_SA'):
            anzahl_wz = len(df[df['WT']]) + len(df[df['SA']])
            anzahl_nwz = len(df[df['SO']])
            anteil = 1 / (anzahl_wz + low * anzahl_nwz)
            df[sp] = anteil
            mask = df['SO']
            df.loc[mask, sp] = low * anteil
    df = (df[['Date', 'S1_WT', 'S1_WT_SA', 'S1_WT_SA_SO', 'S2_WT', 'S2_WT_SA',
             'S2_WT_SA_SO', 'S3_WT', 'S3_WT_SA', 'S3_WT_SA_SO']]
          .set_index('Date'))
    return df


def gas_slp_weekday_params(state, **kwargs):
    """
    Return the weekday-parameters of the gas standard load profiles

    Parameters
    ----------
    state: str
        must be one of ['BW', 'BY', 'BE', 'BB', 'HB', 'HH', 'HE', 'MV',
                        'NI', 'NW', 'RP', 'SL', 'SN',' ST', 'SH', 'TH']
    Returns
    -------
    pd.DataFrame
    """
    year = kwargs.get('year', cfg['base_year'])
    validity_check_nuts1(state)

    idx = pd.date_range(start=str(year), end=str(year+1), freq='d')[:-1]
    df = (pd.DataFrame(data={'Date': idx})
            .assign(Day=lambda x: pd.DatetimeIndex(x['Date']).date)
            .assign(DayOfYear=lambda x:
                    pd.DatetimeIndex(x['Date']).dayofyear.astype(int)))

    mask_holiday = []
    for i in range(0, len(holidays.DE(state=state, years=year))):
        mask_holiday.append('Null')
        mask_holiday[i] = ((df['Day'] == [x for x in holidays.DE(state=state,
                                          years=year).items()][i][0]))
    hd = mask_holiday[0]
    for i in range(1, len(holidays.DE(state=state, years=year))):
        hd = hd | mask_holiday[i]
    df['MO'] = df['Date'].apply(lambda x: x.weekday() == 0)
    df['MO'] = df['MO'] & (~hd)
    df['DI'] = df['Date'].apply(lambda x: x.weekday() == 1)
    df['DI'] = df['DI'] & (~hd)
    df['MI'] = df['Date'].apply(lambda x: x.weekday() == 2)
    df['MI'] = df['MI'] & (~hd)
    df['DO'] = df['Date'].apply(lambda x: x.weekday() == 3)
    df['DO'] = df['DO'] & (~hd)
    df['FR'] = df['Date'].apply(lambda x: x.weekday() == 4)
    df['FR'] = df['FR'] & (~hd)
    df['SA'] = df['Date'].apply(lambda x: x.weekday() == 5)
    df['SA'] = df['SA'] & (~hd)
    df['SO'] = df['Date'].apply(lambda x: x.weekday() == 6)
    df['SO'] = df['SO'] | hd
    hld = [(datetime.date(int(year), 12, 24)),
           (datetime.date(int(year), 12, 31))]
    mask = df['Day'].isin(hld)
    df.loc[mask, ['MO', 'DI', 'MI', 'DO', 'FR', 'SO']] = False
    df.loc[mask, 'SA'] = True
    par = pd.DataFrame.from_dict(gas_load_profile_parameters_dict())
    for slp in par.index:
        df['FW_'+str(slp)] = 0
        for wd in ['MO', 'DI', 'MI', 'DO', 'FR', 'SA', 'SO']:
            df.loc[df[wd], ['FW_'+str(slp)]] = par.loc[slp, wd]
    return df.drop(columns=['DayOfYear']).set_index('Day')


def CTS_power_slp_generator(state, **kwargs):
    """
    Return the electric standard load profiles in normalized units
    ('normalized' means here that the sum over all time steps equals one).

    Parameters
    ----------
    state: str
        must be one of ['BW', 'BY', 'BE', 'BB', 'HB', 'HH', 'HE', 'MV',
                        'NI', 'NW', 'RP', 'SL', 'SN',' ST', 'SH', 'TH']

    Returns
    -------
    pd.DataFrame
    """
    def Leistung(Tag_Zeit, mask, df, df_SLP):
        u = pd.merge(df[mask], df_SLP[['Hour', Tag_Zeit]], on=['Hour'],
                     how='left')
        v = pd.merge(df, u[['Date', Tag_Zeit]], on=['Date'], how='left')
        return v.fillna(0)[Tag_Zeit]

    year = kwargs.get('year', cfg['base_year'])
    validity_check_nuts1(state)
    idx = pd.date_range(start=str(year), end=str(year+1), freq='15T')[:-1]
    df = (pd.DataFrame(data={'Date': idx})
            .assign(Day=lambda x: pd.DatetimeIndex(x['Date']).date)
            .assign(Hour=lambda x: pd.DatetimeIndex(x['Date']).time)
            .assign(DayOfYear=lambda x:
                    pd.DatetimeIndex(x['Date']).dayofyear.astype(int)))
    mask_holidays = []
    for i in range(0, len(holidays.DE(state=state, years=year))):
        mask_holidays.append('Null')
        mask_holidays[i] = ((df['Day'] == [x for x in holidays.DE(state=state,
                                           years=year).items()][i][0]))
    hd = mask_holidays[0]
    for i in range(1, len(holidays.DE(state=state, years=year))):
        hd = hd | mask_holidays[i]
    df['WD'] = df['Date'].apply(lambda x: x.weekday() < 5) & (~hd)
    df['SA'] = df['Date'].apply(lambda x: x.weekday() == 5) & (~hd)
    df['SU'] = df['Date'].apply(lambda x: x.weekday() == 6) | hd
    mask = df['Day'].isin([datetime.date(year, 12, 24),
                           datetime.date(year, 12, 31)])
    df.loc[mask, ['WD', 'SU']] = False
    df.loc[mask, 'SA'] = True
    wiz1 = df.loc[df['Date'] < (str(year) + '-03-21 00:00:00')]
    wiz2 = df.loc[df['Date'] >= (str(year) + '-11-01')]
    soz = (df.loc[((str(year) + '-05-15') <= df['Date'])
                   & (df['Date'] < (str(year) + '-09-15'))])
    uez1 = (df.loc[((str(year) + '-03-21') <= df['Date'])
                       & (df['Date'] < (str(year) + '-05-15'))])
    uez2 = (df.loc[((str(year) + '-09-15') <= df['Date'])
                        & (df['Date'] <= (str(year) + '-10-31'))])
    df = df.assign(WIZ=lambda x: (x.Day.isin(wiz1.Day) | x.Day.isin(wiz2.Day)),
                   SOZ=lambda x: x.Day.isin(soz.Day),
                   UEZ=lambda x: (x.Day.isin(uez1.Day) | x.Day.isin(uez2.Day)))

    last_strings = []
    for profile in ['H0', 'L0', 'L1', 'L2', 'G0', 'G1', 'G2', 'G3', 'G4',
                    'G5', 'G6']:
        f = '39_VDEW_Strom_Repräsentative Profile_{}.xlsx'.format(profile)
        df_load = pd.read_excel(data_in('temporal', 'Power Load Profiles', f),
                                sep=';', decimal=',')
        df_load.columns = ['Hour', 'SA_WIZ', 'SU_WIZ', 'WD_WIZ', 'SA_SOZ',
                           'SU_SOZ', 'WD_SOZ', 'SA_UEZ', 'SU_UEZ', 'WD_UEZ']
        df_load.loc[1] = df_load.loc[len(df_load) - 2]
        df_SLP = df_load[1:97]
        df_SLP = df_SLP.reset_index()[['Hour', 'SA_WIZ', 'SU_WIZ', 'WD_WIZ',
                                       'SA_SOZ', 'SU_SOZ', 'WD_SOZ', 'SA_UEZ',
                                       'SU_UEZ', 'WD_UEZ']]
        wd_wiz = Leistung('WD_WIZ', (df.WD & df.WIZ), df, df_SLP)
        wd_soz = Leistung('WD_SOZ', (df.WD & df.SOZ), df, df_SLP)
        wd_uez = Leistung('WD_UEZ', (df.WD & df.UEZ), df, df_SLP)
        sa_wiz = Leistung('SA_WIZ', (df.SA & df.WIZ), df, df_SLP)
        sa_soz = Leistung('SA_SOZ', (df.SA & df.SOZ), df, df_SLP)
        sa_uez = Leistung('SA_UEZ', (df.SA & df.UEZ), df, df_SLP)
        su_wiz = Leistung('SU_WIZ', (df.SU & df.WIZ), df, df_SLP)
        su_soz = Leistung('SU_SOZ', (df.SU & df.SOZ), df, df_SLP)
        su_uez = Leistung('SU_UEZ', (df.SU & df.UEZ), df, df_SLP)
        Summe = (wd_wiz + wd_soz + wd_uez + sa_wiz + sa_soz + sa_uez
                 + su_wiz + su_soz + su_uez)
        Last = 'Last_' + str(profile)
        last_strings.append(Last)
        df[Last] = Summe
        total = sum(df[Last])
        df_normiert = df[Last] / total
        df[profile] = df_normiert

    return df.drop(columns=last_strings).set_index('Date')


def ambient_T(**kwargs):
    """
    Return the ambient temperature in [°C] per NUTS-3-region and time step.
    """
    kwargs['check_zero'] = False
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
        index:      time step
        columns:    NUTS-3 codes
    """
    year = kwargs.get('year', cfg['base_year'])
    source = kwargs.get('source', cfg[key]['source'])
    table_id = kwargs.get('table_id', cfg[key]['table_id'])
    internal_id = kwargs.get('internal_id', cfg.get(key).get('internal_id'))
    force_update = kwargs.get('force_update', False)
    check_zero = kwargs.get('check_zero', True)

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
              .assign(nuts3=lambda x: x.id_region.map(dict_region_code()))
              .loc[lambda x: (~(x.nuts3.isna()))]
              .set_index('nuts3').sort_index(axis=0)
              .loc[:, 'values']
              .apply(literal_converter))

        df_exp = (pd.DataFrame(df.values.tolist(), index=df.index)
                    .astype(float))
    else:
        raise KeyError('Wrong source key given in config.yaml - must be either'
                       ' `local` or `database` but is: {}'.format(source))
    df_exp = plausibility_check_nuts3(df_exp, check_zero=check_zero)
    return df_exp.pipe(transpose_spatiotemporal, year=year, freq=freq)


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
                 allow_zero_negative=None, force_update=False, **kwargs):
    """
    Get data from the demandregio database.

    Parameters
    ----------
    dimension : str
        Either 'spatial' or 'temporal'.
    table_id : int or str, optional
        ID of table to be queried.
    internal_id : int or str or list, optional
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
            if not all([isinstance(x, int) for x in internal_id]):
                raise ValueError("If `internal_id` is passed as a list, all "
                                 "items must be integers. Passing wildcards "
                                 "e.g. (*) or (?) is currently not possible!")
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
                             crs={'init': 'epsg:25832'}, geometry=geom)
               .assign(nuts3=lambda x: x.id_ags.map(dict_region_code()))
               .set_index('nuts3').sort_index(axis=0))


def validity_check_nuts1(state):
    """
    Check if given NUTS-1 code is valid.

    Parameters
    ----------
    state : str

    """
    if state not in ['BW', 'BY', 'BE', 'BB', 'HB', 'HH', 'HE', 'MV',
                     'NI', 'NW', 'RP', 'SL', 'SN', 'ST', 'SH', 'TH']:
        raise ValueError('Given NUTS-1 code `{}` is not valid!'.format(state))


def plausibility_check_nuts3(df, check_zero=True):
    """
    Check a given pd.DataFrame
    - if all nuts3 regions are available and
    - if all contained values are greater zero.

    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        Holding the values (required index: NUTS-3 codes)
    """
    # 1. Check if there are unknown regions
    A_db = set(dict_region_code().values())
    B_check = set(df.index)
    C_diff = B_check - A_db
    if len(C_diff) > 0:
        logger.warn('The nuts3-codes of the checked DataFrame are not '
                    'congruent with those in the database. These here are not '
                    'in the database: {}'.format(C_diff))
    # 2. Handle possible NUTS-2013 regions:
    nuts_2013 = ['DE915', 'DE919']
    if (df.index.isin(['DE91C']).any() and df.index.isin(nuts_2013).any()):
        logger.info('Dropping old NUTS-v2013 regions.')
        df = df[~(df.index.isin(nuts_2013))]
    elif (not df.index.isin(['DE91C']).any()
          and df.index.isin(nuts_2013).any()):
        logger.info('Merging old Göttingen+Osterode to new NUTS-v2016 region.')
        df.loc['DE91C'] = df.loc[nuts_2013].sum()
        df = df[~(df.index.isin(nuts_2013))]
    # 3. Check if values below zero
    if isinstance(df, pd.Series):
        if check_zero and df.loc[lambda x: x <= 0.0].any():
            logger.warn('There are values less or equal to zero.')
    elif isinstance(df, pd.DataFrame):
        if check_zero and df[df <= 0.0].any().any():
            logger.warn('There are values less or equal to zero.')
    else:
        raise NotImplementedError('Check for given type! Other than pd.Series '
                                  'or pd.DataFrame are not yet possible.')
    return df


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
        if is_real_iterable(internal_id):
            for i, int_id in enumerate(internal_id):
                df = df.loc[lambda x: x.internal_id.str[i] == int_id]
        else:
            df = df.loc[lambda x: x.internal_id == internal_id]
    return df


def append_region_name(df):
    """
    Append the region name as additional column to a DataFrame or Series.

    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        required index: NUTS-1 or NUTS-3 codes

    Returns
    -------
    pd.DataFrame
        with additional column
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if (df.index.str.len().max() == 3 or df.index.name == 'nuts1'):
        level, keys = ['bl', 'natcode_nuts1']
    else:
        level, keys = ['lk', 'natcode_nuts3']
    return df.assign(region_name=lambda x:
                     x.index.map(dict_region_code(keys=keys, values='name',
                                                  level=level)))


def transpose_spatiotemporal(df, freq='1H', **kwargs):
    """
    Transpose a spatiotemporal pd.DataFrame and set/reset the pd.DateTimeIndex.

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


def is_real_iterable(obj):
    """
    Check if passed `obj` of any type is a real iterable.

    Returns
    -------
    bool
    """
    if (isinstance(obj, Iterable) and not isinstance(obj, str)):
        return True
    else:
        return False
