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
from .config import (get_config, _data_in, database_raw, region_id_to_nuts3,
                     literal_converter, wz_dict, hist_weather_year, bl_dict,
                     slp_branch_cts_gas, slp_branch_cts_power,
                     shift_profile_industry, gas_load_profile_parameters_dict)
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

def t_allo(**kwargs):
    """
    Returns allocation temperature from weather data for (historical) year
    
    Returns
    -------
    pd.DataFrame
    """
    year = kwargs.get('year', cfg['base_year'])
    hist_year = hist_weather_year().get(year)
    dic_nuts3 = (region_id_to_nuts3(raw = True)[['natcode_nuts3','ags_lk']]
                    .set_index('natcode_nuts3'))
    dic_nuts3['ags_lk'] = dic_nuts3['ags_lk'].astype(str).str.zfill(5)
    if ((hist_year % 4 == 0) & (hist_year % 100 != 0) | (hist_year % 4 == 0) 
        & (hist_year % 100 == 0) & (hist_year % 400 == 0)):
        periods = 35136
    else:
        periods = 35040       
    df = ambient_T(year = hist_year, internal_id = 2)
    df = (df.assign(date = pd.date_range((str(hist_year) + '-01-01'), 
                    periods = periods / 4 , freq = 'H',
                    tz = 'Europe/Berlin'))
            .set_index('date').resample('D').mean())
    df = (pd.merge(df.transpose(), dic_nuts3, how = 'right',
                      left_index = True, right_index = True)
                     .set_index('ags_lk').transpose())
    df['03159'] = (df['03152'] + df['03156']) / 2
    df.drop(columns = ['03152','03156'], inplace = True)
    df.columns = df.columns.astype(int).astype(str)    
    for district in df.columns:
        te = df[district].values
        for i in range(len(te)-1, -1, -1):
            if (i >= 3):
                te[i] = ((te[i] + 0.5 * te[i - 1] + 0.25 * te[i - 2] + 
                          te[i - 3] * 0.125) / 1.875)
        df[district] = te 
    return df

def h_value(slp, districts):
    """
    Returns h-values depending on allocation temperature  for every district
    
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
    temp_df = t_allo()
    par = pd.DataFrame.from_dict(gas_load_profile_parameters_dict())
    df = temp_df[[x for x in districts]]
    par_slp = par.loc[(par.index == slp)].reset_index()
    A = par_slp['A'][0]
    B = par_slp['B'][0]
    C = par_slp['C'][0]
    D = par_slp['D'][0]
    mH = par_slp['mH'][0]
    bH = par_slp['bH'][0]
    mW = par_slp['mW'][0]
    bW = par_slp['bW'][0]
    for landkreis in districts:        
        te = temp_df[landkreis].values
        for i in range(len(te)):
            df[landkreis][i] = ((A / (1 + pow(B / (te[i] - 40), C)) + D) + 
                                 max(mH * te[i] + bH, mW * te[i] + bW))
        summe = df[landkreis].sum()
        df[landkreis] = df[landkreis] / summe
    return df

def generate_specific_consumption_per_branch():
    """
    Returns specific power and gas consumption per branch. Also returns total 
    power and gas consumption per branch and also the amount of workers per 
    branch and district.
    
    Returns
    ------------
    Tuple that contains four pd.DataFrames
    """
    vb_wz = database_get('spatial', table_id = 38, year = 2015)
    vb_wz = (vb_wz.assign(WZ = [x[0] for x in vb_wz['internal_id']],
                          ET = [x[1] for x in vb_wz['internal_id']]))              
    vb_wz = (vb_wz[(vb_wz['ET'] == 12) | (vb_wz['ET'] == 18) & 
            (vb_wz['WZ'].isin(list(wz_dict().keys())))]
            [['value', 'WZ', 'ET']].replace({'WZ': wz_dict()}))
    vb_wz['value'] = vb_wz['value'] * 1000 / 3.6 
    sv_wz_real = (vb_wz.loc[vb_wz['ET'] == 18][['WZ', 'value']]
                       .groupby(by = 'WZ')[['value']].sum()
                       .rename(columns = {'value': 'SV WZ [MWh]'}))
    gv_wz_real = (vb_wz.loc[vb_wz['ET'] == 12][['WZ', 'value']]
                       .groupby(by = 'WZ')[['value']].sum()
                       .rename(columns = {'value': 'GV WZ [MWh]'}))
    df = database_get('spatial', table_id = 18, year = 2015)
    df = (df.assign(ags = [int(x[:-3]) for x in 
                            df['id_region'].astype(str)],
                    WZ= [x[1] for x in df['internal_id']]))
    bool_list = np.array(df['id_region'].astype(str))
    for i in range(0, len(df)):
        bool_list[i] = (df['internal_id'][i][0] == 9)
    df = (df[((bool_list) & (df['WZ'] > 0))][['ags', 'value', 'WZ']]
            .rename(columns = {'value': 'BZE'}))
    bze_je_lk_wz = (pd.pivot_table(df, values = 'BZE', index = 'WZ', 
                    columns = 'ags', fill_value = 0, dropna = False))
    bze_lk_wz = (pd.DataFrame(0.0 , index = bze_je_lk_wz.columns,
                             columns = wz_dict().values()))
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
    bze_lk_wz['31-32'] = (bze_je_lk_wz.transpose()[31] + 
                          bze_je_lk_wz.transpose()[32])
    for i in [33, 35, 36, 37]:
        bze_lk_wz[str(i)] = bze_je_lk_wz.transpose()[i]
    bze_lk_wz['38-39'] = (bze_je_lk_wz.transpose()[38] +
                          bze_je_lk_wz.transpose()[39]) 
    bze_lk_wz['41-42'] = (bze_je_lk_wz.transpose()[41] +
                          bze_je_lk_wz.transpose()[42])    
    for i in [43, 45, 46, 47, 49, 50, 51, 52, 53]:
        bze_lk_wz[str(i)] = bze_je_lk_wz.transpose()[i]
    bze_lk_wz['55-56'] = (bze_je_lk_wz.transpose()[55] +
                         bze_je_lk_wz.transpose()[56])
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
    bze_lk_wz['86-88'] = (bze_je_lk_wz.transpose()[86] + 
                          bze_je_lk_wz.transpose()[87] + 
                          bze_je_lk_wz.transpose()[88]) 
    for i in [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]:
        bze_lk_wz['90-99'] = bze_lk_wz['90-99'] + bze_je_lk_wz.transpose()[i]
    spez_gv = (pd.DataFrame(bze_lk_wz.transpose().drop_duplicates()
                 .sum(axis = 1)).merge(gv_wz_real, left_index = True, 
                                                   right_index = True))
    # Anpassung Gasverbrauch Lennys Tabelle
    #### spez_gv['GV WZ [MWh]'] * DF
    
    spez_gv['spez. GV'] = (spez_gv['GV WZ [MWh]'] / spez_gv[0]).transpose()
    spez_gv = spez_gv[['spez. GV']].transpose()
    spez_sv = (pd.DataFrame(bze_lk_wz.transpose().drop_duplicates()
                 .sum(axis = 1)).merge(sv_wz_real, left_index = True, 
                                                   right_index = True))
    spez_sv['spez. SV'] = spez_sv['SV WZ [MWh]'] / spez_sv[0]
    spez_sv = spez_sv[['spez. SV']].transpose()
    for item in [[7,8,9], [10,11,12], [13,14,15], [31,32], [38,39], [41,42], 
                 [55,56], [58,59,60,61,62,63], [64,65,66], 
                 [69,70,71,72,73,74,75], [77,78,79,80,81,82], [86,87,88],
                 [90,91,92,93,94,95,96,97,98,99]]:
        for i in item:
            spez_gv[i] = spez_gv[str(item[0]) + "-" + str(item[-1])]
            spez_sv[i] = spez_sv[str(item[0]) + "-" + str(item[-1])]
    spez_gv = spez_gv.drop(columns= ['7-9', '10-12', '13-15', '31-32', '38-39',
                                   '41-42', '55-56', '58-63', '64-66', '69-75',
                                   '77-82', '86-88', '90-99']).transpose()
    spez_gv.index = spez_gv.index.astype(int)
    spez_sv = spez_sv.drop(columns= ['7-9', '10-12', '13-15', '31-32', '38-39',
                                   '41-42', '55-56', '58-63', '64-66', '69-75',
                                   '77-82', '86-88', '90-99']).transpose()
    spez_sv.index = spez_sv.index.astype(int)
    
    return spez_sv.sort_index(), spez_gv.sort_index(), vb_wz, bze_je_lk_wz

def generate_specific_consumption_per_branch_and_district(iterations_power,
                                                          iterations_gas):
    """
    Returns specific power and gas consumption per branch and district.
    
    Parameters
    ----------
    iteration_power: int
        The amount of iterations to generate specific power consumption per 
        branch and district
    iteration_gas: int
        The amount of iterations to generate specific gas consumption per 
        branch and district
    
    Returns
    ------------
    Tuple that contains two pd.DataFrames
    """
    spez_vb = generate_specific_consumption_per_branch()
    spez_sv = spez_vb[0]
    spez_gv = spez_vb[1]
    vb_wz = spez_vb[2]
    bze_je_lk_wz = spez_vb[3]
    vb_LK = database_get('spatial', table_id = 15, year = 2015)
    vb_LK['Verbrauch in MWh'] = vb_LK['value'] / 3.6
    vb_LK['id_region'] = vb_LK['id_region'].astype(str)
    vb_LK = (vb_LK.assign(ags = [int(x[:-3]) for x in vb_LK['id_region']],
                                    ET = [x[0] for x in vb_LK['internal_id']]))
    vb_LK = (vb_LK.loc[((vb_LK['ET'] == 2) | (vb_LK['ET'] == 4))]
                             [['ags', 'Verbrauch in MWh', 'ET']]
                            .replace(to_replace = [3152, 3156], value = 3159))
    sv_LK_real = (vb_LK.loc[vb_LK['ET'] == 2].groupby(by = ['ags'])
                           [['Verbrauch in MWh']].sum())
    gv_LK_real = (vb_LK.loc[vb_LK['ET'] == 4].groupby(by = ['ags'])
                           [['Verbrauch in MWh']].sum())
    lk_ags = (vb_LK.groupby(by = ['ags', 'ET'])[['Verbrauch in MWh']].sum()
                            .reset_index()['ags'].unique())
    spez_gv_lk = pd.DataFrame(index = spez_gv.index, columns = lk_ags)
    spez_sv_lk = pd.DataFrame(index = spez_sv.index, columns = lk_ags)
    for lk in lk_ags:
        spez_gv_lk[lk] = spez_gv['spez. GV']
        spez_sv_lk[lk] = spez_sv['spez. SV']
    sv_lk_wz = bze_je_lk_wz * spez_sv_lk
    gv_lk_wz = bze_je_lk_wz * spez_gv_lk
    
    sv_wz_e_int = (vb_wz.loc[(vb_wz['WZ'].isin(['5','6','7-9','10-12',
                                               '13-15','16','17','18',
                                               '19','20','22','23','24',
                                               '25','27','28','29','33'])
                             &(vb_wz['ET'] == 18))].drop(columns = ['ET'])
                         .set_index('WZ'))
    gv_wz_e_int = (vb_wz.loc[(vb_wz['WZ'].isin(['5','6','7-9','10-12',
                                               '13-15','16','17','18',
                                               '19','20','21','22','23',
                                               '24','25','30']) & 
                             (vb_wz['ET'] == 12))].drop(columns = ['ET'])
                         .set_index('WZ'))
    sv_lk_wz_e_int = sv_lk_wz.loc[[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                   17, 18, 19, 20, 22, 23, 24, 25, 27, 28, 29,
                                   33]]
    gv_lk_wz_e_int = gv_lk_wz.loc[[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                  17, 18, 19, 20, 21, 22, 23, 24, 25, 30]]
    bze_sv_e_int = bze_je_lk_wz.loc[[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                   17, 18, 19, 20, 22, 23, 24, 25, 27, 28, 29,
                                   33]]
    bze_gv_e_int = bze_je_lk_wz.loc[[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                     17, 18, 19, 20, 21, 22, 23, 24, 25, 30]]
    sv_LK_real['Verbrauch e-arme WZ'] = (sv_lk_wz.loc[[21, 26, 30, 31, 32]]
                                                 .sum())
    sv_LK_real['Verbrauch e-int WZ'] = (sv_LK_real['Verbrauch in MWh'] -
                                        sv_LK_real['Verbrauch e-arme WZ'])
    gv_LK_real['Verbrauch e-arme WZ'] = (gv_lk_wz.loc[[26, 27, 28, 31, 32, 33]]
                                                 .sum())
    gv_LK_real['Verbrauch e-int WZ'] = (gv_LK_real['Verbrauch in MWh'] -
                                        gv_LK_real['Verbrauch e-arme WZ'])
    spez_sv_e_int = spez_sv_lk.loc[[5,6,7,8,9,10,11,12,13,14,15,16,17,
                                    18,19,20,22,23,24,25,27,28,29,33]]
    spez_gv_e_int = spez_gv_lk.loc[[5,6,7,8,9,10,11,12,13,14,15,16,
                                    17,18,19,20,21,22,23,24,25,30]]
    ET=[2,4]
    for et in ET:
        if (et == 2):
            sv_LK = sv_LK_real[['Verbrauch e-int WZ']]
            mean_value = sv_LK['Verbrauch e-int WZ'].sum() / len(sv_LK)
            spez_sv_angepasst = spez_sv_e_int.copy()
            spez_sv_angepasst.columns = spez_sv_angepasst.columns
            x = True
            while(x):
                iterations_power = iterations_power - 1
                if(iterations_power == 0):
                    break
                y = True
                i = 0
                while(y):
                    i = i+1
                    sv_LK['SV Modell e-int [MWh]'] = sv_lk_wz_e_int.sum()
                    sv_LK['Normierter relativer Fehler'] = (
                    (sv_LK['Verbrauch e-int WZ'] - 
                     sv_LK['SV Modell e-int [MWh]'])/mean_value)
                    sv_LK['Anpassungsfaktor'] = 1
                    (sv_LK['Anpassungsfaktor']
                    [((sv_LK['Normierter relativer Fehler']>0.1)|
                    (sv_LK['Normierter relativer Fehler']<-0.1))]) = (
                    sv_LK['Verbrauch e-int WZ']/sv_LK['SV Modell e-int [MWh]'])
                    if(sv_LK['Anpassungsfaktor'].sum() == 401):
                        y = False     
                    elif(i < 10):
                        spez_sv_angepasst = (spez_sv_angepasst * 
                                             sv_LK['Anpassungsfaktor']
                                             .transpose())
                        spez_sv_angepasst[spez_sv_angepasst<10] = 10
                        spez_sv_angepasst = (spez_sv_angepasst *
                                             sv_LK['Verbrauch e-int WZ'].sum()/
                                             sv_LK['SV Modell e-int [MWh]']
                                             .sum())
                        sv_lk_wz_e_int = bze_sv_e_int * spez_sv_angepasst
                    else:
                        y = False   
                sv_wz = (pd.DataFrame(sv_lk_wz_e_int.sum(axis = 1),
                                      columns = ['SV WZ Modell [MWh]']))
                k = 0
                z = True
                while(z):
                    k = k + 1
                    sv_wz_t51 = (pd.DataFrame(index= ['5','6','7-9','10-12',
                                                     '13-15','16','17','18',
                                                     '19','20','22','23','24',
                                                     '25','27','28','29','33'],
                                             columns=['SV WZ Modell [MWh]']))
                    sv_wz_t51['SV WZ Modell [MWh]'] = 0.0
                    WZe = [7, 8, 9]
                    for i in WZe:
                        sv_wz_t51['SV WZ Modell [MWh]']['7-9'] = (
                                sv_wz_t51['SV WZ Modell [MWh]']['7-9'] +
                                sv_wz['SV WZ Modell [MWh]'][i])
                    WZe = [10, 11, 12]
                    for i in WZe:
                        sv_wz_t51['SV WZ Modell [MWh]']['10-12'] = (
                                sv_wz_t51['SV WZ Modell [MWh]']['10-12'] +
                                sv_wz['SV WZ Modell [MWh]'][i])
                    WZe = [13, 14, 15]
                    for i in WZe:
                        sv_wz_t51['SV WZ Modell [MWh]']['13-15'] = (
                                sv_wz_t51['SV WZ Modell [MWh]']['13-15'] +
                                sv_wz['SV WZ Modell [MWh]'][i])
                    WZe = [5, 6, 16, 17, 18, 19, 20, 22, 23, 24, 25, 27, 28, 
                           29, 33]
                    for i in WZe:
                        sv_wz_t51['SV WZ Modell [MWh]'][str(i)] = (
                                                sv_wz['SV WZ Modell [MWh]'][i])
                    sv_wz_t51 = (sv_wz_t51.merge(sv_wz_e_int,left_index=True, 
                                                 right_index=True))
                    mean_value2 = sv_wz_t51['value'].sum()/len(sv_wz_t51)
                    sv_wz_t51['Normierter relativer Fehler'] = (
                            (sv_wz_t51['value'] - 
                             sv_wz_t51['SV WZ Modell [MWh]'])/mean_value2)
                    sv_wz_t51['Anpassungsfaktor'] = 1
                    (sv_wz_t51['Anpassungsfaktor']
                    [((sv_wz_t51['Normierter relativer Fehler']>0.01) | 
                    (sv_wz_t51['Normierter relativer Fehler']<-0.01))]) = (
                    sv_wz_t51['value']/sv_wz_t51['SV WZ Modell [MWh]'])
                    sv_wz['Anpassungsfaktor'] = 0.0
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
                        sv_wz = pd.DataFrame(sv_lk_wz_e_int.sum(axis = 1),
                                             columns = ['SV WZ Modell [MWh]'])
                    else:
                        z = False
        elif (et == 4):
            gv_LK = gv_LK_real[['Verbrauch e-int WZ']]
            mean_value = gv_LK['Verbrauch e-int WZ'].sum() / len(gv_LK)
            spez_gv_angepasst = spez_gv_e_int.copy()
            x = True
            while(x):
                iterations_gas = iterations_gas - 1
                if(iterations_gas == 0):
                    break
                y = True
                i = 0
                while(y):
                    i = i + 1
                    gv_LK['GV Modell e-int [MWh]'] = gv_lk_wz_e_int.sum()
                    gv_LK['Normierter relativer Fehler'] = (
                            (gv_LK['Verbrauch e-int WZ'] - 
                             gv_LK['GV Modell e-int [MWh]']) / mean_value)
                    gv_LK['Anpassungsfaktor'] = 1
                    (gv_LK['Anpassungsfaktor']
                    [((gv_LK['Normierter relativer Fehler']>0.1) | 
                    (gv_LK['Normierter relativer Fehler']<-0.1))]) = (
                                gv_LK['Verbrauch e-int WZ'] / 
                                gv_LK['GV Modell e-int [MWh]'])
                    if(gv_LK['Anpassungsfaktor'].sum() == 400):
                        y = False     
                    elif(i < 10):
                        spez_gv_angepasst = (spez_gv_angepasst * 
                                             gv_LK['Anpassungsfaktor']
                                             .transpose())
                        spez_gv_angepasst[spez_gv_angepasst<10] = 10
                        spez_gv_angepasst = (spez_gv_angepasst *
                                             gv_LK['Verbrauch e-int WZ'].sum()
                                             / gv_LK['GV Modell e-int [MWh]']
                                             .sum())
                        gv_lk_wz_e_int = bze_gv_e_int * spez_gv_angepasst
                    else:
                        y = False
                gv_wz = pd.DataFrame(gv_lk_wz_e_int.sum(axis = 1),
                                     columns = ['GV WZ Modell [MWh]'])
                k = 0
                z = True
                while(z):
                    k = k + 1
                    gv_wz_t51 = (pd.DataFrame(index = ['5', '6', '7-9', 
                                                       '10-12', '13-15', '16', 
                                                       '17', '18', '19', '20',
                                                       '21', '22', '23', '24', 
                                                       '25', '30'],
                                              columns=['GV WZ Modell [MWh]']))
                    gv_wz_t51['GV WZ Modell [MWh]'] = 0.0
                    WZe = [7, 8, 9]
                    for i in WZe:
                        gv_wz_t51['GV WZ Modell [MWh]']['7-9'] = (
                                gv_wz_t51['GV WZ Modell [MWh]']['7-9'] +
                                gv_wz['GV WZ Modell [MWh]'][i])
                    WZe = [10, 11, 12]
                    for i in WZe:
                        gv_wz_t51['GV WZ Modell [MWh]']['10-12'] = (
                                gv_wz_t51['GV WZ Modell [MWh]']['10-12'] + 
                                gv_wz['GV WZ Modell [MWh]'][i])
                    WZe = [13, 14, 15]
                    for i in WZe:
                        gv_wz_t51['GV WZ Modell [MWh]']['13-15'] = (
                                gv_wz_t51['GV WZ Modell [MWh]']['13-15'] + 
                                gv_wz['GV WZ Modell [MWh]'][i])
                    WZe = [5, 6, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30]
                    for i in WZe:
                        gv_wz_t51['GV WZ Modell [MWh]'][str(i)] = (
                                gv_wz['GV WZ Modell [MWh]'][i])
                    gv_wz_t51 = gv_wz_t51.merge(gv_wz_e_int, left_index = True, 
                                                right_index = True)
                    mean_value2 = gv_wz_t51['value'].sum() / len(gv_wz_t51)
                    gv_wz_t51['Normierter relativer Fehler'] = (
                            (gv_wz_t51['value'] - 
                             gv_wz_t51['GV WZ Modell [MWh]'])/mean_value2)
                    gv_wz_t51['Anpassungsfaktor'] = 1
                    (gv_wz_t51['Anpassungsfaktor']
                    [((gv_wz_t51['Normierter relativer Fehler']>0.01) | 
                    (gv_wz_t51['Normierter relativer Fehler']<-0.01))]) = (
                        gv_wz_t51['value'] / gv_wz_t51['GV WZ Modell [MWh]'])
                    gv_wz['Anpassungsfaktor'] = 0.0
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
                        spez_gv_angepasst[spez_gv_angepasst<10] = 10
                        gv_lk_wz_e_int = bze_gv_e_int * spez_gv_angepasst
                        gv_wz = pd.DataFrame(gv_lk_wz_e_int.sum(axis = 1),
                                             columns=['GV WZ Modell [MWh]'])
                    else:
                        z = False
    spez_sv_lk.loc[list(spez_sv_angepasst.index)] = spez_sv_angepasst.values
    spez_gv_lk.loc[list(spez_gv_angepasst.index)] = spez_gv_angepasst.values
    spez_gv_lk[3103] = spez_gv['spez. GV']
    spez_sv_lk.sort_index(axis = 1).to_csv(
            './data_in/regional/specific_power_consumption.csv')
    spez_gv_lk.sort_index(axis = 1).to_csv(
            './data_in/regional/specific_gas_consumption.csv')
    return spez_sv_lk.sort_index(axis = 1), spez_gv_lk.sort_index(axis = 1)
    
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
        fn = _data_in('regional', cfg['elc_cons_HH_spatial']['filename'])
        df = read_local(fn, year = year)
    elif source == 'database':
        df = database_get('spatial', table_id = table_id, year = year,
                          force_update = force_update)
    else:
        raise KeyError('Wrong source key given in config.yaml - must be either'
                       ' `local` or `database` but is: {}'.format(source))

    df = (df.assign(nuts3 = lambda x: x.id_region.map(region_id_to_nuts3()))
            .set_index('nuts3').sort_index(axis = 0))['value']
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
    bt = kwargs.get('internal_id_0', cfg['living_space']['internal_id'][0])
    vc = kwargs.get('internal_id_1', cfg['living_space']['internal_id'][1])
    hs = kwargs.get('internal_id_2', cfg['living_space']['internal_id'][2])
    ne = kwargs.get('internal_id_3', cfg['living_space']['internal_id'][3])

    if source == 'local':
        df = read_local(_data_in('regional', cfg['living_space']['filename']))
    elif source == 'database':
        df = database_get('spatial', table_id=table_id, year=year,
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
        df = plausibility_check_nuts3(df)
    else:
        df = df.drop(['id_spatial', 'id_region_type', 'id_region'], axis=1)
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
    df = plausibility_check_nuts3(df)
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
    bt = kwargs.get('internal_id_0', cfg['heat_dem_bld']['internal_id'][0])
    vc = kwargs.get('internal_id_1', cfg['heat_dem_bld']['internal_id'][1])
    hp = kwargs.get('internal_id_2', cfg['heat_dem_bld']['internal_id'][2])
    va = kwargs.get('internal_id_3', cfg['heat_dem_bld']['internal_id'][3])

    if source == 'local':
        raise NotImplementedError('Not here yet!')
    elif source == 'database':
        df = database_get('spatial', table_id=table_id, year=year,
                          force_update=force_update)
    else:
        raise KeyError('Wrong source key given in config.yaml - must be either'
                       ' `local` or `database` but is: {}'.format(source))

    df = (df.assign(nuts3=lambda x: x.id_region.map(region_id_to_nuts3()),
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
    es_rate = (pd.read_excel(
                    './data_in/temporal/Efficiency_Enhancement_Rates.xlsx')
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
        columns: NUTS-3 codes
    """
    
    year = kwargs.get('year', cfg['base_year'])
    scenario = kwargs.get('scenario', cfg['scenario'])
    
    if year in range(2015, 2019):
        df = database_get('spatial', table_id = 18, year = year)
        df = (df.assign(ags = [int(x[:-3]) for x in 
                               df['id_region'].astype(str)],
                               WZ = [x[1] for x in df['internal_id']]))
        bool_list = np.array(df['id_region'].astype(str))
        for i in range(0, len(df)):
            bool_list[i] = (df['internal_id'][i][0] == 9)
        df = (df[((bool_list) & (df['WZ'] > 0))][['ags', 'value', 'WZ']]
                .rename(columns = {'value': 'BZE'}))
        df = (pd.pivot_table(df, values = 'BZE', index = 'WZ', 
                             columns = 'ags', fill_value = 0, dropna = False))
    elif year in range(2019, 2036):
        if scenario == 'Basis':
            df = database_get('spatial', table_id = 27, year = year)
        elif scenario == 'Digital':
            df = database_get('spatial', table_id = 28, year = year)
            
        df = (df.assign(ags = [int(x[:-3]) for x in 
                               df['id_region'].astype(str)],
                               WZ = [x[0] for x in df['internal_id']]))
        df = (pd.pivot_table(df, values = 'value', index = 'WZ', 
                             columns = 'ags', fill_value = 0, dropna = False))
    
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

def Leistung(Tag_Zeit, mask, df, df_SLP):
    """
    Returns
    -------
    pd.Series
    """
    u = (pd.merge(df[mask], df_SLP[['Stunde', Tag_Zeit]], 
                  on = ['Stunde'], how = 'left'))
    v = pd.merge(df, u[['Date', Tag_Zeit]], on = ['Date'], how = 'left')  
    v[Tag_Zeit][v[Tag_Zeit] != v[Tag_Zeit]] = 0  
    return v[Tag_Zeit]

def shift_load_profile_generator(state, **kwargs):
    """
    Return shift load profiles in normalized units
    ('normalized' means here that the sum over all time steps equals one).
    
    Parameter
    -------
    state : str
        Must be one of ['BW','BY','BE','BB','HB','HH','HE','MV',
                        'NI','NW','RP','SL','SN','ST','SH','TH']
    
    Returns
    -------
    pd.DataFrame
    """
    year = kwargs.get('year', cfg['base_year'])
    low = 0.3
    if ((year % 4 == 0) & (year % 100 != 0) | (year % 4 == 0) 
          & (year % 100 == 0) & (year % 400 == 0)):
        periods = 35136
    else:
        periods = 35040 
    df = (pd.DataFrame(data= {"Date": pd.date_range((str(year) + '-01-01'), 
                                                     periods = periods, 
                                                     freq = '15T', 
                                                     tz = 'Europe/Berlin')}))
    df['Tag'] = pd.DatetimeIndex(df['Date']).date
    df['Stunde'] = pd.DatetimeIndex(df['Date']).time
    df['DayOfYear'] = pd.DatetimeIndex(df['Date']).dayofyear.astype(int)
    mask_holiday = [] 
    for i in range(0, len(holidays.DE(state = state, years = year))):
        mask_holiday.append('Null')
        mask_holiday[i] = ((df['Tag'] == [x for x in holidays.DE(state = 
                            state, years = year).items()][i][0]))
    HD = mask_holiday[0]
    for i in range(1, len(holidays.DE(state = state, years = year))):
        HD = HD | mask_holiday[i]
    df['WT'] = df['Date'].apply(lambda x: x.weekday() <  5) 
    df['WT'] = df['WT'] & (HD == False) 
    df['SA'] = df['Date'].apply(lambda x: x.weekday() == 5) 
    df['SA'] = df['SA'] & (HD==False) 
    df['SO'] = df['Date'].apply(lambda x: x.weekday() == 6) 
    df['SO'] = df['SO'] | HD  
    df['WT'][(df['Tag'] == datetime.date(year, 12, 24))] = False
    df['WT'][(df['Tag'] == datetime.date(year, 12, 31))] = False
    df['SO'][(df['Tag'] == datetime.date(year, 12, 24))] = False
    df['SO'][(df['Tag'] == datetime.date(year, 12, 31))] = False
    df['SA'][(df['Tag'] == datetime.date(year, 12, 24))] = True
    df['SA'][(df['Tag'] == datetime.date(year, 12, 31))] = True
    for sp in ['S1_WT','S1_WT_SA','S1_WT_SA_SO','S2_WT',
               'S2_WT_SA','S2_WT_SA_SO','S3_WT','S3_WT_SA',
               'S3_WT_SA_SO']:
        if(sp == 'S1_WT'):
            anzahl_wz =  17 / 48 * len(df[df['WT']])
            anzahl_nwz = (31 / 48 * len(df[df['WT']]) + len(df[df['SO']]) + 
                             len(df[df['SA']]))
            anteil = 1 / (anzahl_wz + low * anzahl_nwz)
            df[sp] = anteil
            df[sp][df['SO']] = low * anteil
            df[sp][df['SA']] = low * anteil
            mask = ((df['WT'])&
                    ((df['Stunde'] < pd.to_datetime('08:00:00').time()) |
                    (df['Stunde'] >= pd.to_datetime('16:30:00').time())))
            df[sp][mask] = low * anteil
        elif(sp == 'S1_WT_SA'):    
            anzahl_wz = (17/48 * len(df[df['WT']]) + 
                         17/48 * len(df[df['SA']]))
            anzahl_nwz = (31/48 * len(df[df['WT']]) + len(df[df['SO']]) + 
                          31/48 * len(df[df['SA']]))
            anteil = 1 / (anzahl_wz + low * anzahl_nwz) 
            df[sp] = anteil
            df[sp][df['SO']] = low * anteil
            mask =((df['WT']) & ((df['Stunde'] < pd.to_datetime('08:00:00')
                      .time()) | (df['Stunde'] >= pd.to_datetime('16:30:00')
                      .time())))
            df[sp][mask] = low * anteil
            mask =((df['SA']) & ((df['Stunde'] < pd.to_datetime('08:00:00')
                     .time()) | (df['Stunde'] >= pd.to_datetime('16:30:00')
                     .time())))
            df[sp][mask] = low * anteil
        elif(sp == 'S1_WT_SA_SO'):    
            anzahl_wz = (17/48* (len(df[df['WT']]) + len(df[df['SO']]) + 
                                 len(df[df['SA']])))
            anzahl_nwz = (31/48 * (len(df[df['WT']]) + len(df[df['SO']]) +
                                   len(df[df['SA']])))
            anteil = 1 / (anzahl_wz + low * anzahl_nwz) 
            df[sp] = anteil
            mask =((df['Stunde'] < pd.to_datetime('08:00:00').time()) |
                    (df['Stunde'] >= pd.to_datetime('16:30:00').time()))
            df[sp][mask] = low * anteil
        elif(sp == 'S2_WT'):    
            anzahl_wz = 17/24 * len(df[df['WT']])
            anzahl_nwz = (7/24 * len(df[df['WT']]) + len(df[df['SO']]) +
                          len(df[df['SA']]))
            anteil = 1 / (anzahl_wz + low * anzahl_nwz) 
            df[sp] = anteil
            df[sp][df['SO']] = low * anteil
            df[sp][df['SA']] = low * anteil
            mask =((df['WT']) &
                   ((df['Stunde'] < pd.to_datetime('06:00:00').time())|
                    (df['Stunde'] >= pd.to_datetime('23:00:00').time())))
            df[sp][mask] = low * anteil
        elif(sp == 'S2_WT_SA'):    
            anzahl_wz = 17/24 * (len(df[df['WT']]) + len(df[df['SA']]))
            anzahl_nwz = (7/24 * len(df[df['WT']]) + len(df[df['SO']]) +
                          7/24* len(df[df['SA']]))
            anteil = 1 / (anzahl_wz + low * anzahl_nwz) 
            df[sp] = anteil
            df[sp][df['SO']] = low * anteil
            mask =(((df['WT']) | (df['SA'])) &
                   ((df['Stunde'] < pd.to_datetime('06:00:00')
                   .time()) | (df['Stunde'] >= pd.to_datetime('23:00:00')
                   .time())))
            df[sp][mask] = low * anteil
        elif(sp == 'S2_WT_SA_SO'):    
            anzahl_wz = (17/24 * (len(df[df['WT']]) + len(df[df['SA']]) +
                                  len(df[df['SO']])))
            anzahl_nwz = (7/24 * (len(df[df['WT']]) + len(df[df['SO']]) +
                                  len(df[df['SA']])))
            anteil = 1 / (anzahl_wz + low * anzahl_nwz) 
            df[sp] = anteil
            mask =(((df['Stunde'] < pd.to_datetime('06:00:00').time()) |
                    (df['Stunde'] >= pd.to_datetime('23:00:00').time())))
            df[sp][mask] = low * anteil
        elif(sp == 'S3_WT_SA_SO'):
            anteil = 1 / periods
            df[sp] = anteil
        elif(sp == 'S3_WT'):    
            anzahl_wz = len(df[df['WT']])
            anzahl_nwz = len(df[df['SO']]) + len(df[df['SA']])
            anteil = 1 / (anzahl_wz + low * anzahl_nwz) 
            df[sp] = anteil
            df[sp][df['SO']] = low * anteil
            df[sp][df['SA']] = low * anteil
        elif(sp == 'S3_WT_SA'):    
            anzahl_wz = len(df[df['WT']]) + len(df[df['SA']])
            anzahl_nwz = len(df[df['SO']])
            anteil = 1 / (anzahl_wz + low * anzahl_nwz) 
            df[sp] = anteil
            df[sp][df['SO']] = low * anteil
    df= (df[['Date','S1_WT','S1_WT_SA','S1_WT_SA_SO','S2_WT','S2_WT_SA',
             'S2_WT_SA_SO','S3_WT','S3_WT_SA','S3_WT_SA_SO']]
            .set_index('Date'))
    return df



def gas_slp_generator(state, **kwargs):
    """
    Return the gas standard load profiles in normalized units
    ('normalized' means here that the sum over all time steps equals 
    365 (or 366 in a leapyear)).
    
    Parameter
    -------
    state: str
        must be one of ['BW','BY','BE','BB','HB','HH','HE','MV',
                        'NI','NW','RP','SL','SN','ST','SH','TH']
    Returns
    -------
    pd.DataFrame
    """
    year = kwargs.get('year', cfg['base_year'])
    if ((year % 4 == 0) & (year % 100 != 0) | (year % 4 == 0) 
        & (year % 100 == 0) & (year % 400 == 0)):
        days = 366
    else:
        days = 365 
        
    df = (pd.DataFrame(data = {"Date": pd.date_range((str(year) + '-01-01'),
                                                   periods = days,
                                                   freq = 'd',
                                                   tz = 'Europe/Berlin')})) 
    df = df.assign(Tag = pd.DatetimeIndex(df['Date']).date,
                   DayOfYear = pd.DatetimeIndex(df['Date'])
                                 .dayofyear.astype(int))
    mask_holiday = [] 
    for i in range(0,len(holidays.DE(state = state, years = year))):
        mask_holiday.append('Null')
        mask_holiday[i] = ((df['Tag'] == [x for x in holidays.DE(state = state, 
                                          years = year).items()][i][0]))
    HD = mask_holiday[0]
    for i in range(1,len(holidays.DE(state = state, years = year))):
        HD = HD | mask_holiday[i]
    df['MO'] = df['Date'].apply(lambda x: x.weekday() ==  0)
    df['MO'] = df['MO'] & (HD == False)
    df['DI'] = df['Date'].apply(lambda x: x.weekday() ==  1)
    df['DI'] = df['DI'] & (HD == False) 
    df['MI'] = df['Date'].apply(lambda x: x.weekday() ==  2) 
    df['MI'] = df['MI'] & (HD == False)
    df['DO'] = df['Date'].apply(lambda x: x.weekday() ==  3) 
    df['DO'] = df['DO'] & (HD == False) 
    df['FR'] = df['Date'].apply(lambda x: x.weekday() ==  4)
    df['FR'] = df['FR'] & (HD == False)
    df['SA'] = df['Date'].apply(lambda x: x.weekday() == 5)
    df['SA'] = df['SA'] & (HD == False) 
    df['SO'] = df['Date'].apply(lambda x: x.weekday() == 6) 
    df['SO'] = df['SO'] | HD  
    df['MO'][(df['Tag'] == datetime.date(int(year), 12, 24))] = False
    df['MO'][(df['Tag'] == datetime.date(int(year), 12, 31))] = False
    df['DI'][(df['Tag'] == datetime.date(int(year), 12, 24))] = False
    df['DI'][(df['Tag'] == datetime.date(int(year), 12, 31))] = False
    df['MI'][(df['Tag'] == datetime.date(int(year), 12, 24))] = False
    df['MI'][(df['Tag'] == datetime.date(int(year), 12, 31))] = False
    df['DO'][(df['Tag'] == datetime.date(int(year), 12, 24))] = False
    df['DO'][(df['Tag'] == datetime.date(int(year), 12, 31))] = False
    df['FR'][(df['Tag'] == datetime.date(int(year), 12, 24))] = False
    df['FR'][(df['Tag'] == datetime.date(int(year), 12, 31))] = False
    df['SA'][(df['Tag'] == datetime.date(int(year), 12, 24))] = True
    df['SA'][(df['Tag'] == datetime.date(int(year), 12, 31))] = True
    df['SO'][(df['Tag'] == datetime.date(int(year), 12, 24))] = False
    df['SO'][(df['Tag'] == datetime.date(int(year), 12, 31))] = False
    par = pd.DataFrame.from_dict(gas_load_profile_parameters_dict())
    for slp in par.index:
        df2 = par.loc[par.index == slp].reset_index()
        df['FW_'+str(slp)] = 0
        df['FW_'+str(slp)][df['MO']] = df2['MO'][0]
        df['FW_'+str(slp)][df['DI']] = df2['DI'][0]
        df['FW_'+str(slp)][df['MI']] = df2['MI'][0]
        df['FW_'+str(slp)][df['DO']] = df2['DO'][0]
        df['FW_'+str(slp)][df['FR']] = df2['FR'][0]
        df['FW_'+str(slp)][df['SA']] = df2['SA'][0]
        df['FW_'+str(slp)][df['SO']] = df2['SO'][0]
        summe = df['FW_'+str(slp)].sum()
        df['FW_'+str(slp)] = df['FW_'+str(slp)] * days/summe
    return df.drop(columns=['DayOfYear']).set_index('Tag')


def power_slp_generator(state, **kwargs):
    """
    Return the electric standard load profiles in normalized units
    ('normalized' means here that the sum over all time steps equals one).
    
    Parameter
    -------
    state: str
        must be one of ['BW','BY','BE','BB','HB','HH','HE','MV',
                        'NI','NW','RP','SL','SN','ST','SH','TH']
    
    Returns
    -------
    pd.DataFrame
    """
    year = kwargs.get('year', cfg['base_year'])
    if ((year % 4 == 0) & (year % 100 != 0) | (year % 4 == 0) 
        & (year % 100 == 0) & (year % 400 == 0)):
        periods = 35136
    else:
        periods = 35040 
    df = (pd.DataFrame(data = {"Date": pd.date_range((str(year) + '-01-01'),
                                                   periods = periods,
                                                   freq = '15T',
                                                   tz = 'Europe/Berlin')}))
    df['Tag'] = pd.DatetimeIndex(df['Date']).date
    df['Stunde'] = pd.DatetimeIndex(df['Date']).time
    df['DayOfYear'] = pd.DatetimeIndex(df['Date']).dayofyear.astype(int)
    mask_holiday = [] 
    for i in range(0,len(holidays.DE(state = state, years = year))):
        mask_holiday.append('Null')
        mask_holiday[i] = ((df['Tag'] == [x for x in holidays.DE(state = state, 
                                          years = year).items()][i][0]))
    HD = mask_holiday[0]
    for i in range(1,len(holidays.DE(state = state, years = year))):
        HD = HD | mask_holiday[i]
    df['WT'] = df['Date'].apply(lambda x: x.weekday() <  5) 
    df['WT'] = df['WT'] & (HD == False) 
    df['SA'] = df['Date'].apply(lambda x: x.weekday() == 5) 
    df['SA'] = df['SA'] & (HD==False) 
    df['SO'] = df['Date'].apply(lambda x: x.weekday() == 6) 
    df['SO'] = df['SO'] | HD
    df['WT'][(df['Tag'] == datetime.date(int(year), 12, 24))] = False
    df['WT'][(df['Tag'] == datetime.date(int(year), 12, 31))] = False
    df['SO'][(df['Tag'] == datetime.date(int(year), 12, 24))] = False
    df['SO'][(df['Tag'] == datetime.date(int(year), 12, 31))] = False
    df['SA'][(df['Tag'] == datetime.date(int(year), 12, 24))] = True
    df['SA'][(df['Tag'] == datetime.date(int(year), 12, 31))] = True
    df_wiz1 = df.loc[df['Date'] < (str(year) + '-03-21 00:00:00')] 
    df_wiz2 = df.loc[df['Date'] >= (str(year) + '-11-01')] 
    df_soz  = (df.loc[((str(year) + '-05-15') <= df['Date']) & 
                     (df['Date'] < (str(year) + '-09-15'))]) 
    df_uez1 = (df.loc[((str(year) + '-03-21') <= df['Date']) & 
                     (df['Date'] < (str(year) + '-05-15'))])
    df_uez2 =  (df.loc[((str(year) + '-09-15') <= df['Date']) & 
                       (df['Date'] <= (str(year) + '-10-31'))])
    df['WIZ'] = (df['Tag'].isin(df_wiz1['Tag']) | 
                 df['Tag'].isin(df_wiz2['Tag'])) 
    df['SOZ'] = df['Tag'].isin(df_soz['Tag'])
    df['UEZ'] = (df['Tag'].isin(df_uez1['Tag']) | 
                 df['Tag'].isin(df_uez2['Tag']))
    for Tarifkundenprofil in ['H0','L0','L1','L2','G0','G1',
                              'G2','G3','G4','G5','G6']:
        path = ('./data_in/temporal/Power Load Profiles/39_VDEW_Strom_' +
                 'Repräsentative Profile_' + Tarifkundenprofil + '.xlsx')
        df_load = pd.read_excel(path, sep = ';', decimal = ',')
        df_load.columns = ['Stunde', 'SA_WIZ', 'SO_WIZ',  'WT_WIZ', 'SA_SOZ',
                           'SO_SOZ',  'WT_SOZ', 'SA_UEZ', 'SO_UEZ',  'WT_UEZ']
        df_load.loc[1] = df_load.loc[len(df_load) - 2] 
        df_SLP = df_load[1:97] 
        df_SLP = df_SLP.reset_index()[['Stunde', 'SA_WIZ', 'SO_WIZ',  'WT_WIZ',
                                   'SA_SOZ', 'SO_SOZ',  'WT_SOZ', 'SA_UEZ', 
                                   'SO_UEZ',  'WT_UEZ']]
        wt_wiz = Leistung ('WT_WIZ', (df.WT & df.WIZ), df, df_SLP) 
        wt_soz = Leistung ('WT_SOZ', (df.WT & df.SOZ), df, df_SLP)
        wt_uez = Leistung ('WT_UEZ', (df.WT & df.UEZ), df, df_SLP)
        wt = wt_wiz + wt_soz + wt_uez
        sa_wiz = Leistung ('SA_WIZ', (df.SA & df.WIZ), df, df_SLP) 
        sa_soz = Leistung ('SA_SOZ', (df.SA & df.SOZ), df, df_SLP)
        sa_uez = Leistung ('SA_UEZ', (df.SA & df.UEZ), df, df_SLP)
        sa = sa_wiz + sa_soz + sa_uez
        so_wiz = Leistung ('SO_WIZ', (df.SO & df.WIZ), df, df_SLP) 
        so_soz = Leistung ('SO_SOZ', (df.SO & df.SOZ), df, df_SLP)
        so_uez = Leistung ('SO_UEZ', (df.SO & df.UEZ), df, df_SLP)
        so = so_wiz + so_soz + so_uez
        Summe = wt + sa + so
        Last = 'Last_' + str(Tarifkundenprofil)
        df[Last] = Summe       
        total = sum(df[Last])
        df_normiert = df[Last] / total
        df[Tarifkundenprofil] = df_normiert

    slp_bl = (df.drop(columns = ['Last_H0', 'Last_L0', 'Last_L1', 'Last_L2',
                                 'Last_G0', 'Last_G1', 'Last_G2', 'Last_G3',
                                 'Last_G4', 'Last_G5', 'Last_G6'])
                .set_index('Date'))
    return slp_bl



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
              .assign(nuts3=lambda x: x.id_region.map(region_id_to_nuts3()))
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
    int_id_1 = kwargs.get('internal_id_1', None)
    int_id_2 = kwargs.get('internal_id_2', None)
    int_id_3 = kwargs.get('internal_id_3', None)
    int_id_4 = kwargs.get('internal_id_4', None)
    int_id_5 = kwargs.get('internal_id_5', None)
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
    if table == 'v_demandregio_spatial_lk401':
        if int_id_1 is not None:
            query += '&&' + 'internal_id_1' + '=eq.{' + str(int_id_1) + '}'
        if int_id_2 is not None:
            query += '&&' + 'internal_id_1' + '=eq.{' + str(int_id_2) + '}'
        if int_id_3 is not None:
            query += '&&' + 'internal_id_1' + '=eq.{' + str(int_id_3) + '}'
        if int_id_4 is not None:
            query += '&&' + 'internal_id_1' + '=eq.{' + str(int_id_4) + '}'
        if int_id_5 is not None:
            query += '&&' + 'internal_id_1' + '=eq.{' + str(int_id_5) + '}'
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
    A_db = set(region_id_to_nuts3().values())
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
