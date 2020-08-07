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
Provides configuration settings.
"""

import os
import yaml
import requests
import pandas as pd
import logging
import hashlib
from ast import literal_eval as lit_eval
logger = logging.getLogger(__name__)


def data_out(*fn):
    dirpath = os.path.join(os.path.dirname(__file__), '..', 'data_out')
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)
    return os.path.join(dirpath, *fn)


def data_in(*fn):
    return os.path.join(os.path.dirname(__file__), '..', 'data_in', *fn)


def get_config(filename=None, **kwargs):
    if filename is None:
        filename = os.path.join(os.path.dirname(__file__), 'config.yaml')
    assert os.path.exists(filename), (
        "The config file '{}' does not exist yet. "
        "Copy config_example.yaml to config.yaml and fill in details, "
        "as necessary.".format(filename))
    yaml_ver = [int(v) for v in yaml.__version__.split('.')]
    if (yaml_ver[0] > 5) or (yaml_ver[0] == 5 and yaml_ver[1] >= 1):
        with open(filename) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        logger.warn("Please update your `PyYAML` package to v5.1 or higher.")
        with open(filename) as f:
            config = yaml.load(f)
    return config


def database_raw(query, force_update=False):
    """
    Perform a string-based Rest-API query.

    Parameters
    ----------
    query : str
        The query string.
    force_update : bool, optional
        If True, perform a fresh database query and overwrite data in cache.

    Returns
    -------
    pd.DataFrame
    """
    # Check if query is str and create hash from that string
    if not isinstance(query, str):
        raise ValueError('`query` must be a string.')
    sha1 = hashlib.sha1(query.encode()).hexdigest()

    # Check if caching directory exists and create if not.
    if not os.path.isdir(data_in('__cache__/')):
        os.mkdir(data_in('__cache__/'))

    # If file has already been cached, read cache, else query from API + save.
    filename = data_in('__cache__/{}.csv'.format(sha1))
    if os.path.exists(filename) and (force_update is False):
        return pd.read_csv(filename, index_col='idx', encoding='utf-8',
                           engine='c',
                           converters={'internal_id': literal_converter,
                                       'region_types': literal_converter,
                                       'values': literal_converter,
                                       'years': literal_converter,
                                       'years_step': literal_converter,
                                       'years_weather': literal_converter})
    else:
        host = get_config()['database_host']
        logger.info('Querying from:\n' + host + query)
        try:
            df = pd.read_json(requests.get(host + query).content,
                              dtype=get_config()['dtypes'])
        except ValueError:
            ser = pd.read_json(requests.get(host + query).content,
                               dtype=get_config()['dtypes'], typ='series')
            logger.error(ser)
            raise ValueError('Query returned a series:\n{}'.format(ser))
        if len(df) == 0:
            raise ValueError('The returned pd.DataFrame is empty!')
        else:
            df.to_csv(filename, index_label='idx', encoding='utf-8')
            return df


def clear_local_cache():
    """
    Clear the local query cache.
    """
    # Check if caching directory exists and create if not.
    if not os.path.isdir(data_in('__cache__/')):
        os.mkdir(data_in('__cache__/'))

    deleted = False
    for the_file in os.listdir(data_in('__cache__')):
        file_path = os.path.join(data_in('__cache__'), the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                deleted = True
        except Exception as e:
            print(e)
    if deleted:
        logger.info('Successfully cleared local cache.')
    else:
        logger.info('Local cache already empty.')


def dict_region_code(keys='id_ags', values='natcode_nuts3', level='lk',
                     raw=False, **kwargs):
    """
    Read and return a dictionary with regional information.

    Examples
    --------
    keys='id_ags', values='natcode_nuts3', level='lk':
        return <region_id>: <nuts3> e.g. '1001000': 'DEF01'

    keys='natcode_nuts3', values='name', level='lk':
        return <nuts3>: <name> e.g. 'DEF01': 'Flensburg, Kreisfreie Stadt'

    Parameters
    ----------
    keys : str
        The column label for the keys.
    values : str
        The column label for the values.
    level : str, default 'lk'
        must be in ['lk', 'bl']. 'lk'=Landkreise=nuts3, 'bl'=Bundesländer=nuts1
    raw : bool, default False
        return untouched pd.DataFrame if True else dict

    Returns
    -------
    dict or pd.DataFrame
    """
    dict_source = kwargs.get('dict_source', get_config()['dict_source'])
    # Argument error handling
    assert dict_source in ['local', 'database'], (
        "`dict_source` must be in ['local', 'database']")
    if level == 'lk':
        columns = ['id_t_nuts3_lk', 'id_nuts3', 'natcode_nuts3', 'name',
                   'id_ags_lk', 'ags_lk', 'id_ags', 'ags_bl', 'bl']
    elif level == 'bl':
        columns = ['id_t_nuts1_bl', 'id_nuts1', 'natcode_nuts1', 'name',
                   'id_ags_bl', 'ags_bl', 'id_ags', 'bl']
    else:
        raise ValueError("`level` must be in ['lk', 'bl']")
    assert keys in columns, "`keys` must be in {}".format(columns)
    assert values in columns, "`values` must be in {}".format(columns)
    # Read the requested data
    if dict_source == 'local' and level == 'lk':
        df = pd.read_csv(data_in('regional/t_nuts3_lk.csv'), encoding='utf-8')
    elif dict_source == 'local' and level == 'bl':
        df = pd.read_csv(data_in('regional/t_nuts1_bl.csv'), encoding='utf-8')
    elif dict_source == 'database' and level == 'lk':
        df = database_raw('t_nuts3_lk')
    elif dict_source == 'database' and level == 'bl':
        df = database_raw('t_nuts1_bl')
    else:
        raise ValueError("ELSE reached, this cannot be!")
    # Filter and return
    if raw:
        return df
    else:
        return df.set_index(keys).loc[:, values].to_dict()


def literal_converter(val):
    try:
        return lit_eval(val)
    except (SyntaxError, ValueError):
        return val


def wz_dict():
    """
    This dictionary translates the database namings to real WZ branch names.
    """
    return {2: '1', 3: '2', 4: '3', 6: '5', 7: '6', 8: '7-9', 10: '10-12',
            11: '13-15', 12: '16', 13: '17', 14: '18', 15: '19', 18: '20',
            19: '21', 20: '22', 21: '23', 24: '24', 28: '25', 29: '26',
            30: '27', 31: '28', 32: '29', 33: '30', 34: '31-32', 35: '33',
            36: '35', 40: '36', 42: '37', 43: '38-39', 45: '41-42', 46: '43',
            48: '45', 49: '46', 50: '47', 52: '49', 53: '49', 54: '50',
            55: '51', 56: '52', 57: '53', 58: '55-56', 59: '58-63',
            60: '64-66', 61: '68', 62: '69-75', 63: '77-82', 64: '84',
            65: '85', 66: '86-88', 67: '90-99'}


def hist_weather_year():
    """
    This dictionary assigns the temperature data of a historical year to
    future years
    """
    return {2015: 2015, 2016: 2016, 2017: 2017, 2018: 2018, 2019: 2006,
            2020: 2008, 2021: 2009, 2022: 2010, 2023: 2011, 2024: 2012,
            2025: 2013, 2026: 2014, 2027: 2015, 2028: 2016, 2029: 2017,
            2030: 2018, 2031: 2007, 2032: 2008, 2033: 2009, 2034: 2010,
            2035: 2021}


def bl_dict():
    """
    This dictionary translates the Bundesland number to its abbreciation.
    """
    return {1: 'SH', 2: 'HH', 3: 'NI', 4: 'HB', 5: 'NW', 6: 'HE',
            7: 'RP', 8: 'BW', 9: 'BY', 10: 'SL', 11: 'BE', 12: 'BB',
            13: 'MV', 14: 'SN', 15: 'ST', 16: 'TH'}


def slp_branch_cts_power():
    """
    This dictionary assignes a power load profile (SLP) to every CTS branch
    """
    return {1: 'L0', 2: 'L0', 3: 'G3', 35: 'G3', 36: 'G3', 37: 'G3',
            38: 'G3', 39: 'G3', 41: 'G1', 42: 'G1', 43: 'G1', 45: 'G4',
            46: 'G0', 47: 'G0', 49: 'G3', 50: 'G3', 51: 'G3', 52: 'G3',
            53: 'G4', 55: 'G2', 56: 'G2', 58: 'G1', 59: 'G0', 60: 'G3',
            61: 'G3', 62: 'G3', 63: 'G3', 64: 'G1', 65: 'G1', 66: 'G1',
            68: 'G1', 69: 'G1', 70: 'G1', 71: 'G1', 72: 'G1', 73: 'G1',
            74: 'G1', 75: 'G1', 77: 'G4', 78: 'G1', 79: 'G4', 80: 'G3',
            81: 'L0', 82: 'G0', 84: 'G1', 85: 'G1', 86: 'G3', 87: 'G2',
            88: 'H0', 90: 'G0', 91: 'G0', 92: 'G2', 93: 'G2', 94: 'G6',
            95: 'G4', 96: 'G1', 97: 'H0', 98: 'H0', 99: 'G1'}


def slp_branch_cts_gas():
    """
    This dictionary assignes a gas load profile to every CTS branch
    """
    return {1: 'GB', 2: 'GB', 3: 'GB', 36: 'MF', 37: 'MF', 38: 'BD', 39: 'BD',
            41: 'MK', 42: 'MK', 43: 'MK', 45: 'MK', 46: 'HA', 47: 'HA',
            49: 'BD', 50: 'GA', 51: 'GA', 52: 'BD', 53: 'KO', 55: 'BH',
            56: 'GA', 58: 'PD', 59: 'BD', 60: 'KO', 61: 'BD', 62: 'BD',
            63: 'BD', 64: 'KO', 65: 'KO', 66: 'KO', 68: 'BD', 69: 'BD',
            70: 'BD', 71: 'BD', 72: 'KO', 73: 'KO', 74: 'BD', 75: 'BD',
            77: 'BD', 78: 'KO', 79: 'BD', 80: 'BD', 81: 'BD', 82: 'BD',
            84: 'KO', 85: 'KO', 86: 'BH', 87: 'KO', 88: 'MF', 90: 'BD',
            91: 'KO', 92: 'BD', 93: 'KO', 94: 'KO', 95: 'MK', 96: 'BD',
            97: 'MF', 98: 'MF', 99: 'KO'}


def shift_profile_industry():
    """
    This dictionary assignes a shift profile to every industry branch
    """
    return {5: 'S3_WT_SA', 6: 'S3_WT_SA_SO', 7: 'S3_WT_SA', 8: 'S3_WT_SA',
            9: 'S3_WT_SA', 10: 'S2_WT', 11: 'S3_WT', 12: 'S3_WT_SA',
            13: 'S2_WT', 14: 'S2_WT', 15: 'S2_WT_SA', 16: 'S2_WT_SA',
            17: 'S3_WT_SA_SO', 18: 'S3_WT_SA_SO', 19: 'S3_WT_SA_SO',
            20: 'S3_WT_SA_SO', 21: 'S3_WT_SA_SO', 22: 'S2_WT_SA',
            23: 'S3_WT_SA_SO', 24: 'S3_WT_SA_SO', 25: 'S3_WT', 26: 'S2_WT',
            27: 'S2_WT_SA', 28: 'S2_WT', 29: 'S3_WT', 30: 'S3_WT_SA_SO',
            31: 'S1_WT_SA', 32: 'S3_WT_SA_SO', 33: 'S2_WT_SA'}


def gas_load_profile_parameters_dict():
    """
    This dictionary assignes parameters to gas load profiles
    """
    return {'A': {'BA': 0.277008711731108, 'BD': 1.4633681573375,
                  'BH': 0.987428301992787, 'GA': 1.15820816823062,
                  'GB': 1.82137779524266, 'HA': 1.97247753750471,
                  'KO': 1.35545152289308, 'MF': 1.23286546541232,
                  'MK': 1.42024191542431, 'PD': 1.71107392562331,
                  'WA': 0.333783832123808},
            'B': {'BA': -33.0, 'BD': -36.17941165, 'BH': -35.25321235,
                  'GA': -36.28785839, 'GB': -37.5, 'HA': -36.96500652,
                  'KO': -35.14125631, 'MF': -34.72136051, 'MK': -34.88061302,
                  'PD': -35.8, 'WA': -36.02379115},
            'C': {'BA': 5.72123025, 'BD': 5.926516165, 'BH': 6.154440641,
                  'GA': 6.588512639, 'GB': 6.346214795, 'HA': 7.225694671,
                  'KO': 7.130339509, 'MF': 5.816430402, 'MK': 6.595189922,
                  'PD': 8.4, 'WA': 4.866274683},
            'D': {'BA': 0.4865118291885, 'BD': 0.0808834761578303,
                  'BH': 0.226571574644788, 'GA': 0.223568019279065,
                  'GB': 0.0678117914984112, 'HA': 0.0345781570412447,
                  'KO': 0.0990618615825365, 'MF': 0.0873351930206002,
                  'MK': 0.038531702714089, 'PD': 0.0702545839208687,
                  'WA': 0.491227957971774},
            'mH': {'BA': -0.00948491309440127, 'BD': -0.047579990370696,
                   'BH': -0.0339019728779373, 'GA': -0.0410334784248699,
                   'GB': -0.0607665689685263, 'HA': -0.0742174022298938,
                   'KO': -0.0526486914295292, 'MF': -0.0409283994003907,
                   'MK': -0.0521084240793636, 'PD': -0.0745381134111297,
                   'WA': -0.0092263492839078},
            'bH': {'BA': 0.463023693687715, 'BD': 0.82307541850402,
                   'BH': 0.693823369584483, 'GA': 0.752645138542657,
                   'GB': 0.930815856582958, 'HA': 1.04488686764057,
                   'KO': 0.862608575142234, 'MF': 0.767292039450741,
                   'MK': 0.864791873696473, 'PD': 1.04630053886108,
                   'WA': 0.45957571089625},
            'mW': {'BA': -0.000713418600565782, 'BD': -0.00192725690584626,
                   'BH': -0.00128490078017325, 'GA': -0.000908768552979623,
                   'GB': -0.00139668882761774, 'HA': -0.000829544720239446,
                   'KO': -0.000880838956026602, 'MF': -0.00223202741619469,
                   'MK': -0.00143692105046127, 'PD': -0.000367207932817838,
                   'WA': -0.000967642449895133},
            'bW': {'BA': 0.386744669887959, 'BD': 0.107704598925155,
                   'BH': 0.202973165694549, 'GA': 0.191664070308203,
                   'GB': 0.0850398799492811, 'HA': 0.0461794912976014,
                   'KO': 0.0964014193937084, 'MF': 0.119920720218609,
                   'MK': 0.0637601910393071, 'PD': 0.0621882262236128,
                   'WA': 0.396429075178636},
            'MO': {'BA': 1.0848, 'BD': 1.1052, 'BH': 0.9767, 'GA': 0.9322,
                   'GB': 0.9897, 'HA': 1.0358, 'KO': 1.0354, 'MF': 1.0354,
                   'MK': 1.0699, 'PD': 1.0214, 'WA': 1.2457},
            'DI': {'BA': 1.1211, 'BD': 1.0857, 'BH': 1.0389, 'GA': 0.9894,
                   'GB': 0.9627, 'HA': 1.0232, 'KO': 1.0523, 'MF': 1.0523,
                   'MK': 1.0365, 'PD': 1.0866, 'WA': 1.2615},
            'MI': {'BA': 1.0769, 'BD': 1.0378, 'BH': 1.0028, 'GA': 1.0033,
                   'GB': 1.0507, 'HA': 1.0252, 'KO': 1.0449, 'MF': 1.0449,
                   'MK': 0.9933, 'PD': 1.072, 'WA': 1.2707},
            'DO': {'BA': 1.1353, 'BD': 1.0622, 'BH': 1.0162, 'GA': 1.0109,
                   'GB': 1.0552, 'HA': 1.0295, 'KO': 1.0494, 'MF': 1.0494,
                   'MK': 0.9948, 'PD': 1.0557, 'WA': 1.243},
            'FR': {'BA': 1.1402, 'BD': 1.0266, 'BH': 1.0024, 'GA': 1.018,
                   'GB': 1.0297, 'HA': 1.0253, 'KO': 0.9885, 'MF': 0.9885,
                   'MK': 1.0659, 'PD': 1.0117, 'WA': 1.1276},
            'SA': {'BA': 0.4852, 'BD': 0.7629, 'BH': 1.0043, 'GA': 1.0356,
                   'GB': 0.9767, 'HA': 0.9675, 'KO': 0.886, 'MF': 0.886,
                   'MK': 0.9362, 'PD': 0.9001, 'WA': 0.3877},
            'SO': {'BA': 0.9565000000000001, 'BD': 0.9196,
                   'BH': 0.9587000000000012, 'GA': 1.0106000000000002,
                   'GB': 0.9352999999999998, 'HA': 0.8935000000000004,
                   'KO': 0.9434999999999993, 'MF': 0.9434999999999993,
                   'MK': 0.9033999999999995, 'PD': 0.8524999999999991,
                   'WA': 0.4638}}
