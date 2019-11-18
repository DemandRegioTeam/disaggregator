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


def _data(fn):
    return os.path.join(os.path.dirname(__file__), fn)


def _data_out(*fn):
    dirpath = os.path.join(os.path.dirname(__file__), '..', 'data_out')
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)
    return os.path.join(dirpath, *fn)


def _data_in(*fn):
    return os.path.join(os.path.dirname(__file__), '..', 'data_in', *fn)


def get_config(filename=None, **kwargs):
    if filename is None:
        filename = _data('config.yaml')
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
    if not os.path.isdir(_data_in('__cache__/')):
        os.mkdir(_data_in('__cache__/'))

    # If file has already been cached, read cache, else query from API + save.
    filename = _data_in('__cache__/{}.csv'.format(sha1))
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
    if not os.path.isdir(_data_in('__cache__/')):
        os.mkdir(_data_in('__cache__/'))

    deleted = False
    for the_file in os.listdir(_data_in('__cache__')):
        file_path = os.path.join(_data_in('__cache__'), the_file)
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


def region_id_to_nuts3(raw=False, nuts3_to_name=False):
    """
    Read and return a dictionary with regional information.

    Parameters
    ----------
    raw : bool, default False
        If True, return untouched pd.DataFrame.
        If False (default), return dict.
    nuts3_to_name : bool, default False
        Determines the dict data to be returned.
        If True,  return <region_id>: <nuts3> e.g. '1001000': 'DEF01'
        If False, return <nuts3>: <name> e.g. 'DEF01': 'Flensburg, Kreisfre...'

    Returns
    -------
    dict or pd.DataFrame
    """
    dict_source = get_config()['dict_source']
    if dict_source == 'local':
        df = pd.read_csv(_data_in('regional/t_nuts3_lk.csv'), encoding='utf-8')
    else:
        df = database_raw('t_nuts3_lk')
    if raw:
        return df
    else:
        if nuts3_to_name:  # e.g. 'DEF01': 'Flensburg, Kreisfreie Stadt'
            return df.set_index('natcode_nuts3').loc[:, 'name'].to_dict()
        else:              # e.g. '1001000': 'DEF01'
            return df.set_index('id_ags').loc[:, 'natcode_nuts3'].to_dict()


def literal_converter(val):
    try:
        return lit_eval(val)
    except (SyntaxError, ValueError):
        return val
