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

from .data import (elc_consumption_HH, households_per_size,
                   standard_load_profile, zve_percentages_applications,
                   zve_percentages_baseload, zve_application_profiles,
                   database_shapes, transpone_spatiotemporal)
import numpy as np
import pandas as pd
import xarray as xa
import geopandas as gpd
import logging
logger = logging.getLogger(__name__)


def disagg_temporal(spat, temp, time_indexed=False, **kwargs):
    """
    Disagreggate spatial data temporally through one or a set of time series.

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
            return temp.multiply(spat, axis=0)
        else:
            return temp.multiply(spat, axis=0).T
    else:
        raise ValueError('`spat` must be either a pd.Series or a '
                         'one-level-indexed (!) pd.DataFrame.')


def create_load_profile():
    raise NotImplementedError('This function is not ready to use!')
    # Number of households per size
    xr_HH = households_per_size().to_xarray().to_array()
    n_HH = xr_HH.shape[0]
    # Specific electricity consumption by household size
    xr_elc_cons_HH = elc_consumption_HH(by_HH_size=True).to_xarray().to_array()
    # standard load profile
    df_slp = standard_load_profile()
    # ...
    df_perc_app = zve_percentages_applications()
    # ...
    df_app_prof = zve_application_profiles()
    # ...
    df_perc_baseload = zve_percentages_baseload()

    DE = database_shapes()
    assert(isinstance(DE, gpd.GeoDataFrame))
    # Derive lat/lon tuple as representative point for each shape
    DE['coords'] = DE.to_crs({'init': 'epsg:4326'}).geometry.apply(
            lambda x: x.representative_point().coords[:][0])
    return


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
    # Sonnenrise / sunset at central local time (MOZ)
    sunset_MOZ = 12 + time_diff - time_equation
    sunrise_MOZ = 12 - time_diff - time_equation
    # Difference between MOZ und MEZ/MESZ
    MOZ_MEZ_diff = -lon/15.0 + UTC_diff
    # Time in MEZ/MESZ
    sunset = sunset_MOZ + MOZ_MEZ_diff
    sunrise = sunrise_MOZ + MOZ_MEZ_diff
    return sunset, sunrise
