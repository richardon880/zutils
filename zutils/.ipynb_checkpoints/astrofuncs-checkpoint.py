"""
Import modules
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import mastcasjobs
import os
import getpass
import io
import healpy as hp
from urllib.parse import urlencode, quote
from itertools import pairwise
from PIL import Image
from astropy import units as u
import astropy.cosmology.units as cu
from astropy.cosmology import Planck18
from astropy.coordinates import SkyCoord, Angle, match_coordinates_sky
from astropy.table import Table
from astropy.io import fits
from astroquery.vizier import Vizier

Vizier.ROW_LIMIT = -1




def app_to_abs_mag(distances, app_magnitudes):
    """
    distances: array of distances in Mpc
    app_magnitudes: apparent kron mags
    returns array of abs mags same len as app_magnitudes
    """
    # 5 - 5*np.log(distances*10e6) + magnitudes
    abs_magnitudes = app_magnitudes - (5*np.log10(distances*1e6) - 5)
    return abs_magnitudes

def z_to_distance(redshifts):
    """
    redhisfts - arrray of values
    returns array of distances in Mpc.
    """
    redshifts = redshifts * cu.redshift
    distances = redshifts.to(u.Mpc, cu.redshift_distance(Planck18, kind="comoving")).value
    return distances


def make_ps_query(shape, ra, ra2, dec, dec2, table_name=None):
    query = """SELECT s.objID, s.raMean, s.decMean, s.nDetections, s.primaryDetection,
            gPSFMag, gApMag, gKronMag, gpsfChiSq, gKronRad, gExtNSigma,
            rPSFMag, rApMag, rKronMag, rpsfChiSq, rKronRad, rExtNSigma,
            iPSFMag, iApMag, iKronMag, ipsfChiSq, iKronRad, iExtNSigma,
            zPSFMag, zApMag, zKronMag, zpsfChiSq, zKronRad, zExtNSigma,
            yPSFMag, yApMag, yKronMag, ypsfChiSq, yKronRad, yExtNSigma """

    if shape == "rect":
        query += f"from fGetObjFromRect({ra},{ra2},{dec},{dec2}) nb \
            inner join StackObjectView s on s.objid=nb.objid and s.primaryDetection=1 "

    if shape == "circ":
        query += f"from fGetNearbyObjEq({ra},{dec},{rad}) nb \
            inner join StackObjectView s on s.objid=nb.objid and s.primaryDetection=1 "

    if table_name != None and type(table_name) == str:
        query += f"INTO {table_name}"

    return query


"""
___________________________________________________________
Skymap Object for hpx files
___________________________________________________________
"""


class SkyMap:

    sky_area = 4 * 180**2 / np.pi

    def __init__(self, hpx):
        self.hpx = hpx
        self.npix = len(hpx)
        self.nside = hp.npix2nside(self.npix)
        self.pixarea = hp.nside2pixarea(self.nside, degrees=True)
        self.pixdensity = 1 / self.pixarea
        self.index = np.arange(0, self.npix)
        self.sorted_index = self._sort_index()
        self.sorted_credible_levels = self.get_sorted_credible_levels()
        self.credible_levels = self.get_credible_levels()

    def _sort_index(self):
        sorted_index = np.flipud(
            np.argsort(self.hpx)
        )  # return index of values from highest to lowest
        return sorted_index

    def get_sorted_credible_levels(self):
        sorted_credible_levels = np.cumsum(
            self.hpx[self.sorted_index]
        )  # cumulative sum of values in that order
        return sorted_credible_levels

    def get_credible_levels(self):
        credible_levels = np.empty_like(
            self.sorted_credible_levels
        )  # new array to store
        credible_levels[self.sorted_index] = (
            self.sorted_credible_levels
        )  # same values, but in original order
        return credible_levels

    def get_conf_int(self, frac=0.9):
        self.pix_mask = (
            self.credible_levels < frac
        )  # bool array if point in sky is above credible level from cumulative sum
        self.pix_inds = self.index[self.pix_mask]  # return index of these valuse
        # return self.pix_mask, self.pix_inds

    def check_in_conf_interval(self, ra_vals, dec_vals):
        if isinstance(ra_vals, float) or isinstance(ra_vals, int):
            ra_vals = [ra_vals]
        if isinstance(dec_vals, float) or isinstance(dec_vals, int):
            dec_vals = [dec_vals]  # check they are list or array like
        theta, phi = self.celest2polar(
            ra_vals, dec_vals
        )  # need to provide coords in polar form
        pixel_locs = hp.ang2pix(self.nside, theta, phi)  # get pixel loc of ra and dec
        in_roi = [
            self.pix_mask[pix] for pix in pixel_locs
        ]  # for each pixel location, check if in conf interval by returning bool mask from pix_mask
        return np.array(in_roi)

    def polar2celest(self, theta, phi):  # in radians
        ra = np.rad2deg(phi)
        dec = np.rad2deg(0.5 * np.pi - theta)
        return ra, dec

    def celest2polar(self, ra, dec):  # in degrees
        theta = 0.5 * np.pi - np.deg2rad(dec)
        phi = np.deg2rad(ra)
        return theta, phi

    def showmap(self):
        # plot mollweide view of hpx map
        hp.mollview(self.hpx)
        plt.show()


"""
___________________________________________________________
FUNCTION TO CHECK IF SOURCE IS INSIDE GALAXY
___________________________________________________________
"""


def check_in_ellipse(x, y, center_x, center_y, semi_major, semi_minor, angle, tol=1):
    """
    Function to check if point or array of points are in a defined ellipse

    x = ra of points
    y = dec of points
    center_x = ra of ellipse center
    center_y = dec of ellipse center
    semi_major = semi major axis in deg
    semi_minor = semi minor axis in deg
    angle = rotated angle of ellipse
    tol = i.e. 1.1 adds 10% of radius size to axis as a
          "fudge factor"

    returns array of bools for whether or not is in ellipse
    """
    angle = np.pi / 2 + np.deg2rad(angle)
    # turn points into skycoords this works for a list or single point for testpoints
    center = SkyCoord(ra=center_x * u.deg, dec=center_y * u.deg)
    testpoint = SkyCoord(ra=x * u.deg, dec=y * u.deg)
    # transform the test_points to a ref frame with center of frame at the ellipse center
    testpoint_trans = testpoint.transform_to(center.skyoffset_frame())
    # getting the diff values, the lat and lon are the differences since the center is ell
    dx = testpoint_trans.lon.to(u.deg).value
    dy = testpoint_trans.lat.to(u.deg).value
    # including rotation of ellipse
    x_rot = np.cos(angle) * dx - np.sin(angle) * dy
    y_rot = np.sin(angle) * dx + np.cos(angle) * dy
    semi_major = (semi_major / 2) * tol
    semi_minor = (semi_minor / 2) * tol
    # get array of inside ellipse
    is_inside = (x_rot**2 / semi_major**2 + y_rot**2 / semi_minor**2) < 1
    # return result
    return is_inside


"""
___________________________________________________________
CATALOGUE SEARCH FUNCTIONS FOR VIZIER
___________________________________________________________
"""


def search_hleda(ra, dec, rad):
    co_ord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
    rad = rad * u.deg
    catalog = "VII/237/pgc"
    result = Vizier.query_region(co_ord, radius=rad, catalog=catalog, frame="icrs")

    cols = [
        "PGC",
        "RA",
        "DEC",
        "RAJ2000",
        "DEJ2000",
        "OType",
        "MType",
        "logD25",
        "logR25",
        "PA",
        "ANames",
    ]

    if len(result) == 0:
        return pd.DataFrame(columns=cols)
    else:
        result = result[0].to_pandas()
    result["RA"] = np.zeros(len(result))
    result["DEC"] = np.zeros(len(result))
    for idx, row in result.iterrows():
        result.at[idx, "RA"] = Angle(
            row.RAJ2000.split()[0]
            + "h"
            + row.RAJ2000.split()[1]
            + "m"
            + row.RAJ2000.split()[2]
            + "s"
        ).deg

        result.at[idx, "DEC"] = Angle(
            row.DEJ2000.split()[0]
            + "d"
            + row.DEJ2000.split()[1]
            + "m"
            + row.DEJ2000.split()[2]
            + "s"
        ).deg

    result = result[cols]
    result["semi_major"] = 0.1 * (10 ** (result.logD25)) / 60  # in degrees
    result["R25"] = 10 ** (result.logR25)
    result["semi_minor"] = result.semi_major / result.R25  # in degrees

    return result


def square_search_hleda(ra, dec, width, height):
    co_ord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
    width = width * u.degree
    height = height * u.degree
    catalog = "VII/237/pgc"
    result = Vizier.query_region(
        co_ord, width=width, height=height, catalog=catalog, frame="icrs"
    )

    cols = [
        "PGC",
        "RA",
        "DEC",
        "RAJ2000",
        "DEJ2000",
        "OType",
        "MType",
        "logD25",
        "logR25",
        "PA",
        "ANames",
    ]

    if len(result) == 0:
        return pd.DataFrame(columns=cols)
    else:
        result = result[0].to_pandas()

    result["RA"] = np.zeros(len(result))
    result["DEC"] = np.zeros(len(result))
    for idx, row in result.iterrows():
        result.at[idx, "RA"] = Angle(
            row.RAJ2000.split()[0]
            + "h"
            + row.RAJ2000.split()[1]
            + "m"
            + row.RAJ2000.split()[2]
            + "s"
        ).deg

        result.at[idx, "DEC"] = Angle(
            row.DEJ2000.split()[0]
            + "d"
            + row.DEJ2000.split()[1]
            + "m"
            + row.DEJ2000.split()[2]
            + "s"
        ).deg

    result = result[cols]
    result["semi_major"] = 0.1 * (10 ** (result.logD25)) / 60  # in degrees
    result["R25"] = 10 ** (result.logR25)
    result["semi_minor"] = result.semi_major / result.R25  # in degrees

    return result


def square_query_HLEDA(ra, dec, width, height):
    ra = Angle(ra, u.degree).hour
    dec = Angle(dec, u.degree).hour
    width = Angle(width, u.degree).hour
    height = Angle(height, u.degree).hour
    # print(f"ra{ra}, dec{dec}, width{width}, height{height}")
    sql = f"objtype='G' and al2000 > {ra} and al2000 < {ra+width} and de2000 > {dec} and de2000 < {dec+height}"
    base_url = "http://atlas.obs-hp.fr/hyperleda/fG.cgi"
    params = {
        "n": "meandata",
        "c": "o",
        "of": "1,leda,simbad,pgc",
        "nra": "l",
        "nakd": "1",
        "sql": sql,
        "ob": "",
        "a": "t[,]",
    }
    columns = [
        "objname",
        "pgc",
        "ob",
        "al1950",
        "de1950",
        "al2000",
        "de2000",
        "l2",
        "b2",
        "sgl",
        "sgb",
        "f_astr",
        "type",
        "agnclass",
        "logd25",
        "e_logd25",
        "logr25",
        "e_logr25",
        "pa",
        "brief",
        "e_brief",
        "ut",
        "e_ut",
        "bt",
        "e_bt",
        "vt",
        "e_vt",
        "it",
        "e_it",
        "kt",
        "e_kt",
        "m21",
        "e_m21",
        "mfir",
        "ube",
        "bve",
        "vmaxg",
        "e_vmaxg",
        "vmaxs",
        "e_vmaxs",
        "vdis",
        "e_vdis",
        "vrad",
        "e_vrad",
        "vopt",
        "e_vopt",
        "v",
        "e_v",
        "ag",
        "ai",
        "incl",
        "a21",
        "logdc",
        "btc",
        "itc",
        "ubtc",
        "bvtc",
        "bri25",
        "vrot",
        "e_vrot",
        "mg2",
        "e_mg2",
        "m21c",
        "hic",
        "vlg",
        "vgsr",
        "vvir",
        "v3k",
        "modz",
        "e_modz",
        "mod0",
        "e_mod0",
        "modbest",
        "e_modbes",
        "mabs",
        "e_mabs",
    ]
    col_dtype = {
        "objname": str,
        "pgc": float,
        "ob": str,
        "al1950": float,
        "de1950": float,
        "al2000": float,
        "de2000": float,
        "l2": float,
        "b2": float,
        "sgl": float,
        "sgb": float,
        "f_astr": bool,
        "type": str,
        "agnclass": str,
        "logd25": float,
        "e_logd25": float,
        "logr25": float,
        "e_logr25": float,
        "pa": float,
        "brief": str,
        "e_brief": str,
        "ut": float,
        "e_ut": float,
        "bt": float,
        "e_bt": float,
        "vt": float,
        "e_vt": float,
        "it": float,
        "e_it": float,
        "kt": float,
        "e_kt": float,
        "m21": float,
        "e_m21": float,
        "mfir": float,
        "ube": float,
        "bve": float,
        "vmaxg": float,
        "e_vmaxg": float,
        "vmaxs": float,
        "e_vmaxs": float,
        "vdis": float,
        "e_vdis": float,
        "vrad": float,
        "e_vrad": float,
        "vopt": float,
        "e_vopt": float,
        "v": float,
        "e_v": float,
        "ag": float,
        "ai": float,
        "incl": float,
        "a21": float,
        "logdc": float,
        "btc": float,
        "itc": float,
        "ubtc": float,
        "bvtc": float,
        "bri25": float,
        "vrot": float,
        "e_vrot": float,
        "mg2": float,
        "e_mg2": float,
        "m21c": float,
        "hic": float,
        "vlg": float,
        "vgsr": float,
        "vvir": float,
        "v3k": float,
        "modz": float,
        "e_modz": float,
        "mod0": float,
        "e_mod0": float,
        "modbest": float,
        "e_modbes": float,
        "mabs": float,
        "e_mabs": float,
    }

    encoded_params = urlencode(params, quote_via=quote)
    full_url = f"{base_url}?{encoded_params}"

    # check if has returned results
    res = requests.get(full_url)
    search_info = res._content.split(b"#")
    if b" The complete request returns no record.\n" in search_info:
        print("No records found for search region.")
        result = pd.DataFrame(columns=columns)
        result["Z"] = pd.Series(dtype="float")
        result["RA"] = pd.Series(dtype="float")
        result["DEC"] = pd.Series(dtype="float")
        result["semi_major"] = pd.Series(dtype="float")
        result["R25"] = pd.Series(dtype="float")
        result["semi_minor"] = pd.Series(dtype="float")
        return result
    # read the data
    df = pd.read_csv(full_url, header=88, skipfooter=5, engine="python")[
        11:
    ].reset_index(drop=True)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"#$objname": "objname"}, errors="raise")
    for colname in df.columns:
        df[colname] = df[colname].str.strip()
    df = df.replace("", np.NaN)
    df = df.drop(["b", "r", "m", "c", "t", "e_t"], axis=1)
    result = df.astype(col_dtype)

    # process result
    result["Z"] = result.vvir / 3e5
    result["RA"] = Angle(result.al2000, u.hour).degree
    result["DEC"] = Angle(result.de2000, u.hour).degree
    result = result.rename(columns={"pa": "PA"})
    result["semi_major"] = 0.1 * (10 ** (result.logd25)) / 60  # in degrees
    result["R25"] = 10 ** (result.logr25)
    result["semi_minor"] = result.semi_major / result.R25  # in degrees

    return result


def square_search_apass(ra, dec, width, height):
    co_ord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
    width = width * u.degree
    height = height * u.degree
    catalog = "apass"
    result = Vizier.query_region(
        co_ord, width=width, height=height, catalog=catalog, frame="icrs"
    )
    cols = [
        "RAJ2000",
        "DEJ2000",
        "e_RAJ2000",
        "e_DEJ2000",
        "Field",
        "nobs",
        "mobs",
        "B-V",
        "e_B-V",
        "Vmag",
        "e_Vmag",
        "Bmag",
        "e_Bmag",
        "g_mag",
        "e_g_mag",
        "r_mag",
        "e_r_mag",
        "i_mag",
        "e_i_mag",
    ]
    if len(result) == 0:
        return pd.DataFrame(columns=cols)
    else:
        # print(len(result[0].to_pandas()))
        # print(result[0].to_pandas().columns)
        result = result[0].to_pandas().drop("recno", axis=1, errors="ignore")
    return result


def search_apass(ra, dec, rad):
    co_ord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
    rad = rad * u.deg
    catalog = "apass"
    result = Vizier.query_region(co_ord, radius=rad, catalog=catalog, frame="icrs")
    cols = [
        "RAJ2000",
        "DEJ2000",
        "e_RAJ2000",
        "e_DEJ2000",
        "Field",
        "nobs",
        "mobs",
        "B-V",
        "e_B-V",
        "Vmag",
        "e_Vmag",
        "Bmag",
        "e_Bmag",
        "g_mag",
        "e_g_mag",
        "r_mag",
        "e_r_mag",
        "i_mag",
        "e_i_mag",
    ]
    if len(result) == 0:
        return pd.DataFrame(columns=cols)
    else:
        result = result[0].to_pandas().drop("recno", axis=1)
    return result


def search_tycho(ra, dec, rad):
    co_ord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
    rad = rad * u.deg
    catalog = "tyc2"
    result = Vizier.query_region(co_ord, radius=rad, catalog=catalog, frame="icrs")
    cols = [
        "RA_ICRS_",
        "DE_ICRS_",
        "pmRA",
        "pmDE",
        "BTmag",
        "VTmag",
        "HIP",
        "TYC1",
        "TYC2",
        "TYC3",
    ]
    if len(result) == 0:
        return pd.DataFrame(columns=cols)
    else:
        result = result[0].to_pandas()
    result = result[cols]
    return result


def square_search_tycho(ra, dec, width, height):
    co_ord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
    width = width * u.degree
    height = height * u.degree
    catalog = "tyc2"
    result = Vizier.query_region(
        co_ord, width=width, height=height, catalog=catalog, frame="icrs"
    )
    cols = [
        "RA_ICRS_",
        "DE_ICRS_",
        "pmRA",
        "pmDE",
        "BTmag",
        "VTmag",
        "HIP",
        "TYC1",
        "TYC2",
        "TYC3",
    ]
    if len(result) == 0:
        return pd.DataFrame(columns=cols)
    else:
        result = result[0].to_pandas()
    result = result[cols]
    return result


def search_bright_stars(ra, dec, rad):

    apass = search_apass(ra, dec, rad)
    tycho = search_tycho(ra, dec, rad)

    cols = [
        "key_0",
        "AP_RA",
        "AP_DEC",
        "e_RAJ2000",
        "e_DEJ2000",
        "Field",
        "nobs",
        "mobs",
        "B-V",
        "e_B-V",
        "Vmag",
        "e_Vmag",
        "Bmag",
        "e_Bmag",
        "g_mag",
        "e_g_mag",
        "r_mag",
        "e_r_mag",
        "i_mag",
        "e_i_mag",
        "TY_RA",
        "TY_DEC",
        "pmRA",
        "pmDE",
        "BTmag",
        "VTmag",
        "HIP",
        "TYC1",
        "TYC2",
        "TYC3",
        "RA",
        "Dec",
    ]

    if len(apass) == 0 and len(tycho) == 0:
        return pd.DataFrame(columns=cols)

    elif len(apass) == 0:
        bright_stars = tycho.rename(columns={"RA_ICRS_": "RA", "DE_ICRS_": "Dec"})
        bright_stars["duplicated"] = bright_stars.duplicated(subset=["RA", "Dec"])
        dupes = bright_stars.query("duplicated == True").index
        bright_stars = bright_stars.drop(dupes)
        return bright_stars

    elif len(tycho) == 0:
        bright_stars = apass.rename(columns={"RAJ2000": "RA", "DEJ2000": "Dec"})
        bright_stars["duplicated"] = bright_stars.duplicated(subset=["RA", "Dec"])
        dupes = bright_stars.query("duplicated == True").index
        bright_stars = bright_stars.drop(dupes)
        return bright_stars

    else:

        # tycho_match = SkyCoord(ra=tycho.RA_ICRS_*u.degree, dec=tycho.DE_ICRS_*u.degree)
        tycho_match = SkyCoord(ra=tycho.RA_ICRS_, dec=tycho.DE_ICRS_, unit=u.degree)

        # apass_match = SkyCoord(ra=apass.RAJ2000*u.degree, dec=apass.DEJ2000*u.degree)
        apass_match = SkyCoord(ra=apass.RAJ2000, dec=apass.DEJ2000, unit=u.degree)

        idx, sep2d, dist3d = match_coordinates_sky(
            tycho_match, apass_match
        )  # match results

        tycho = tycho.set_index(idx)
        bright_stars = pd.merge(
            apass, tycho, left_index=True, right_on=tycho.index, how="left"
        ).reset_index(drop=True)

        # bright_stars["duplicated"] = bright_stars.duplicated(subset=["RAJ2000","DEJ2000"])
        # dupes = bright_stars.query("duplicated == True").index
        # bright_stars = bright_stars.drop(dupes)

        bright_stars = bright_stars.rename(
            columns={
                "RA_ICRS_": "TY_RA",
                "DE_ICRS_": "TY_DEC",
                "RAJ2000": "AP_RA",
                "DEJ2000": "AP_DEC",
            }
        )

        bright_stars["RA"] = np.where(
            np.isnan(bright_stars.AP_RA), bright_stars.TY_RA, bright_stars.AP_RA
        )
        bright_stars["Dec"] = np.where(
            np.isnan(bright_stars.AP_DEC), bright_stars.TY_DEC, bright_stars.AP_DEC
        )

        bright_stars["duplicated"] = bright_stars.duplicated(subset=["RA", "Dec"])
        dupes = bright_stars.query("duplicated == True").index
        bright_stars = bright_stars.drop("duplicated", axis=1)

    return bright_stars


def square_search_bright_stars(ra, dec, width, height):

    apass = square_search_apass(ra, dec, width, height)
    tycho = square_search_tycho(ra, dec, width, height)

    cols = [
        "key_0",
        "AP_RA",
        "AP_DEC",
        "e_RAJ2000",
        "e_DEJ2000",
        "Field",
        "nobs",
        "mobs",
        "B-V",
        "e_B-V",
        "Vmag",
        "e_Vmag",
        "Bmag",
        "e_Bmag",
        "g_mag",
        "e_g_mag",
        "r_mag",
        "e_r_mag",
        "i_mag",
        "e_i_mag",
        "TY_RA",
        "TY_DEC",
        "pmRA",
        "pmDE",
        "BTmag",
        "VTmag",
        "HIP",
        "TYC1",
        "TYC2",
        "TYC3",
        "RA",
        "Dec",
    ]

    if len(apass) == 0 and len(tycho) == 0:
        return pd.DataFrame(columns=cols)

    elif len(apass) == 0:
        bright_stars = tycho.rename(columns={"RA_ICRS_": "RA", "DE_ICRS_": "Dec"})
        bright_stars["duplicated"] = bright_stars.duplicated(subset=["RA", "Dec"])
        dupes = bright_stars.query("duplicated == True").index
        bright_stars = bright_stars.drop(dupes)
        return bright_stars

    elif len(tycho) == 0:
        bright_stars = apass.rename(columns={"RAJ2000": "RA", "DEJ2000": "Dec"})
        bright_stars["duplicated"] = bright_stars.duplicated(subset=["RA", "Dec"])
        dupes = bright_stars.query("duplicated == True").index
        bright_stars = bright_stars.drop(dupes)
        return bright_stars

    else:

        tycho_match = SkyCoord(
            ra=tycho.RA_ICRS_ * u.degree, dec=tycho.DE_ICRS_ * u.degree, unit=u.degree
        )
        apass_match = SkyCoord(
            ra=apass.RAJ2000 * u.degree, dec=apass.DEJ2000 * u.degree, unit=u.degree
        )

        idx, sep2d, dist3d = match_coordinates_sky(
            tycho_match, apass_match
        )  # match results

        tycho = tycho.set_index(idx)
        bright_stars = pd.merge(
            apass, tycho, left_index=True, right_on=tycho.index, how="left"
        ).reset_index(drop=True)

        # bright_stars["duplicated"] = bright_stars.duplicated(subset=["RAJ2000","DEJ2000"])
        # dupes = bright_stars.query("duplicated == True").index
        # bright_stars = bright_stars.drop(dupes)

        bright_stars = bright_stars.rename(
            columns={
                "RA_ICRS_": "TY_RA",
                "DE_ICRS_": "TY_DEC",
                "RAJ2000": "AP_RA",
                "DEJ2000": "AP_DEC",
            }
        )

        bright_stars["RA"] = np.where(
            np.isnan(bright_stars.AP_RA), bright_stars.TY_RA, bright_stars.AP_RA
        )
        bright_stars["Dec"] = np.where(
            np.isnan(bright_stars.AP_DEC), bright_stars.TY_DEC, bright_stars.AP_DEC
        )

        bright_stars["duplicated"] = bright_stars.duplicated(subset=["RA", "Dec"])
        dupes = bright_stars.query("duplicated == True").index
        bright_stars = bright_stars.drop("duplicated", axis=1)

        return bright_stars


"""
___________________________________________________________
CATALOG SEARCH FUNCTIONS FOR MAST CASJOBS
___________________________________________________________
"""


# call this to sign in to enable mastcasjobs functions
def mastcasjobs_init():
    if not os.environ.get("CASJOBS_USERID"):
        os.environ["CASJOBS_USERID"] = input("Enter Casjobs username:")
    if not os.environ.get("CASJOBS_PW"):
        os.environ["CASJOBS_PW"] = getpass.getpass("Enter Casjobs password:")


def mastQuery(request, json_return=False):
    """Perform a MAST query.

    Parameters
    ----------
    request (dictionary): The MAST request json object

    Returns the text response or (if json_return=True) the json response
    """

    url = "https://mast.stsci.edu/api/v0/invoke"

    # Encoding the request as a json string
    requestString = json.dumps(request)

    # make the query
    r = requests.post(url, data=dict(request=requestString))

    # raise exception on error
    r.raise_for_status()

    if json_return:
        return r.json()
    else:
        return r.text


def resolve(name):
    """Get the RA and Dec for an object using the MAST name resolver

    Parameters
    ----------
    name (str): Name of object

    Returns RA, Dec tuple with position"""

    resolverRequest = {
        "service": "Mast.Name.Lookup",
        "params": {"input": name, "format": "json"},
    }
    resolvedObject = mastQuery(resolverRequest, json_return=True)
    # The resolver returns a variety of information about the resolved object,
    # however for our purposes all we need are the RA and Dec
    try:
        objRa = resolvedObject["resolvedCoordinate"][0]["ra"]
        objDec = resolvedObject["resolvedCoordinate"][0]["decl"]
    except IndexError as e:
        raise ValueError("Unknown object '{}'".format(name))
    return (objRa, objDec)


def search_circ_region(
    ra, dec, rad, table_name=None, task_name="My Query", keep_table=True
):
    mastcasjobs_init()
    rad = rad * 60
    ps_jobs = mastcasjobs.MastCasJobs(context="PanSTARRS_DR2")

    if table_name == None:
        if keep_table == True:
            raise ValueError(
                "table_name must be provided if keep_table is True, stopping search."
            )
        table_name = "_temp"
        ps_jobs.drop_table_if_exists(table_name)

    if table_name in ps_jobs.list_tables():
        raise NameError(f"Table '{table_name}' Already Exists")

    ps_query = f"""SELECT s.objID, s.raMean, s.decMean,
        s.gKronMag, s.gPSFMag, s.gKronMagErr, s.gPSFMagErr, gExtNSigma,
        s.rKronMag, s.rPSFMag, s.rKronMagErr, s.rPSFMagErr, rExtNSigma,
        s.iKronMag, s.iPSFMag, s.iKronMagErr, s.iPSFMagErr, iExtNSigma,
        s.zKronMag, s.zPSFMag, s.zKronMagErr, s.zPSFMagErr, zExtNSigma,
        s.yKronMag, s.yPSFMag, s.yKronMagErr, s.yPSFMagErr, yExtNSigma,
        s.nDetections
        from fGetNearbyObjEq({ra},{dec},{rad}) nb
        inner join StackObjectView s on s.objid=nb.objid and s.primaryDetection=1
        INTO {table_name}
        """

    ps_job_id = ps_jobs.submit(ps_query, task_name=f"{task_name}")
    ps_jobs.monitor(ps_job_id)
    ps_df = ps_jobs.fast_table(table_name).to_pandas()

    if keep_table == False:
        ps_jobs.drop_table_if_exists(table_name)

    return ps_df


def get_nearest_obj_PS1_data(ralist, declist, searchrad):
    indexes = list(range(0, len(ralist)))
    input_coords = ", ".join(
        [f"({idx},{ra}, {dec})" for idx, ra, dec in zip(indexes, ralist, declist)]
    )

    ps_query = f"""
    CREATE TABLE #InputCoords (idx INT, ra FLOAT, dec FLOAT)
    
    INSERT INTO #InputCoords (idx, ra, dec)
    VALUES {input_coords}
    
    SELECT ic.idx, s.objID, s.raMean, s.decMean,
           s.gKronMag, s.gPSFMag, s.gKronMagErr, s.gPSFMagErr, s.gExtNSigma,
           s.rKronMag, s.rPSFMag, s.rKronMagErr, s.rPSFMagErr, s.rExtNSigma,
           s.iKronMag, s.iPSFMag, s.iKronMagErr, s.iPSFMagErr, s.iExtNSigma,
           s.zKronMag, s.zPSFMag, s.zKronMagErr, s.zPSFMagErr, s.zExtNSigma,
           s.yKronMag, s.yPSFMag, s.yKronMagErr, s.yPSFMagErr, s.yExtNSigma,
           s.nDetections, nb.distance
    FROM #InputCoords ic
    CROSS APPLY fGetNearestObjEq(ic.ra, ic.dec, {searchrad}) nb
    INNER JOIN StackObjectView s ON s.objid = nb.objid AND s.primaryDetection = 1
    
    DROP TABLE #InputCoords
    """
    # print(ps_query)
    ps_jobs = mastcasjobs.MastCasJobs(context="PanSTARRS_DR2")
    ps_df = ps_jobs.quick(ps_query, task_name="My Query").to_pandas()
    return ps_df


def quick_search_circ_region(ra, dec, rad, task_name="My Query"):
    rad = rad * 60
    ps_jobs = mastcasjobs.MastCasJobs(context="PanSTARRS_DR2")

    ps_query = f"""SELECT s.objID, s.raMean, s.decMean,
        s.gKronMag, s.gPSFMag, s.gKronMagErr, s.gPSFMagErr, gExtNSigma,
        s.rKronMag, s.rPSFMag, s.rKronMagErr, s.rPSFMagErr, rExtNSigma,
        s.iKronMag, s.iPSFMag, s.iKronMagErr, s.iPSFMagErr, iExtNSigma,
        s.zKronMag, s.zPSFMag, s.zKronMagErr, s.zPSFMagErr, zExtNSigma,
        s.yKronMag, s.yPSFMag, s.yKronMagErr, s.yPSFMagErr, yExtNSigma,
        s.nDetections
        from fGetNearbyObjEq({ra},{dec},{rad}) nb
        inner join StackObjectView s on s.objid=nb.objid and s.primaryDetection=1
        """

    ps_df = ps_jobs.quick(ps_query, task_name=f"{task_name}").to_pandas()

    return ps_df


def retrieve_table(table_name):
    """
    Function to retrieve table name from PS1 mastcasjobs
    """
    ps_jobs = mastcasjobs.MastCasJobs(context="PanSTARRS_DR2")
    ps_df = ps_jobs.fast_table(table_name).to_pandas()
    return ps_df


def search_rect_region(ra1, ra2, dec1, dec2, table_name=None, task_name="My Query"):
    ps_query = make_ps_query(
        "rect", ra=ra1, ra2=ra2, dec=dec1, dec2=dec2, table_name=table_name
    )
    # ps_query = f"""
    #             SELECT s.objID, s.raMean, s.decMean, s.nDetections, s.primaryDetection,
    #             gPSFMag, gApMag, gKronMag, gpsfChiSq, gKronRad, gExtNSigma,
    #             rPSFMag, rApMag, rKronMag, rpsfChiSq, rKronRad, rExtNSigma,
    #             iPSFMag, iApMag, iKronMag, ipsfChiSq, iKronRad, iExtNSigma,
    #             zPSFMag, zApMag, zKronMag, zpsfChiSq, zKronRad, zExtNSigma,
    #             yPSFMag, yApMag, yKronMag, ypsfChiSq, yKronRad, yExtNSigma
    #             from fGetObjFromRect({ra1},{ra2},{dec1},{dec2}) nb
    #             inner join StackObjectView s on s.objid=nb.objid and s.primaryDetection=1
    #             """

    ps_jobs = mastcasjobs.MastCasJobs(context="PanSTARRS_DR2")

    if table_name in ps_jobs.list_tables():
        raise NameError(f"Table '{table_name}' Already Exists")

    if table_name is not None:
        # ps_query = ps_query+f"INTO {table_name}"
        ps_job_id = ps_jobs.submit(ps_query, task_name=f"{task_name}")
        print(ps_query)
        ps_jobs.monitor(ps_job_id)
        ps_df = ps_jobs.fast_table(f"{table_name}").to_pandas()

    elif table_name is None:
        ps_df = ps_jobs.quick(ps_query, task_name=f"{task_name}").to_pandas()

    return ps_df


# custom function
def get_mast_table(table_name):
    return mastcasjobs.MastCasJobs().fast_table(table_name).to_pandas()


def proximity_search(ralist, declist, point, rad):
    """
    Function to search and return all points within a radius of skycoords

    ralist = list of right ascension of points
    declist = list of declinations of points
    point = point the search will be located around, tuple or list
    rad = radius in degrees of search

    returns: indexes of the matching points
    """
    # converting point and ra,dec to skycoords
    point = SkyCoord(point[0] * u.deg, point[1] * u.deg)
    # coords = SkyCoord(ralist*u.deg, declist*u.deg)
    coords = SkyCoord(ralist, declist, unit=u.deg)
    # calculating separation of point vs all coords in this list
    seplist = coords.separation(point)
    # this is a boolean array whether or not each instance is in or out of the radius
    in_rad = seplist.deg <= rad
    # this gets the index of all the nonzero points (i.e. true, inside rad)
    inds_in_rad = in_rad.nonzero()[0]
    return inds_in_rad


"""
___________________________________________________________
FUNCTION TO PLOT CUTOUTS VIA ONLINE QUERY
___________________________________________________________
"""


def getimages(ra, dec, filters="grizy"):
    """Query ps1filenames.py service to get a list of images

    ra, dec = position in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    Returns a table with the results
    """

    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = f"{service}?ra={ra}&dec={dec}&filters={filters}"
    table = Table.read(url, format="ascii")
    return table


def geturl(
    ra, dec, size=240, output_size=None, filters="grizy", format="jpg", color=False
):
    """Get URL for images in the table

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png" or "fits")
    color = if True, creates a color image (only for jpg or png format).
            Default is return a list of URLs for single-filter grayscale images.
    Returns a string with the URL
    """

    if color and format == "fits":
        raise ValueError("color images are available only for jpg or png formats")
    if format not in ("jpg", "png", "fits"):
        raise ValueError("format must be one of jpg, png, fits")
    table = getimages(ra, dec, filters=filters)
    url = (
        f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
        f"ra={ra}&dec={dec}&size={size}&format={format}"
    )
    if output_size:
        url = url + "&output_size={}".format(output_size)
    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table["filter"]]
    table = table[np.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0, len(table) // 2, len(table) - 1]]
        for i, param in enumerate(["red", "green", "blue"]):
            url = url + "&{}={}".format(param, table["filename"][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in table["filename"]:
            url.append(urlbase + filename)
    return url


def getcolorim(ra, dec, size=240, output_size=None, filters="grizy", format="jpg"):
    """Get color image at a sky position

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png")
    Returns the image
    """

    if format not in ("jpg", "png"):
        raise ValueError("format must be jpg or png")
    url = geturl(
        ra,
        dec,
        size=size,
        filters=filters,
        output_size=output_size,
        format=format,
        color=True,
    )
    r = requests.get(url)
    im = Image.open(io.BytesIO(r.content))
    return im


# custom function
def plot_cutouts(ra, dec, nrows=1, ncols=1, size=240, figsize=(6, 6)):
    if type(ra) == pd.core.series.Series:
        ra = ra.reset_index(drop=True)
        dec = dec.reset_index(drop=True)
    else:
        ra = [ra]
        dec = [dec]
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    if ncols * nrows > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    for i in range(nrows * ncols):
        cim = getcolorim(ra[i], dec[i], size=size)
        axs[i].imshow(cim, origin="upper")
        axs[i].tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
        )
        axs[i].set_yticklabels([])
        axs[i].set_xticklabels([])
        axs[i].set_title(f"Loc: {ra[i]}\n{dec[i]}", fontsize=10)
    plt.tight_layout()
    return


def get_fits_image(ra, dec, size=240, filters="r"):
    fitsurl = geturl(ra, dec, size=size, filters=filters, format="fits")
    fh = fits.open(fitsurl[0])
    fim = fh[0].data
    fim[np.isnan(fim)] = 0.0
    return fim


"""
___________________________________________________________

___________________________________________________________
"""


def getImageTable(
    tra, tdec, size=240, filters="grizy", format="fits", imagetypes="stack"
):
    """Query ps1filenames.py service for multiple positions to get a list of images
    This adds a url column to the table to retrieve the cutout.

    tra, tdec = list of positions in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    format = data format (options are "fits", "jpg", or "png")
    imagetypes = list of any of the acceptable image types.  Default is stack;
        other common choices include warp (single-epoch images), stack.wt (weight image),
        stack.mask, stack.exp (exposure time), stack.num (number of exposures),
        warp.wt, and warp.mask.  This parameter can be a list of strings or a
        comma-separated string.

    Returns pandas dataframe with the results
    """
    pd.set_option("display.max_colwidth", 10000)

    ps1filename = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    fitscut = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"

    if format not in ("jpg", "png", "fits"):
        raise ValueError("format must be one of jpg, png, fits")
    # if imagetypes is a list, convert to a comma-separated string
    if not isinstance(imagetypes, str):
        imagetypes = ",".join(imagetypes)

    # put the positions in an in-memory file object
    cbuf = io.StringIO()
    cbuf.write("\n".join(["{} {}".format(ra, dec) for (ra, dec) in zip(tra, tdec)]))
    cbuf.seek(0)
    # use requests.post to pass in positions as a file
    r = requests.post(
        ps1filename, data=dict(filters=filters, type=imagetypes), files=dict(file=cbuf)
    )
    r.raise_for_status()
    tab = Table.read(r.text, format="ascii")

    urlbase = "{}?size={}&format={}".format(fitscut, size, format)
    tab["url"] = [
        "{}&ra={}&dec={}&red={}".format(urlbase, ra, dec, filename)
        for (filename, ra, dec) in zip(tab["filename"], tab["ra"], tab["dec"])
    ]

    tab = tab.to_pandas()
    return tab


def readFitsImage(url):
    """Gets header and image data from fits files retrieved from urls in pandas
    dataframe from function getImageTable. Returns header and image data.
    """
    r = requests.get(url)
    memory_file = io.BytesIO(r.content)

    with fits.open(memory_file) as hdulist:
        data = hdulist[0].data
        header = hdulist[0].header

    return header, data


"""
___________________________________________________________
FUNCTION TO PLOT and PROCESS FLEXCODE MODEL OUTPUTS
___________________________________________________________
"""

# def check_in_roi(cdes, zgrid, max_z, prob, prob_upr=1):
#     inds = list()
#     bin_width = 1/len(zgrid)
#     cde_ind_lim = int(len(zgrid) * max_z)
#     for ind, cde in enumerate(cdes):
#         val = cde[:cde_ind_lim].sum()
#         sumcde = (val * bin_width)
#         if sumcde>prob and sumcde<=prob_upr:
#             inds.append(ind)
#     return inds


def check_in_roi(cdes, zgrid, min_z, max_z, prob, prob_upr=1):
    inds = list()
    bin_width = 1 / len(zgrid)
    cde_ind_max = int(len(zgrid) * max_z)
    cde_ind_min = int(len(zgrid) * min_z)
    for ind, cde in enumerate(cdes):
        val = cde[cde_ind_min:cde_ind_max].sum()
        sumcde = val * bin_width
        if sumcde > prob and sumcde <= prob_upr:
            inds.append(ind)
    return inds


def get_prob_range(cdes, zgrid, min_z, max_z):
    probs = list()
    len_z = len(zgrid)
    bin_width = (max(zgrid)[0] - min(zgrid[0])) / len_z
    cde_ind_zmax = int((max_z - min(zgrid))[0] / bin_width)
    cde_ind_zmin = int((min_z - min(zgrid))[0] / bin_width)
    # print(bin_width, cde_ind_zmax, cde_ind_zmin)
    for ind, cde in enumerate(cdes):
        val = cde[cde_ind_zmin:cde_ind_zmax].sum()
        sumcde = val * bin_width
        probs.append(sumcde)
    return probs


def plot_cdes(cdes, zgrid, shape, start=0, figsize=(12, 12), marker=None):
    nrows = shape[0]
    ncols = shape[1]
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    # could be issue here wehere will not flatten if 1 axs needed only
    axs = axs.flatten()
    for i in range(start, nrows * ncols):
        # axs[i].plot(zgrid, cdes[candidates_ind[i]])
        axs[i].plot(zgrid, cdes[i])
        if marker is not None:
            axs[i].axvline(marker, color="red", alpha=0.5)
    plt.show()


def get_cdfs(cdes, zgrid):
    cdfs = np.zeros((len(cdes), len(zgrid)))
    for ind, cde in enumerate(cdes):
        cumsum = 0
        for i, val in enumerate(cde):
            cumsum += val
            cdfs[ind, i] = cumsum
    return cdfs


def plot_cde_range(cdes, y_test, zgrid, val_range, shape, subset=None, roi=True):
    """
    Function to plot CDES from FlexCode.

    cdes: array
            list of cdes as recieved from flexcode model predictions

    y_test: array MUST BE SERIES OR PANDAS DATAFRAME
            list of target variables corresponding to the passed cdes

    zgrid: listlike
            zgrid as passed from flexcode

    val_range: list/tuple consisting of a range as such -> (start,finish)
            this defines the cdes you want to look at based on their corresponding target var
            i.e. look at cdes where target var between 0 and 0.06 -> (0,0.06)

    shape: tuple, list in form of -> (nrows, ncols)
            the shape the subplots will be plotted into when viewing cdes

    subset: tuple, list like -> (start, finish) default:None
            the subset of all cdes inside val range to plot i.e. the 3rd to 10th cdes

    roi: boolean default:True
            whether or not to plot a vertical green line at 0.06 (this is roi for redshifts we are looking at)
    """
    lwr = val_range[0]
    upr = val_range[1]
    nrows = shape[0]
    ncols = shape[1]

    zVals = y_test.reset_index(drop=True)
    if type(zVals) == pd.core.frame.DataFrame:
        test_inds = zVals.query("z>@lwr & z<@upr").index.values.tolist()
    else:
        zVals = zVals.to_frame()
        test_inds = zVals.query("z>@lwr & z<@upr").index.values.tolist()
    n_tests = len(test_inds)  # list of indices of zVals within target range

    if subset != None:
        # if start point > list no points to plot
        if subset[0] > n_tests:
            raise IndexError("No Values in Range")
            # else if end point
        elif subset[1] > n_tests:
            subset[1] = n_tests
        test_inds = test_inds[subset[0] : subset[1]]
        n_tests = len(test_inds)

    fig = plt.figure(figsize=(40, 20))

    for plotNum in range(nrows * ncols):
        if plotNum == n_tests:
            break
        cde = cdes[test_inds[plotNum]]
        trueZ = zVals.iloc[test_inds[plotNum]][0]

        ax = fig.add_subplot(nrows, ncols, plotNum + 1)

        if roi == True:
            plt.axvline(0.06, color="green", label=r"$GW Thresh$", alpha=0.8)

        plt.plot(zgrid, cde, label=r"$\hat{p}(z| x_{\rm obs})$")
        plt.axvline(trueZ, color="red", label=r"$z_{\rm obs}$")
        plt.xticks(size=16)
        plt.yticks(size=16)
        plt.xlabel(r"Redshift $z$", size=20)
        plt.ylabel("CDE", size=20)
        plt.xlim(0, 1)
        plt.legend(loc="upper right", prop={"size": 20})
    plt.show()
    return


"""
___________________________________________________________
FUNCTIONS FOR CROSSMATCHING DATA
___________________________________________________________
"""


def rough_search(ralist, declist, point, boxlim):
    """
    Function to run a quick search to get sources around a point
    ralist: array-like, ra of points to search through
    declist: array-like, dec of points to search through
    point: tuple, list 2 elems, the ra and dec of point to search arnd
    boxlim: dist from point to edge of box that will be searched within
    """
    ra = point[0]  # get ra and dec from point
    dec = point[1]
    ra_mask = np.abs(ralist - ra) < boxlim  # return all ra and dec that
    dec_mask = np.abs(declist - dec) < boxlim  # are within the box using a mask
    mask = ra_mask & dec_mask  # take only values that are True, True, i.e. both in box
    return mask


def get_nearest_dists(ralist, declist, point, boxlim):
    """
    Function to get the distances from points in a list from a point in the sky
    Returns indice of Nearest point in masked list, mask for list, min dist from ra declist to point
    """
    ra = point[0]
    dec = point[1]

    mask = rough_search(ralist, declist, point, boxlim)  # get mask from rough search

    ralist_masked = ralist[mask]  # only take the points in this mask
    declist_masked = declist[mask]
    dist_arr = np.sqrt(
        ((ra - ralist_masked) * np.cos(dec))
        ** 2  # calculate distances accounting for sky being sphere
        + (dec - declist_masked) ** 2
    )

    nn_ind = np.argmin(dist_arr)  # index of matching point with mask applied
    min_dist = dist_arr.iloc[nn_ind]
    return nn_ind, mask, min_dist


def get_xmatch_inds(gama_df, ps1_df, searchrad):
    """
    Function to get the matching indices of GAMA and PS1 data within a search radius
    Returns the indices of matching rows of gama and ps1 data and list of min distances
    """
    ps1_inds = []
    gama_inds = []
    min_distances = []
    for row in gama_df.iterrows():  # matching gama to ps1 data, for each gama point
        ind, mask, min_dist = (
            get_nearest_dists(  # get the ind, mask and dist to nearest point
                ps1_df.RA.to_numpy(),
                ps1_df.DEC.to_numpy(),
                (row[1].RA, row[1].DEC),
                searchrad,
            )
        )
        ps1_ind_match = ps1_df[mask].index[
            ind
        ]  # get the index of the row with nearest point
        gama_ind_match = row[
            0
        ]  # get the index of the row in gama that matched the ps1 data
        ps1_inds.append(ps1_ind_match)  # add the indices to list to return
        gama_inds.append(gama_ind_match)
        min_distances.append(min_dist)  # add the min dist to list for returning
    return gama_inds, ps1_inds, min_distances


"""
___________________________________________________________
CHECK IF LIST OF POINTS IS INSIDE POLYGON
___________________________________________________________
"""


def getslope(point1, point2):
    """
    Gets slope of line between 2 points, if xdiff is very small,
    slope is assumed to be large
    """
    ydiff = point2[1] - point1[1]  # y2 - y1
    xdiff = point2[0] - point1[0]  # x2 - x1
    if abs(xdiff) < sys.float_info.min:
        return sys.float_info.max
    else:
        return ydiff / xdiff


def ray_intersect_edge(point, edge):
    """
    Function to check if a raycasted point intersects an edge of a polygon (line)
    point: tuple, x and y coordinates
    edge: tuple of 2 points with their own x and y coords
    returns: True or False Depending on whether the point intersects
    the line when raycast to the right
    """
    a, b = edge[0], edge[1]
    if (
        a[1] > b[1]
    ):  # ensure that a is the lower point, this is to check if point outside range
        a, b = b, a
    if point[1] == a[1] or point[1] == b[1]:  # if point on same y level as a or b
        point[1] += 1e-6  # shift up slightly to avoiud ambiguity
        # if outside bounding box of edge, does not intersect
    if (point[1] > b[1] or point[1] < a[1]) or (point[0] > max(a[0], b[0])):
        return False
        # if inside bounding box and to the left of either point on the line, will intersect
    if point[0] < min(a[0], b[0]):
        return True

    else:  # in bounding box but between x position of each point
        segment_slope = getslope(a, b)  # slope of segment
        line_to_edge_slope = getslope(a, point)  # slope of point to bottom of segment
        if line_to_edge_slope >= segment_slope:  # if true, they intersect
            return True
        else:
            return False


def check_in_hull(points, hull):
    """
    Function to check if list of points are in or outside of a polygon
    points: list of points, which is a tuple of x and y position
    hull: list of points which contruct a polygon, or points(ConvexHull.vertices)
    """
    in_hull = (
        []
    )  # to keep mask for list of points provided True if in hull, false otherwise
    edges = [
        (a, b) for a, b in pairwise(hull)
    ]  # iterate over hull 2 objects at a time to get edges
    for point in points:  # for each point
        total = sum(
            ray_intersect_edge(point, edge) for edge in edges
        )  # check sum of how many intersections
        if total % 2 == 1:
            in_hull.append(True)  # if odd, is inside polygon
        else:
            in_hull.append(False)  # if not odd is false.
    return np.array(in_hull)
