"""
Created on Tue Dec 17 11:57:19 2024

@author: richard.oneill1@ucdconnect.ie
"""

import pandas as pd
import numpy as np
import pickle

from tqdm import tqdm
from astropy import units as u
from astropy.coordinates import SkyCoord
import zutils.astrofuncs as af

import os

#'photo_z_estimator.sav'
base_dir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(base_dir, "ml_model", "photo_z_estimator_nan_handling.sav")
model = pickle.load(open(model_path, "rb"))

# bump_threshold = 0.09
# sharpen_alpha = 1.04

"""
to_flag = df[df.isna().any(axis=1)].index.values
test = np.zeros(len(df), dtype=int)
flags = np.ones(len(df), dtype=int)
test[to_flag] = test[to_flag] | 1
"""


def make_flag_array(df):
    df["data_flag"] = np.zeros(len(df), dtype=int)
    return df


def drop_invalid(df):
    print("Flagging invalid values...")
    df = df.replace(-999, np.nan)
    # flagging appropriate values
    feature_data = df[
        [
            "gPSFMag",
            "gApMag",
            "gKronMag",
            "gpsfChiSq",
            "gKronRad",
            "rPSFMag",
            "rApMag",
            "rKronMag",
            "rpsfChiSq",
            "rKronRad",
            "iPSFMag",
            "iApMag",
            "iKronMag",
            "ipsfChiSq",
            "iKronRad",
            "zPSFMag",
            "zApMag",
            "zKronMag",
            "zpsfChiSq",
            "zKronRad",
            "yPSFMag",
            "yApMag",
            "yKronMag",
            "ypsfChiSq",
            "yKronRad",
        ]
    ]
    to_flag = feature_data[feature_data.isna().any(axis=1)].index.values
    to_flag_mask = np.isin(df.index, to_flag)
    df["data_flag"] = np.where(to_flag_mask, df["data_flag"] | 2**0, df["data_flag"])
    return df


def get_in_range_magnitudes(df):
    print("Flagging magnitudes out of range...")
    lwr_mag_lim = 0  # 17
    upper_mag_lim = 19.72
    to_flag = df.query(
        "rKronMag < @lwr_mag_lim or\
                   rKronMag > @upper_mag_lim"
    ).index.values
    to_flag_mask = np.isin(df.index, to_flag)
    df["data_flag"] = np.where(to_flag_mask, df["data_flag"] | 2**1, df["data_flag"])
    return df


def drop_duplicate_detections(df):
    print("Flagging Duplicate detections...") # 2**3
    # print("Dropping Duplicates and non-primary detections...")
    df = df.sort_values(by="nDetections", ascending=False, kind="stable")
    # df["isDupe"] = df.duplicated("objID", keep="first")
    to_flag_dupe_mask = df.duplicated("objID", keep="first")
    df["data_flag"] = np.where(to_flag_dupe_mask, df["data_flag"] | 2**3, df["data_flag"])
    # df = df.query("isDupe == False").drop("isDupe", axis=1) 
    return df


def drop_non_primary_detections(df):
    if "primaryDetection" in df.columns:
        print("Flagging Non Primary Detections...")
        init_len = len(df)
        final_len = len(df.query("primaryDetection == 1"))
        print(f"Number of Non-Primary Detections: {init_len - final_len}")
        to_flag_primary_detections = df.query("primaryDetection == 1").index.values
        to_flag_mask = np.isin(df.index, to_flag_primary_detections)
        df["data_flag"] = np.where(to_flag_mask, df["data_flag"] | 2**6, df["data_flag"])
    return df


def get_probable_extended_sources(df):
    print("Flagging non-extended sources...")
    nSigmaConf = 5
    df["Extended"] = np.where(
        ((df.rExtNSigma < nSigmaConf) | (df.gExtNSigma < nSigmaConf)), True, False
    )
    to_flag = df.query("Extended == True").index.values
    to_flag_mask = np.isin(df.index, to_flag)
    df["data_flag"] = np.where(to_flag_mask, df["data_flag"] | 2**2, df["data_flag"])
    # df = df.drop(["Extended"], axis=1).reset_index(drop=True)
    return df


def get_bright_star_mask_size(star_magnitude):
    limiting_mag = 13
    a, b, c = 0.99504614, -30.94434771, 245.40805416
    mask_pixels = a * star_magnitude**2 + b * star_magnitude + c
    mask_degrees = mask_pixels / (4 * 60 * 60)  # convert to degrees
    # where star mag > limiting_mag yield 0 else keep mask
    mask_degrees = np.where(star_magnitude > limiting_mag, 0, mask_degrees)
    return mask_degrees


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


def first_pass_mask(ralist, declist, point, boxlim):  ## needs to be refactored
    ra = point[0]
    dec = point[1]
    ra_mask = np.abs(ralist - ra) < boxlim
    dec_mask = np.abs(declist - dec) < boxlim
    mask_data = {"ra_mask": ra_mask, "dec_mask": dec_mask}
    ra_dec_mask = pd.DataFrame(data=mask_data)
    mask_inds = ra_dec_mask.query("ra_mask == True & dec_mask == True").index
    # masked_ra = ralist[mask_inds]
    # masked_dec = declist[mask_inds]
    return np.asarray(mask_inds)


def mask_bright_stars(df, bright_stars):

    if len(bright_stars) == 0:
        return df

    else:
        bright_stars = bright_stars[bright_stars["Vmag"].notna()]
        bright_stars = bright_stars.query("Vmag < 15").reset_index(drop=True)

        bright_stars["mask_deg"] = get_bright_star_mask_size(bright_stars.Vmag)

        ralist = df.raMean
        declist = df.decMean
        num_dropped = 0
        # for index, row in bright_stars.iterrows():
        print("Masking Bright Stars...")
        for index, row in tqdm(bright_stars.iterrows(), total=bright_stars.shape[0]):
            star_ra = row.RA
            star_dec = row.Dec
            mask = row.mask_deg

            nearby_inds = first_pass_mask(ralist, declist, (star_ra, star_dec), 0.02)

            to_flag = proximity_search(
                ralist[nearby_inds], declist[nearby_inds], [star_ra, star_dec], mask
            )
            
            # ralist = ralist.drop(to_drop).reset_index(drop=True)
            # declist = declist.drop(to_drop).reset_index(drop=True)
            # df = df.drop(to_drop).reset_index(drop=True)
            to_flag_mask = np.isin(df.index, to_flag)
            df["data_flag"] = np.where(to_flag_mask, df["data_flag"] | 2**4, df["data_flag"])

            # print(len(to_drop))
            num_dropped += len(to_flag)
        print(f"Flagged {num_dropped} sources from bright star masking.")

    return df


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
    testpoint = SkyCoord(ra=x, dec=y, unit=u.deg)
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


def mask_resolved_galaxies(df, gals):

    if len(gals) == 0:
        return df

    else:
        gals.PA = gals.PA.fillna(0)
        gals = gals[gals.semi_major.notna()].reset_index(drop=True)

        ralist = df.raMean
        declist = df.decMean

        print("Masking Resolved Galaxies...")
        num_dropped = 0
        for index, row in tqdm(gals.iterrows(), total=gals.shape[0]):

            gal_ra = row.RA
            gal_dec = row.DEC
            semi_major = row.semi_major
            semi_minor = row.semi_minor
            angle = row.PA

            nearby_inds = first_pass_mask(
                ralist, declist, (gal_ra, gal_dec), semi_major
            )

            in_gal = check_in_ellipse(
                ralist[nearby_inds],
                declist[nearby_inds],
                gal_ra,
                gal_dec,
                semi_major,
                semi_minor,
                angle,
            )

            to_flag = in_gal.nonzero()[0]
            to_flag_mask = np.isin(df.index, to_flag)

            df["data_flag"] = np.where(to_flag_mask, df["data_flag"] | 2**5, df["data_flag"])
            # ralist = ralist.drop(to_drop).reset_index(drop=True)
            # declist = declist.drop(to_drop).reset_index(drop=True)
            # df = df.drop(to_drop).reset_index(drop=True)
            
            # print(len(to_drop))
            num_dropped += len(to_flag)
        print(f"Flagged {num_dropped} sources from galaxy masking.")
    return df


def create_features(df):
    # creating colour features
    df["gr_PS"] = df.gKronMag - df.rKronMag
    df["ri_PS"] = df.rKronMag - df.iKronMag
    df["iz_PS"] = df.iKronMag - df.zKronMag
    df["zy_PS"] = df.zKronMag - df.yKronMag
    return df


def get_features_in(df):
    columns = [
        "gPSFMag",
        "gApMag",
        "gKronMag",
        "gpsfChiSq",
        "gKronRad",
        "rPSFMag",
        "rApMag",
        "rKronMag",
        "rpsfChiSq",
        "rKronRad",
        "iPSFMag",
        "iApMag",
        "iKronMag",
        "ipsfChiSq",
        "iKronRad",
        "zPSFMag",
        "zApMag",
        "zKronMag",
        "zpsfChiSq",
        "zKronRad",
        "yPSFMag",
        "yApMag",
        "yKronMag",
        "ypsfChiSq",
        "yKronRad",
        "gr_PS",
        "ri_PS",
        "iz_PS",
        "zy_PS",
    ]
    df = df[columns]
    return df


def square_search_stars_galaxies(ra, dec, width, height):
    bright_stars = af.square_search_bright_stars(ra, dec, width, height)
    gals = af.square_query_HLEDA(ra, dec, width, height)
    return bright_stars, gals


def square_search_ps1(ra1, ra2, dec1, dec2, table_name=None, task_name="My Query"):
    af.mastcasjobs_init()
    ps_df = af.search_rect_region(
        ra1, ra2, dec1, dec2, table_name=table_name, task_name=task_name
    )
    return ps_df


def get_bounding_box(df):
    minra = min(df.raMean)
    maxra = max(df.raMean)
    mindec = min(df.decMean)
    maxdec = max(df.decMean)
    width = maxra - minra
    height = maxdec - mindec
    return minra, mindec, width, height

def retrieve_table(table_name):
    table = af.retrieve_table(table_name)
    return table


def process_data(df, stars, galaxies):

    df = (
        df.pipe(make_flag_array)
        .pipe(drop_invalid)  # 2**0
        .pipe(drop_non_primary_detections) # 2**6
        .pipe(get_in_range_magnitudes)  # 2**1
        .pipe(drop_duplicate_detections)  # 2**3
        .pipe(get_probable_extended_sources)  # 2**2
        .pipe(mask_bright_stars, bright_stars=stars) # 2**4
        .pipe(mask_resolved_galaxies, gals=galaxies) # 2**5
        .pipe(create_features)
    )

    prediction_features = get_features_in(df)

    return df, prediction_features


def estimate(prediction_features, n_grid=1000):
    cdes, zgrid = model.predict(prediction_features, n_grid=n_grid)
    return cdes, zgrid
