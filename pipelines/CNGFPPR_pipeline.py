import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing
from astropy import units as u
from astropy.coordinates import SkyCoord
import astroFuncs as af


def replace_invalid_with_nan(df):
    df = df.replace(-999, np.nan)
    return df


def get_valid_r_magnitudes(df):
    df = df[df.rKronMag.notna()]
    df = df[df.rKronMagErr.notna()]
    df = df[df.rPSFMag.notna()]
    df = df[df.rPSFMagErr.notna()]
    return df


def get_r_magnitudes_in_range(df):
    lwr_ext_maglim = 17
    upr_ext_maglim = 19.72  # 20
    df = df.query(
        "rKronMag >= @lwr_ext_maglim &\
                   rKronMag <= @upr_ext_maglim"
    )
    return df


def drop_duplicate_detections(df):
    df = df.sort_values(by="nDetections", ascending=False, kind="stable")
    df["isDupe"] = df.duplicated("objID", keep="first")
    df = df.query("isDupe == False")
    return df


def drop_non_primary_detections(df):
    if "primaryDetection" in df.columns:
        df = df.query("primaryDetection == 1")
    return df


def get_probable_extended_sources(df):
    nSigmaConf = 5
    df["Extended"] = np.where(
        (df.rExtNSigma > nSigmaConf) & (df.gExtNSigma > nSigmaConf), True, False
    )
    df = df.query("Extended == True")
    df = df.drop(["Extended", "isDupe"], axis=1).reset_index(drop=True)
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

            to_drop = proximity_search(
                ralist[nearby_inds], declist[nearby_inds], [star_ra, star_dec], mask
            )

            ralist = ralist.drop(to_drop).reset_index(drop=True)
            declist = declist.drop(to_drop).reset_index(drop=True)
            df = df.drop(to_drop).reset_index(drop=True)
            # print(len(to_drop))
            num_dropped += len(to_drop)
        print(f"Masked {num_dropped} sources from bright star masking.")

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

        # for index, row in gals.iterrows():
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

            to_drop = in_gal.nonzero()[0]

            ralist = ralist.drop(to_drop).reset_index(drop=True)
            declist = declist.drop(to_drop).reset_index(drop=True)
            df = df.drop(to_drop).reset_index(drop=True)
            # print(len(to_drop))
            num_dropped += len(to_drop)
        print(f"Masked {num_dropped} sources from galaxy masking.")
    return df


def create_features_for_ml_prediction(df):
    # creating colour features
    df["gr_PS"] = df.gKronMag - df.rKronMag
    df["ri_PS"] = df.rKronMag - df.iKronMag
    df["iz_PS"] = df.iKronMag - df.zKronMag
    df["zy_PS"] = df.zKronMag - df.yKronMag
    return df


def return_model_compatible_testing_data(df):
    df = df[["rKronMag", "gr_PS", "ri_PS", "iz_PS", "zy_PS"]]
    return df


def process_data(df):

    min_ra = np.min(df.raMean)
    max_ra = np.max(df.raMean)

    min_dec = np.min(df.decMean)
    max_dec = np.max(df.decMean)

    width = max_ra - min_ra
    height = max_dec - min_dec

    bright_stars = af.square_search_bright_stars(min_ra, min_dec, width, height)

    gals = af.square_query_HLEDA(min_ra, min_dec, width, height)

    df = (
        df.pipe(replace_invalid_with_nan)
        .pipe(get_valid_r_magnitudes)
        .pipe(get_r_magnitudes_in_range)
        .pipe(drop_duplicate_detections)
        .pipe(drop_non_primary_detections)
        .pipe(get_probable_extended_sources)
        # .pipe(mask_bright_stars, bright_stars = bright_stars)
        .pipe(mask_resolved_galaxies, gals=gals)
        .pipe(create_features_for_ml_prediction)
    )

    # training_df = df.pipe(return_model_compatible_testing_data)

    return df


def process_data_no_sigma_cut(df):

    min_ra = np.min(df.raMean)
    max_ra = np.max(df.raMean)

    min_dec = np.min(df.decMean)
    max_dec = np.max(df.decMean)

    width = max_ra - min_ra
    height = max_dec - min_dec

    bright_stars = af.square_search_bright_stars(min_ra, min_dec, width, height)

    gals = af.square_query_HLEDA(min_ra, min_dec, width, height)

    df = (
        df.pipe(replace_invalid_with_nan)
        .pipe(get_valid_r_magnitudes)
        .pipe(get_r_magnitudes_in_range)
        .pipe(drop_duplicate_detections)
        .pipe(drop_non_primary_detections)
        # .pipe(get_probable_extended_sources)
        .pipe(mask_bright_stars, bright_stars=bright_stars)
        .pipe(mask_resolved_galaxies, gals=gals)
        .pipe(create_features_for_ml_prediction)
    )

    # training_df = df.pipe(return_model_compatible_testing_data)

    return df


def process_data_return_gals(df):

    min_ra = np.min(df.raMean)
    max_ra = np.max(df.raMean)

    min_dec = np.min(df.decMean)
    max_dec = np.max(df.decMean)

    width = max_ra - min_ra
    height = max_dec - min_dec

    bright_stars = af.square_search_bright_stars(min_ra, min_dec, width, height)
    # gals = af.square_search_hleda(min_ra, min_dec, width, height)
    gals = af.square_query_HLEDA(min_ra, min_dec, width, height)

    df = (
        df.pipe(replace_invalid_with_nan)
        .pipe(get_valid_r_magnitudes)
        .pipe(get_r_magnitudes_in_range)
        .pipe(drop_duplicate_detections)
        .pipe(drop_non_primary_detections)
        .pipe(get_probable_extended_sources)
        .pipe(mask_bright_stars, bright_stars=bright_stars)
        # .pipe(mask_resolved_galaxies, gals = gals)
        .pipe(create_features_for_ml_prediction)
    )

    # training_df = df.pipe(return_model_compatible_testing_data)
    # df = pd.concat([df, gals])
    return df, gals
