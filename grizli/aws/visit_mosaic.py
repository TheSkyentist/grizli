"""
Insert mosaics from visit_processor onto 4 arcmin subtiles
on the same sky tessellation as the cutouts.

"""

import os
import glob
import time

import numpy as np

import astropy.io.fits as pyfits

try:
    from .. import utils, prep
    from . import db
except ImportError:
    from grizli import utils, prep
    from grizli.aws import db

SUBTILE_SIZE = 4.096  # arcmin = (1024 + 2048) 80 mas pixels
SUBTILE_N = 62
SUBTILE_NPIX = 1024 + 2048
SUBTILE_REF_SCALE = 0.08


def prepare_tiles():
    from grizli.aws import tile_mosaic

    tile_mosaic.PIXEL_SCALE = 0.08

    tab = tile_mosaic.define_tile_grid(a=4, round_npix=(2048 + 1024), offset_crpix=0)

    xc = np.array([tab[c] for c in ["r1", "r2", "r3", "r4"]])
    yc = np.array([tab[c] for c in ["d1", "d2", "d3", "d4"]])

    for c in ["r1", "r2", "r3", "r4", "d1", "d2", "d3", "d4"]:
        tab[c].format = ".4f"

    for c in ["crval1", "crval2"]:
        tab[c].format = ".7f"

    xdiff = xc.max(axis=0) - xc.min(axis=0)
    xfix = ((xdiff > 180)[None, :] * np.ones((4, 1))) > 0
    xfix &= xc > 300
    xc[xfix] -= 360

    sr = utils.SRegion(
        [np.array([xc[:, j], yc[:, j]]).T for j in range(len(tab))], wrap=False
    )

    tab["footprint"] = sr.polystr(precision=4)
    tab["pixel_scale_mas"] = int(tile_mosaic.PIXEL_SCALE * 1000)
    tab["round_npix"] = 2048 + 1024

    tab.write("/tmp/mosaic_sky_tessellation.csv", overwrite=True)


def parse_global_tiles(pad_arcsec=10.24, rows=None):
    """ """
    from tqdm import tqdm

    tab = db.SQL("select * from mosaic_tiles")

    # rows = db.SQL(f"SELECT * from assoc_mosaic WHERE assoc_name like '%%nexus%%' AND filter = 'F444W-CLEAR'")

    if rows is None:
        rows = db.SQL(
            f"SELECT * from assoc_mosaic WHERE assoc_name like '%%nexus%%' AND filter > 'F26'"
        )

    # foot = [utils.SRegion(f) for f in rows['footprint']]
    tile_info = []

    for row in tqdm(rows):
        k = np.where(tab["tile"] == row["tile"])[0][0]
        sr = utils.SRegion(row["footprint"])

        tile = tab[k]

        htile, wtile = utils.make_wcsheader(
            tile["crval1"],
            tile["crval2"],
            size=SUBTILE_SIZE * SUBTILE_N * 60,  # ~4 deg
            pixscale=row["pixscale_mas"] / 1000,
            get_hdu=False,
        )

        pix_corners = np.round(wtile.all_world2pix(*sr.xy[0].T, 0)).astype(int)

        tile_npix = int(SUBTILE_NPIX * SUBTILE_REF_SCALE * 1000 / row["pixscale_mas"])

        # pad = int(128 * 80 / row['pixscale_mas'])

        sub_i, sub_j = np.floor(pix_corners / tile_npix).astype(int)
        # ij_indices = np.unique(ij, axis=1)

        pad = int(pad_arcsec / row["pixscale_mas"] * 1000)

        for i in range(sub_i.min(), sub_i.max() + 1):
            for j in range(sub_j.min(), sub_j.max() + 1):
                slx = slice(
                    *np.clip(
                        [i * tile_npix - pad, (i + 1) * tile_npix + pad],
                        0,
                        wtile.pixel_shape[0],
                    )
                )

                sly = slice(
                    *np.clip(
                        [j * tile_npix - pad, (j + 1) * tile_npix + pad],
                        0,
                        wtile.pixel_shape[0],
                    )
                )

                # if 0:
                #     wsl = wtile.slice((sly, slx))
                #     plt.plot(*wsl.calc_footprint().T, alpha=0.1, color="0.5")

                # slice_header = utils.get_wcs_slice_header(wtile, slx, sly)

                subtile_prefix = "-".join(
                    [
                        "tile",
                        f"{tile['tile']:04d}x{i:02d}y{j:02d}",
                        # row["filter"].lower()
                    ]
                )

                rowd = {}
                for k in [
                    "tile",
                    "assoc_name",
                    "filter",
                    "version",
                    "pixscale_mas",
                ]:
                    rowd[k] = row[k]

                for k in ["crval1", "crval2"]:
                    rowd[k] = tile[k]

                rowd["subtile_prefix"] = subtile_prefix
                rowd["subtile_i"] = i
                rowd["subtile_j"] = j
                rowd["subtile_size"] = SUBTILE_SIZE
                rows["subtile_n"] = SUBTILE_N
                rowd["subtile_xstart"] = slx.start
                rowd["subtile_xstop"] = slx.stop
                rowd["subtile_ystart"] = sly.start
                rowd["subtile_ystop"] = sly.stop
                rowd["status"] = 0
                rowd["ctime"] = time.time()

                tile_info.append(rowd)

    return utils.GTable(tile_info)


def initialize_table():
    """
    Parse database rows
    """
    from grizli.aws import visit_mosaic

    rows = db.SQL(
        f"SELECT * from assoc_mosaic WHERE assoc_name like '%%nexus%%' AND filter > 'F2' AND filter not like '%%GRISM%%' AND filter < 'F5'"
    )

    # VENUS
    progs = ["6882"]
    progs = ["2514", "5398"]
    progs = ["2561"] # UNCOVER
    
    prog_strings = ",".join(db.quoted_strings(progs))

    # ven = db.SQL("select assoc_name, avg(ra) as ra, avg(dec) as dec from assoc_table where proposal_id = '6882' and filter = 'F444W-CLEAR' group by assoc_name")
    # ven = db.SQL("""select DISTINCT ON (assoc_name, filter, status) assoc_name, filter, status, proposal_id FROM assoc_table, (select assoc_name as venus_assoc, avg(ra) as venus_ra, avg(dec) as venus_dec from assoc_table where proposal_id = '6882' and filter = 'F444W-CLEAR' group by assoc_name) ven
    # WHERE instrument_name = 'NIRCAM'
    # AND filter not like '%%GRISM%%'
    # AND status = 2
    # AND CIRCLE(POINT(ra-venus_ra, dec - venus_dec), 0.3) @> POINT(0,0)
    # ORDER BY status
    # """)

    rows = db.SQL(
        f"""
    SELECT assoc_mosaic.*, parent.proposal_id from assoc_mosaic,
        (SELECT DISTINCT ON (assoc_name, filter, status)
            assoc_name, filter, status, proposal_id FROM assoc_table,
            (
                SELECT assoc_name as venus_assoc,
                avg(ra) as venus_ra, avg(dec) as venus_dec from assoc_table
                WHERE proposal_id in ({prog_strings})
                AND filter = 'F444W-CLEAR'
                group by assoc_name
            ) ven
        WHERE instrument_name = 'NIRCAM'
        AND filter not like '%%GRISM%%'
        AND status = 2
        AND CIRCLE(POINT(ra-venus_ra, dec - venus_dec), 0.3) @> POINT(0,0)
        ) parent
    WHERE assoc_mosaic.assoc_name = parent.assoc_name
    AND assoc_mosaic.filter not like '%%GRISM%%'
    """
    )

    now = utils.nowtime().mjd

    prev = db.SQL(f"""
        SELECT DISTINCT ON (subtile_prefix, assoc_mosaic_combine.assoc_name, assoc_mosaic_combine.filter)
        assoc_mosaic_combine.assoc_name, assoc_mosaic_combine.filter,
               subtile_prefix, tile, assoc_mosaic_combine.status,
               ({now} - modtime) as days_ago
        FROM assoc_mosaic_combine, assoc_table
        WHERE assoc_mosaic_combine.assoc_name = assoc_table.assoc_name
        ORDER BY subtile_prefix, assoc_mosaic_combine.assoc_name, assoc_mosaic_combine.filter, assoc_table.modtime
    """)

    prev = db.SQL("SELECT assoc_name, filter, subtile_prefix from assoc_mosaic_combine")
    keep = ~np.isin(rows["assoc_name"], prev["assoc_name"])
    rows = rows[keep]

    # # COSMOS
    # rows = db.SQL(
    #     f"SELECT * from assoc_mosaic WHERE POLYGON(CIRCLE(POINT(150.2, 2.2), 2)) @> POLYGON(footprint) AND filter > 'F0' AND detector like 'NRC%%' AND filter not like '%%GRISM%%' AND filter < 'F5'"
    # )
    #
    # # NEXUS
    # rows = db.SQL(
    #     f"SELECT * from assoc_mosaic WHERE assoc_name like '%%nexus%%' AND detector like 'NRC%%' AND filter not like '%%GRISM%%' AND filter < 'F5'"
    # )

    tile_info = visit_mosaic.parse_global_tiles(rows=rows)
    entry = "{assoc_name}-{filter}-{subtile_prefix}"
    newf = [entry.format(**row) for row in tile_info]

    oldf = [entry.format(**row) for row in prev]
    new_rows = ~np.isin(newf, oldf)

    tile_info["status"] = 70

    if new_rows.sum() > 0:
        print(f"send {new_rows.sum()} rows to assoc_mosaic_combine")
        db.send_to_database(
            "assoc_mosaic_combine", tile_info[new_rows], if_exists="append"
        )

    # Field + filters by exposure time
    pre = db.SQL(
        """
        SELECT subtile_prefix, avg(ra) as ra, avg(dec) as dec, 
               assoc_mosaic_combine.filter, count(*) as nassoc,
               sum(exptime) / 3600. as exptime
        FROM assoc_mosaic_combine, assoc_table
        WHERE assoc_mosaic_combine.status = 70
        AND assoc_mosaic_combine.assoc_name = assoc_table.assoc_name
        AND assoc_mosaic_combine.filter = assoc_table.filter
        GROUP BY subtile_prefix, assoc_mosaic_combine.filter
        ORDER BY sum(exptime) desc
    """
    )
    
    # Fields with multiple filters
    pre = db.SQL(
        """
        SELECT subtile_prefix, string_agg(distinct(filter), ' '), count(distinct(filter)) as nfilt, count(*) as nassoc
        FROM assoc_mosaic_combine
        WHERE status = 70
        GROUP BY subtile_prefix
        ORDER BY count(distinct(filter)) desc, subtile_prefix
    """
    )

    from grizli.aws import db
    import numpy as np
    from grizli import utils

    pre = db.SQL(
        """
        SELECT subtile_prefix, string_agg(distinct(filter), ' '), count(distinct(filter)) as nfilt, count(*) as nassoc
        FROM assoc_mosaic_combine
        WHERE status = 2
        GROUP BY subtile_prefix
        ORDER BY count(distinct(filter)) desc, subtile_prefix
    """
    )

    done = db.SQL(
        """
        SELECT subtile_prefix, filter, count(*)
        FROM assoc_mosaic_combine
        WHERE status = 2
        GROUP BY subtile_prefix, filter
        ORDER BY filter, count(*)
    """
    )
    utils.Unique(done["filter"])

    todo = db.SQL(
        """
        SELECT subtile_prefix, filter, count(*)
        FROM assoc_mosaic_combine
        WHERE status = 0
        GROUP BY subtile_prefix, filter
        ORDER BY filter, count(*)
    """
    )

    utils.Unique(todo["filter"])

    print(f"Done: {len(done)}  ToDo: {len(todo)}")

    so = np.argsort(np.random.normal(size=len(todo)))

    for row in todo[so]:
        result = process_subtile(
            subtile_prefix=row["subtile_prefix"],
            filter=row["filter"],
            clean=True,
            avoid_overlap=True,
        )


def get_subtile_cutout_wcs(
    tile=0,
    crval1=0.0,
    crval2=0.0,
    subtile_prefix="subtile",
    subtile_size=0,
    subtile_xstart=0,
    subtile_xstop=0,
    subtile_ystart=0,
    subtile_ystop=0,
    pixscale_mas=40,
    **kwargs,
):
    """ """
    import astropy.wcs as pywcs

    htile, wtile = utils.make_wcsheader(
        crval1,
        crval2,
        size=SUBTILE_SIZE * SUBTILE_N * 60,
        pixscale=pixscale_mas / 1000.0,
        get_hdu=False,
    )

    slx, sly = slice(subtile_xstart, subtile_xstop), slice(
        subtile_ystart, subtile_ystop
    )
    # print(slx)

    wsl = pywcs.WCS(utils.get_wcs_slice_header(wtile, slx, sly))
    base_wcs = pywcs.WCS(utils.get_wcs_slice_header(wtile, slx, sly))

    weight_type = "jwst"

    # pixscale_mas = int(np.round(utils.get_wcs_pscale(wsl)*1000))

    out = {
        "tile": tile,
        "base_wcs": base_wcs,
        "wcs": wsl,
        "slx": slx,
        "sly": sly,
        "skip": 16,
        "weight_type": weight_type,
        "pixscale_mas": pixscale_mas,
    }

    return out


def run_one_subtile(row=None):
    """
    Run one subtile from the queue
    """
    if row is None:
        row = db.SQL("""
            SELECT subtile_prefix, filter, count(*)
            FROM assoc_mosaic_combine
            WHERE status = 0
            GROUP BY subtile_prefix, filter
            ORDER BY RANDOM() LIMIT 1
        """)
        if len(row) == 0:
            row = None
        else:
            row = row[0]

    elif row in ["test"]:
        row = db.SQL("""
            SELECT subtile_prefix, filter, count(*), max(status) as status
            FROM assoc_mosaic_combine
            WHERE subtile_prefix = 'tile-0661x20y12'
                  AND filter = 'F356W-CLEAR'
            GROUP BY subtile_prefix, filter
            ORDER BY RANDOM() LIMIT 1
        """)
        if len(row) == 0:
            row = None
        else:
            row = row[0]

    if row is None:
        print("Nothing to do in assoc_mosaic_combine")
        return None

    result = process_subtile(
        subtile_prefix=row["subtile_prefix"],
        filter=row["filter"],
        clean=True,
        avoid_overlap=True,
    )

    return row


def process_subtile(
    subtile_prefix="tile-2582x30y36",
    filter="F444W-CLEAR",
    version="v7.0",
    count_only=False,
    avoid_overlap=True,
    clean=False,
):

    from msaexp.cloud.utils import download_from_s3, upload_to_s3

    tile_rows = db.SQL(
        f"""
        SELECT * FROM assoc_mosaic_combine
        WHERE subtile_prefix = '{subtile_prefix}'
        AND filter = '{filter}'
    """
    )

    bucket = "grizli-v2"
    object_path = f"assoc_mosaic/combined/{version}/{tile_rows['tile'][0]}"

    s3_path = "s3://{bucket}/{object_path}/{subtile_prefix}-{filter}_drc_{ext}.fits.gz"

    remote_file = s3_path.format(
        bucket=bucket,
        object_path=object_path,
        subtile_prefix=subtile_prefix,
        filter=filter,
        ext="sci",
    ).lower()

    local_file = os.path.basename(remote_file)

    complete = tile_rows["status"] == 2
    nold = complete.sum()
    nnew = (~complete).sum()

    msg = f"process_subtile: {local_file} Nold = {nold}  Nnew = {nnew}"
    utils.log_comment(utils.LOGFILE, msg, verbose=True)

    if (~complete).sum() == 0:
        return tile_rows
    elif count_only:
        return tile_rows

    in_process = tile_rows["status"] == 1
    if (in_process.sum() > 0) & (avoid_overlap):
        msg = f"process_subtile: {local_file} running N = {in_process.sum()}"
        utils.log_comment(utils.LOGFILE, msg, verbose=True)
        return tile_rows

    db.execute(
        f"""
        UPDATE assoc_mosaic_combine
        SET status=1, ctime={time.time()}
        WHERE subtile_prefix = '{subtile_prefix}'
        AND filter = '{filter}'
        AND assoc_name in ({','.join(db.quoted_strings(tile_rows['assoc_name'][~complete]))})
    """
    )

    out = get_subtile_cutout_wcs(**tile_rows[0])
    wcs = out["wcs"]
    sh = out["wcs"].pixel_shape

    clean_files = []

    if complete.sum() > 0:
        for ext in ["sci", "wht", "var"]:
            local_file_i = s3_path.format(
                bucket=bucket,
                object_path=object_path,
                subtile_prefix=subtile_prefix,
                filter=filter,
                ext=ext,
            ).lower()

            if not os.path.exists(os.path.basename(local_file_i)):
                msg = f"    Fetch {local_file_i}"
                utils.log_comment(utils.LOGFILE, msg, verbose=True)

                download_from_s3(local_file_i, overwrite=True)

        msg = f"process_subtile: Restart from {local_file}"
        utils.log_comment(utils.LOGFILE, msg, verbose=True)

        with pyfits.open(local_file) as im:
            sci_ = im[0].data * 1
            outh = im[0].header.copy()

        with pyfits.open(local_file.replace("_sci.fits", "_wht.fits")) as im:
            den = im[0].data * 1

        with pyfits.open(local_file.replace("_sci.fits", "_var.fits")) as im:
            var_ = im[0].data * 1

        num = sci_ * den
        varnum = var_ * den**2

        start_count = outh["NASSOC"]

    else:

        msg = f"process_subtile: Initialize {local_file}"
        utils.log_comment(utils.LOGFILE, msg, verbose=True)

        outh = utils.to_header(wcs)

        start_count = 0
        outh["NASSOC"] = 0

        num = np.zeros((outh["NAXIS2"], outh["NAXIS1"]), dtype=np.float32)
        varnum = np.zeros((outh["NAXIS2"], outh["NAXIS1"]), dtype=np.float32)
        den = np.zeros_like(num)

    outh["NASSOC"] += len(tile_rows[~complete])

    s3url = (
        "s3://grizli-v2/assoc_mosaic/{version}/{assoc_name}-{filter}_drc_{ext}.fits.gz"
    )

    for i, row in enumerate(tile_rows[~complete]):

        msg = f"process_subtile: ({i+1} / {nnew}) add {row['assoc_name']} to output"
        utils.log_comment(utils.LOGFILE, msg, verbose=True)

        local_files = {}
        open_files = {}
        for ext in ["sci", "wht", "var"]:
            s3_assoc = s3url.format(
                version=version, assoc_name=row["assoc_name"], filter=filter, ext=ext
            ).lower()

            local_files[ext] = os.path.basename(s3_assoc)
            clean_files.append(local_files[ext])

            if not os.path.exists(local_files[ext]):
                msg = f"    download {s3_assoc}"
                utils.log_comment(utils.LOGFILE, msg, verbose=True)

                _ = download_from_s3(s3_assoc, overwrite=False)

            open_files[ext] = pyfits.open(local_files[ext])

        nx, ny, valid, sl1, sl2 = utils.get_fits_slices(outh, open_files["sci"])
        ssl = open_files["sci"][0].data[sl2]
        ssl[~np.isfinite(ssl)] = 0

        wsl = open_files["wht"][0].data[sl2]
        wsl[~np.isfinite(wsl)] = 0

        vsl = open_files["var"][0].data[sl2]
        vsl[~np.isfinite(vsl)] = 0

        num[sl1] += ssl * wsl
        varnum[sl1] += vsl * wsl**2
        den[sl1] += wsl

        outh[f"ASSOC{i + start_count:03d}"] = row["assoc_name"]

        header = open_files["sci"][0].header.copy()

        for k in header:
            if k not in outh:
                outh[k] = header[k]

        for ext in open_files:
            open_files[ext].close()

        if clean:
            for ext in local_files:
                file_ = local_files[ext]
                if os.path.exists(file_):
                    msg = f"    rm {file_}"
                    utils.log_comment(utils.LOGFILE, msg, verbose=True)
                    os.remove(file_)

    xsci = num / den
    xsci[den <= 0] = 0
    xvar = varnum / den**2
    xvar[den <= 0] = 0

    pyfits.writeto(
        f"{subtile_prefix}-{filter.lower()}_drc_sci.fits.gz",
        data=xsci,
        header=outh,
        overwrite=True,
    )

    pyfits.writeto(
        f"{subtile_prefix}-{filter.lower()}_drc_var.fits.gz",
        data=xvar,
        header=outh,
        overwrite=True,
    )

    pyfits.writeto(
        f"{subtile_prefix}-{filter.lower()}_drc_wht.fits.gz",
        data=den,
        header=outh,
        overwrite=True,
    )

    clean_files += glob.glob(f"{subtile_prefix}-{filter.lower()}*fits.gz")

    for ext in ["sci", "wht", "var"]:
        file_i = local_file.replace("_sci.fits", f"_{ext}.fits")

        msg = f"Send {file_i} to s3://{bucket}/{object_path}"
        utils.log_comment(utils.LOGFILE, msg, verbose=True)

        upload_to_s3(
            file_i,
            bucket,
            object_name=None,
            object_path=object_path,
            ExtraArgs={"ACL": "public-read"},
            content_types={
                "png": "image/png",
                "jpg": "image/jpeg",
                "gif": "image/gif",
                "fits": "application/fits",
                "csv": "text/csv",
            },
            verbose=False,
            # **kwargs,
        )

    if clean:
        for file_ in clean_files:
            if os.path.exists(file_):
                msg = f"    rm {file_}"
                utils.log_comment(utils.LOGFILE, msg, verbose=True)
                os.remove(file_)

    db.execute(
        f"""
        UPDATE assoc_mosaic_combine
        SET status=2, ctime={time.time()}
        WHERE subtile_prefix = '{subtile_prefix}'
        AND filter = '{filter}'
        AND assoc_name in ({','.join(db.quoted_strings(tile_rows['assoc_name'][~complete]))})
    """
    )

    return True


def all_catalogs():
    """
    Make all catalogs
    """
    pre = db.SQL(
        """
        SELECT subtile_prefix, string_agg(distinct(filter), ' '), count(distinct(filter)) as nfilt, count(*) as nassoc
        FROM assoc_mosaic_combine
        WHERE status = 2
        GROUP BY subtile_prefix
        ORDER BY count(distinct(filter)) desc, subtile_prefix
    """
    )

    so = np.argsort(np.random.normal(size=len(pre)))

    for prefix in pre["subtile_prefix"][so]:
        print(prefix)
        make_subtile_catalog(subtile_prefix=prefix)


def make_subtile_catalog(
    subtile_prefix="tile-2582x30y36",
    version="v7.0",
    count_only=False,
    avoid_overlap=True,
    clean=True,
):
    """
    Make a catalog of all available filters for a particular subtile
    """
    from msaexp.cloud.utils import download_from_s3, upload_to_s3
    from grizli.pipeline import auto_script
    from grizli import prep

    lock_file = f"{subtile_prefix}.lock"
    if os.path.exists(lock_file):
        return lock_file

    with open(lock_file, "w") as fp:
        fp.write(time.ctime())

    if 0:
        pre = db.SQL(
            """
            SELECT subtile_prefix, string_agg(distinct(filter), ' '), count(distinct(filter)) as nfilt, count(*) as nassoc
            FROM assoc_mosaic_combine
            WHERE status = 2
            GROUP BY subtile_prefix
            ORDER BY count(distinct(filter)) desc, subtile_prefix
        """
        )

    tile_rows = db.SQL(
        f"""
        SELECT * FROM assoc_mosaic_combine
        WHERE subtile_prefix = '{subtile_prefix}'
    """
    )

    bucket = "grizli-v2"
    object_path = f"assoc_mosaic/combined/{version}/{tile_rows['tile'][0]}"

    unf = utils.Unique(tile_rows["filter"])

    s3_path = "s3://{bucket}/{object_path}/{subtile_prefix}-{filter}_drc_{ext}.fits.gz"

    clean_files = []

    for filter in unf.values:
        for ext in ["sci", "wht", "var"]:
            local_file_i = s3_path.format(
                bucket=bucket,
                object_path=object_path,
                subtile_prefix=subtile_prefix,
                filter=filter,
                ext=ext,
            ).lower()

            if not os.path.exists(os.path.basename(local_file_i)):
                msg = f"    Fetch {local_file_i}"
                utils.log_comment(utils.LOGFILE, msg, verbose=True)

                download_from_s3(local_file_i, overwrite=True)

            clean_files.append(os.path.basename(local_file_i))

    root = subtile_prefix

    comb = {
        "ir": [
            "F444W-CLEAR",
            "F356W-CLEAR",
            "F277W-CLEAR",
            "F410M-CLEAR",
            "F300M-CLEAR",
            "F335M-CLEAR",
            "F360M-CLEAR",
            "F430M-CLEAR",
            "F460M-CLEAR",
            "F480M-CLEAR",
            "CLEARP-F277W",
            "CLEARP-F356W",
            "CLEARP-F444W",
        ]
    }

    block_filters = [
        "F090W-CLEAR",
        "F115W-CLEAR",
        "F150W-CLEAR",
        "F200W-CLEAR",
        "F182M-CLEAR",
        "F210M-CLEAR",
    ]

    mosaic_file = f"{root}-ir_drc_sci.fits.gz"
    if not os.path.exists(mosaic_file):
        status = auto_script.make_filter_combinations(
            root,
            filter_combinations=comb,
            weight_fnu=2,
            force_photfnu=1.0e-8,
            block_filters=block_filters,
        )

    # Fewer apertures
    phot_apertures = prep.SEXTRACTOR_PHOT_APERTURES_ARCSEC[:3]
    print(phot_apertures)

    auto_script.multiband_catalog(
        field_root=root,
        threshold=1.5,
        bkg_params={"bw": 50, "bh": 50, "fw": 3, "fh": 3, "pixel_scale": 0.04},
        get_all_filters=True,
        phot_err_scale=1.0,
        phot_apertures=phot_apertures,
    )

    bkg_files = glob.glob(f"{root}*bkg.fits")
    bkg_files.sort()
    for file_ in bkg_files:
        if os.path.exists(file_):
            msg = f"    rm {file_}"
            utils.log_comment(utils.LOGFILE, msg, verbose=True)
            os.remove(file_)

    cmd = f"gzip --force {root}-ir_seg.fits"
    utils.log_comment(utils.LOGFILE, cmd, verbose=True)
    os.system(cmd)

    cat_files = glob.glob(f"{root}-ir*")
    cat_files += glob.glob(f"{root}_phot.fits")
    cat_files.sort()

    for file_ in cat_files:
        msg = f"Send {file_} to s3://{bucket}/{object_path}"
        utils.log_comment(utils.LOGFILE, msg, verbose=True)

        upload_to_s3(
            file_,
            bucket,
            object_name=None,
            object_path=object_path,
            ExtraArgs={"ACL": "public-read"},
            content_types={
                "png": "image/png",
                "jpg": "image/jpeg",
                "gif": "image/gif",
                "fits": "application/fits",
                "csv": "text/csv",
            },
            verbose=False,
            # **kwargs,
        )

        clean_files.append(os.path.basename(file_))

    clean_files.sort()
    if clean:
        for file_ in clean_files:
            if os.path.exists(file_):
                msg = f"    rm {file_}"
                utils.log_comment(utils.LOGFILE, msg, verbose=True)
                os.remove(file_)


### TBD: make catalogs and detection images.  Photometry by epoch?
