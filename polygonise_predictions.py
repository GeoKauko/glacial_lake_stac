"""Turn the prediction zarr into lake polygons - ONE GeoPackage per year.

Writes prediction_labels/lakes_<year>.gpkg, matching the layout of the manual
labels/ folder so predictions can be compared or corrected and reused as labels.

    python polygonize_predictions.py
    python polygonize_predictions.py --threshold 0.4 --min-area 5000
    python polygonize_predictions.py --exclude 1992 1993 1998 2002 2009 2013
"""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import rioxarray  # noqa: registers .rio
import geopandas as gpd
from shapely.geometry import shape
from rasterio.features import shapes as rio_shapes

PROJECT_ROOT = Path(r"D:\Users\b1120440\projects\morpheo\glacial_lake_pred")

# Years flagged as hazy / snow-covered by the median screen in notebook 1.
# Their predictions are not trustworthy, so they are not vectorised.
BAD_YEARS = [1992, 1993, 1998, 2002, 2009, 2013]

ap = argparse.ArgumentParser()
ap.add_argument("--zarr", default=str(PROJECT_ROOT / "new_area_predictions.zarr"))
ap.add_argument("--outdir", default=str(PROJECT_ROOT / "prediction_labels"))
ap.add_argument("--crs", default="EPSG:32627", help="fallback if the zarr lost its CRS")
ap.add_argument("--threshold", type=float, default=None,
                help="re-threshold the probability; default uses the saved 'lake' mask")
ap.add_argument("--min-area", type=float, default=2000.0,
                help="drop polygons smaller than this (m^2) - removes speckle")
ap.add_argument("--simplify", type=float, default=0.0,
                help="Douglas-Peucker tolerance in metres (0 = keep pixel edges)")
ap.add_argument("--exclude", type=int, nargs="*", default=None,
                help=f"years to skip (default: {BAD_YEARS})")
a = ap.parse_args()

exclude = set(BAD_YEARS if a.exclude is None else a.exclude)

outdir = Path(a.outdir)
outdir.mkdir(parents=True, exist_ok=True)

ds = xr.open_zarr(a.zarr)

# open_zarr does not always restore the CRS written by rioxarray
crs = ds.rio.crs
if crs is None:
    ds = ds.rio.write_crs(a.crs)
    crs = ds.rio.crs
    print(f"zarr had no CRS -> assuming {crs}")

# transform from the coords, so it is right even without a stored CRS
ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
transform = ds.rio.transform(recalc=True)
res = abs(float(ds.x[1] - ds.x[0]))

years = [int(y) for y in ds.year.values]
print(f"{len(years)} years in {Path(a.zarr).name} | CRS {crs} | {res:g} m pixels")
print(f"excluding {sorted(exclude)}")
print(f"min-area {a.min_area:g} m^2 = {a.min_area / res**2:.1f} pixels -> {outdir}\n")

total, written = 0, 0
for t, yr in enumerate(years):
    if yr in exclude:
        print(f"{yr}: excluded (flagged hazy/snow)")
        continue

    if a.threshold is None:
        mask = ds["lake"].isel(year=t).values.astype(bool)
    else:
        mask = ds["prob"].isel(year=t).values > a.threshold
    mask &= ds["valid"].isel(year=t).values.astype(bool)   # never vectorise no-data

    if not mask.any():
        print(f"{yr}: no lake pixels - skipped")
        continue

    geoms = []
    for g, v in rio_shapes(mask.astype("uint8"), mask=mask, transform=transform):
        if v != 1:
            continue
        geom = shape(g)
        if geom.area < a.min_area:
            continue
        if a.simplify > 0:
            geom = geom.simplify(a.simplify, preserve_topology=True)
        geoms.append(geom)

    if not geoms:
        print(f"{yr}: all polygons below --min-area - skipped")
        continue

    # geometry only - GPKG still assigns an automatic integer fid
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    dst = outdir / f"lakes_{yr}.gpkg"
    gdf.to_file(dst, driver="GPKG", layer="lakes")
    total += len(gdf); written += 1
    print(f"{yr}: {len(gdf):4d} polygon(s), {sum(g.area for g in geoms)/1e6:8.3f} km²"
          f"  -> {dst.name}")

print(f"\nwrote {total} polygons across {written} files in {outdir}")