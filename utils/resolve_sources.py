"""Resolve RA/Dec positions into everything Cutana needs.

Input: a table with RA and Dec (any of ra/dec, right_ascension/declination,
target_ra/target_dec, case-insensitive). Output: one row per source with

    SourceID     "{tile_index}_{object_id}" (negative ids written "NEG<abs>")
    RA, Dec      the positions you gave
    tile_index   tile whose MER_FINAL_CATALOG holds the object
    release_dir  release that tile is read from (the Q1 mount)
    object_id    MER OBJECT_ID of the nearest catalogue entry
    match_sep_arcsec, matched

Columns you already have are trusted and kept: tile_index (aliases tile_id,
tile), release_dir and SourceID (aliases source_id, id). object_id is NOT
trusted: it is always re-read from the catalogue. A row with
tile_index + SourceID skips the lookup entirely; a row with only tile_index
gets its object_id from that tile's catalogue; a row with neither is resolved
from the position as below. resolve_sources(..., check=True) warns about any
given tile that does not contain its position.

How: utils/tile_centres_q1.csv lists every tile's centre and half-size. For each
position the tiles that could contain it are tried nearest tile centre first,
and the first MER_FINAL_CATALOG with an entry within MATCH_RADIUS_ARCSEC wins
(tiles overlap, so the position alone does not say which catalogue holds the
object). A source with no entry that close is
kept with matched=False and a position-based SourceID, read from the nearest
tile, so it still gets a cutout.

The same sky object can be catalogued in two overlapping tiles; you get the
first one in the order above, so its tile prefix may differ from a SourceID
made elsewhere.
"""

import glob
import logging
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.spatial import cKDTree

MATCH_RADIUS_ARCSEC = 1.0
_HERE = os.path.dirname(os.path.abspath(__file__))
# Per release: tile table (in utils/), preference rank of each table's `release` label
# (lower = preferred), and the mount directories, preferred first.
RELEASES = {
    "q1": {"table": "tile_centres_q1.csv", "rank": {"Q1_R1": 0},
           "dirs": ["/media/home/data/euclid_q1/Q1_R1"]},
}


def available_releases():
    return [r for r, v in RELEASES.items() if os.path.exists(os.path.join(_HERE, v["table"]))]

_RA_ALIASES = ["ra", "right_ascension", "target_ra"]
_DEC_ALIASES = ["dec", "declination", "target_dec"]
_TILE_ALIASES = ["tile_index", "tile_id", "tile"]
_ID_ALIASES = ["sourceid", "source_id", "id"]


def _xyz(ra, dec):
    ra, dec = np.deg2rad(ra), np.deg2rad(dec)
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def _chord(arcsec):
    return 2 * np.sin(np.deg2rad(arcsec / 3600.0) / 2)


def _normalise_radec(df):
    cols = {c.lower().strip(): c for c in df.columns}
    ra = next((cols[a] for a in _RA_ALIASES if a in cols), None)
    dec = next((cols[a] for a in _DEC_ALIASES if a in cols), None)
    if ra is None or dec is None:
        raise ValueError(f"need RA/Dec columns (accepted {_RA_ALIASES} / {_DEC_ALIASES}), "
                         f"found {list(df.columns)}")
    return df.rename(columns={ra: "RA", dec: "Dec"})


def _final_catalogue(tile, release_dir):
    hits = glob.glob(os.path.join(release_dir, "MER_FINAL_CATALOG", str(tile),
                                  "EUC_MER_FINAL-CAT_TILE*.fits"))
    if not hits:
        return None
    with fits.open(hits[0], memmap=True) as hdul:
        t = hdul[1].data
        oid = np.asarray(t["OBJECT_ID"], dtype=np.int64)
        xyz = _xyz(np.asarray(t["RIGHT_ASCENSION"], dtype=np.float64),
                   np.asarray(t["DECLINATION"], dtype=np.float64))
    return oid, cKDTree(xyz)


def make_source_id(tile, object_id):
    return f"{tile}_NEG{abs(object_id)}" if object_id < 0 else f"{tile}_{object_id}"


def _candidate_tiles(xyz, tiles):
    """Per source: row indices into `tiles` whose footprint may hold it, preferred first."""
    centres = _xyz(tiles["ra"].values, tiles["dec"].values)
    reach = np.deg2rad(tiles["half_size_deg"].values.max() * 1.5)
    tree = cKDTree(centres)
    out = []
    for p, near in zip(xyz, tree.query_ball_point(xyz, 2 * np.sin(reach / 2))):
        near = np.asarray(near, dtype=int)
        d = np.linalg.norm(centres[near] - p, axis=1)
        inside = d <= 2 * np.sin(np.deg2rad(tiles["half_size_deg"].values[near] * 1.5) / 2)
        near, d = near[inside], d[inside]
        order = np.lexsort((d, tiles["rank"].values[near]))
        out.append(near[order])
    return out


def _rename_alias(df, aliases, new):
    cols = {c.lower().strip(): c for c in df.columns}
    for a in aliases:
        if a in cols:
            return df.rename(columns={cols[a]: new}) if cols[a] != new else df
    return df


def _default_release(tile, dirs):
    for d in dirs:
        if os.path.isdir(os.path.join(d, "MER", str(tile))):
            return d
    return None


def _match_in_tile(cache, tile, release_dir, xyz, idx, max_chord):
    """Nearest catalogue entry of one given tile for sources `idx`: (object_id, sep) or None."""
    key = (int(tile), release_dir)
    if key not in cache:
        cache[key] = _final_catalogue(*key)
        if len(cache) > 6:
            cache.popitem(last=False)
    if cache[key] is None:
        return None
    ids, tree = cache[key]
    dist, j = tree.query(xyz[idx])
    return ids[j], np.rad2deg(2 * np.arcsin(dist / 2)) * 3600.0, dist <= max_chord


def resolve_sources(sources, match_radius_arcsec=MATCH_RADIUS_ARCSEC, release=None,
                    check=False):
    """`sources`: DataFrame or CSV path with RA/Dec columns (extra columns are kept).

    release: "q1" (the only release shipped here).
    check=True: warn about given tiles that do not contain their position.
    """
    avail = available_releases()
    release = release or (avail[0] if avail else None)
    if release not in avail:
        raise ValueError(f"release must be one of {avail}, got {release!r}")
    cfg = RELEASES[release]
    df = pd.read_csv(sources) if isinstance(sources, (str, os.PathLike)) else sources.copy()
    df = _normalise_radec(df).reset_index(drop=True)
    df = _rename_alias(df, _TILE_ALIASES, "tile_index")
    df = _rename_alias(df, _ID_ALIASES, "SourceID")
    n = len(df)
    for col in ("tile_index", "release_dir", "SourceID", "object_id"):
        if col not in df.columns:
            df[col] = None

    tiles = pd.read_csv(os.path.join(_HERE, cfg["table"]))
    tile_release = dict(zip(tiles["tile_index"], tiles["release_dir"]))
    tiles["rank"] = tiles["release"].map(cfg["rank"])
    tiles = tiles.dropna(subset=["rank"]).reset_index(drop=True)

    xyz = _xyz(df["RA"].values, df["Dec"].values)
    tile = pd.to_numeric(df["tile_index"], errors="coerce").fillna(0).astype(np.int64).to_numpy(copy=True)
    given_tile = tile > 0
    rdir = np.array([None if pd.isna(v) else v for v in df["release_dir"]], dtype=object)
    sid = np.array([None if pd.isna(v) or str(v).strip() == "" else str(v) for v in df["SourceID"]],
                   dtype=object)
    oid = pd.array(df["object_id"], dtype="Int64")
    sep = np.full(n, np.nan)
    matched = np.array([s is not None for s in sid])        # a given SourceID counts as resolved
    searched = np.zeros(n, dtype=bool)                      # tile found from the position
    max_chord = _chord(match_radius_arcsec)
    cache = OrderedDict()           # (tile, release_dir) -> catalogue; a few at a time

    # given tile: default release (the table's)
    for i in np.flatnonzero(given_tile & (rdir == None)):    # noqa: E711
        rdir[i] = tile_release.get(tile[i]) or _default_release(tile[i], cfg["dirs"])

    # given tile, no SourceID: object_id from that tile's catalogue
    need = np.flatnonzero(given_tile & ~matched & (rdir != None))   # noqa: E711
    for (t, rd), grp in pd.Series(need).groupby([tile[need], rdir[need]]):
        idx = grp.values
        res = _match_in_tile(cache, t, rd, xyz, idx, max_chord)
        if res is None:
            continue
        ids, seps, ok = res
        for i, o, sp, k in zip(idx, ids, seps, ok):
            if k:
                matched[i], oid[i], sep[i] = True, o, sp

    # no tile: find it from the position (tile_centres + catalogue match)
    pos = np.flatnonzero(~given_tile)
    cands = _candidate_tiles(xyz[pos], tiles) if len(pos) else []
    for k in range(max((len(c) for c in cands), default=0)):
        by_tile = {}
        for li, i in enumerate(pos):
            if not searched[i] and len(cands[li]) > k:
                by_tile.setdefault(int(cands[li][k]), []).append(i)
        for row, idx in by_tile.items():
            t, rd = int(tiles.at[row, "tile_index"]), tiles.at[row, "release_dir"]
            res = _match_in_tile(cache, t, rd, xyz, idx, max_chord)
            if res is None:
                continue
            ids, seps, ok = res
            for i, o, sp, kk in zip(idx, ids, seps, ok):
                if kk:
                    searched[i] = matched[i] = True
                    tile[i], rdir[i], oid[i], sep[i] = t, rd, o, sp
    for li, i in enumerate(pos):                      # unmatched: nearest preferred tile
        if tile[i] == 0 and len(cands[li]):
            row = cands[li][0]
            tile[i], rdir[i] = int(tiles.at[row, "tile_index"]), tiles.at[row, "release_dir"]

    if check and given_tile.any():
        _check_tiles(df, tile, given_tile, tiles)

    df["tile_index"], df["release_dir"] = tile, rdir
    df["object_id"], df["match_sep_arcsec"], df["matched"] = oid, sep, matched

    outside = pd.isna(df["release_dir"]) | (df["tile_index"] == 0)
    if outside.any():
        logging.warning("%d source(s) outside every Q1 tile / tile not on this machine -- "
                        "dropped: rows %s", outside.sum(), list(df.index[outside][:20]))
    keep = ~outside.values
    df, sid, oidk = df[keep].reset_index(drop=True), sid[keep], df.loc[keep, "object_id"].values
    nomatch = ~df["matched"].values
    if nomatch.any():
        logging.warning("%d source(s) have no MER catalogue entry within %.1f\" -- given a "
                        "position-based SourceID", nomatch.sum(), match_radius_arcsec)
    df["SourceID"] = [
        s if s is not None else
        (make_source_id(t, int(o)) if m and not pd.isna(o) else f"{t}_RA{r:.6f}_DEC{d:.6f}")
        for s, t, o, m, r, d in zip(sid, df["tile_index"], oidk, df["matched"], df["RA"], df["Dec"])
    ]
    first = ["SourceID", "RA", "Dec", "tile_index", "release_dir", "object_id",
             "match_sep_arcsec", "matched"]
    return df[first + [c for c in df.columns if c not in first]]


def _check_tiles(df, tile, given_tile, tiles, slack=1.05):
    """Warn when a given tile's footprint (square of the tile's half-size) misses the position."""
    info = tiles.drop_duplicates("tile_index").set_index("tile_index")
    bad, unknown = [], []
    for i in np.flatnonzero(given_tile):
        if tile[i] not in info.index:
            unknown.append(i)
            continue
        r = info.loc[tile[i]]
        dra = (df["RA"].iat[i] - r["ra"] + 180) % 360 - 180
        dra *= np.cos(np.deg2rad(df["Dec"].iat[i]))
        if abs(dra) > r["half_size_deg"] * slack or abs(df["Dec"].iat[i] - r["dec"]) > r["half_size_deg"] * slack:
            bad.append(i)
    if bad:
        logging.warning("check: %d source(s) lie outside their given tile -- rows %s",
                        len(bad), bad[:20])
    if unknown:
        logging.warning("check: %d given tile(s) not in the tile table -- rows %s",
                        len(unknown), unknown[:20])
