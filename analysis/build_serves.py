"""Turn Match Charting Project point files into one row per serve attempt.

Reads data/raw/charting-{m,w}-{matches,points-*}.csv and writes data/serves.pkl.
Each row is a serve the server chose to hit: which serve (1st/2nd), where it was
aimed (wide/body/T), which side of the court, and whether the server won the point.
"""
from pathlib import Path
import glob

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "serves.pkl"

DIRS = {"4": "wide", "5": "body", "6": "T"}
FAULT_CHARS = set("ndwxeg!VWDNX")
REGULAR_POINTS = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}


def load_matches():
    frames = []
    for tour in ("m", "w"):
        m = pd.read_csv(RAW / f"charting-{tour}-matches.csv", dtype=str, encoding="latin-1")
        m["tour"] = "ATP" if tour == "m" else "WTA"
        frames.append(m)
    m = pd.concat(frames, ignore_index=True)
    m = m.rename(columns={"Player 1": "p1", "Player 2": "p2", "Pl 1 hand": "h1", "Pl 2 hand": "h2"})
    m["surface"] = m["Surface"].where(m["Surface"].isin(["Hard", "Clay", "Grass"]), "Other")
    m["date"] = pd.to_datetime(m["Date"], format="%Y%m%d", errors="coerce")
    for c in ("p1", "p2", "h1", "h2"):
        m[c] = m[c].fillna("").str.strip()
    return m[["match_id", "tour", "p1", "p2", "h1", "h2", "date", "Tournament", "surface"]]


def score_state(pts: pd.Series):
    """Court side and break-point flag from the server-first point score ("30-40")."""
    parts = pts.fillna("").str.split("-", n=1, expand=True)
    a, b = parts[0].str.strip(), parts[1].fillna("").str.strip()
    regular = a.isin(REGULAR_POINTS.keys()) & b.isin(REGULAR_POINTS.keys())
    # A tiebreak score ("3-2") counts points directly; a regular one maps 0/15/30/40/AD to 0..4.
    ra, rb = a.map(REGULAR_POINTS), b.map(REGULAR_POINTS)
    na, nb = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    total = np.where(regular, ra + rb, na + nb)
    side = np.where(pd.isna(total), None, np.where(np.nan_to_num(total) % 2 == 0, "deuce", "ad"))
    has_tb_numbers = ~regular & na.notna() & nb.notna()
    break_point = regular & (((rb == 3) & (ra <= 2)) | (rb == 4))
    return pd.Series(side, index=pts.index), break_point, has_tb_numbers


def serve_parts(code: pd.Series):
    """Aim and outcome flags from a raw serve code such as '6*', '4n' or 'c5+f1...'."""
    c = code.fillna("").str.strip().str.lstrip("c").str.replace("+", "", regex=False)
    aim = c.str[:1].map(DIRS)
    nxt = c.str[1:2]
    fault = nxt.isin(FAULT_CHARS) | c.str[:1].isin(FAULT_CHARS)
    ace = nxt.eq("*")
    unreturned = nxt.eq("#")
    return aim, fault, ace, unreturned


def main():
    matches = load_matches()
    pts = pd.concat(
        [pd.read_csv(f, dtype=str, encoding="latin-1") for f in sorted(glob.glob(str(RAW / "charting-*-points-*.csv")))],
        ignore_index=True,
    )
    pts = pts.merge(matches, on="match_id", how="inner")
    pts = pts[pts["Svr"].isin(["1", "2"]) & pts["PtWinner"].isin(["1", "2"])].copy()
    pts["pt"] = pd.to_numeric(pts["Pt"], errors="coerce")
    pts = pts.sort_values(["match_id", "pt"])

    s1 = pts["Svr"] == "1"
    pts["server"] = np.where(s1, pts["p1"], pts["p2"])
    pts["returner"] = np.where(s1, pts["p2"], pts["p1"])
    pts["server_hand"] = np.where(s1, pts["h1"], pts["h2"])
    pts["returner_hand"] = np.where(s1, pts["h2"], pts["h1"])
    pts["won"] = (pts["Svr"] == pts["PtWinner"]).astype(np.int8)
    pts["side"], pts["break_point"], pts["tiebreak"] = score_state(pts["Pts"])

    aim1, fault1_code, ace1, unret1 = serve_parts(pts["1st"])
    has_second = pts["2nd"].fillna("").str.strip().ne("")
    aim2, fault2, ace2, unret2 = serve_parts(pts["2nd"])
    # A first serve was a fault when the charter logged a second serve; the code letter agrees >99.9% of the time.
    fault1 = has_second

    keep = ["match_id", "pt", "tour", "date", "Tournament", "surface", "server", "returner",
            "server_hand", "returner_hand", "side", "break_point", "tiebreak", "won"]
    first = pts[keep].copy()
    first["serve"] = 1
    first["aim"] = aim1.values
    first["in"] = (~fault1).astype(np.int8).values
    first["ace"] = (ace1 & ~fault1).astype(np.int8).values
    first["unreturned"] = ((ace1 | unret1) & ~fault1).astype(np.int8).values

    second = pts.loc[has_second, keep].copy()
    second["serve"] = 2
    second["aim"] = aim2[has_second].values
    second["in"] = (~fault2[has_second]).astype(np.int8).values
    second["ace"] = (ace2[has_second] & ~fault2[has_second]).astype(np.int8).values
    second["unreturned"] = ((ace2 | unret2)[has_second] & ~fault2[has_second]).astype(np.int8).values

    serves = pd.concat([first, second], ignore_index=True)
    serves = serves[serves["aim"].notna() & serves["side"].notna() & (serves["server"] != "") & (serves["returner"] != "")]
    serves = serves.sort_values(["match_id", "pt", "serve"]).reset_index(drop=True)

    # Previous first-serve aim by the same server on the same side earlier in the match (for the predictability test).
    f = serves["serve"] == 1
    prev = serves[f].groupby(["match_id", "server", "side"])["aim"].shift(1)
    serves["prev_aim_same_side"] = None
    serves.loc[f, "prev_aim_same_side"] = prev

    serves.to_pickle(OUT)
    print(f"{len(pts):,} points from {pts['match_id'].nunique():,} matches -> {len(serves):,} serve attempts")
    print(serves.groupby(["tour", "serve", "aim"]).size().unstack())
    print(serves.groupby(["tour", "serve"])[["in", "won"]].mean().round(3))


if __name__ == "__main__":
    main()
