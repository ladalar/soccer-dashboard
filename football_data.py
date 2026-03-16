"""
football_data.py
~~~~~~~~~~~~~~~~R
Fetches data from the Football-Data.org API and builds a SQLite relational
database with three tables:

  - competitions  (competition, country, season)
  - teams         (team_id, team_name)
  - matches       (match_id, competition, season, date, matchday,
                   home_team_id, away_team_id, home_goals, away_goals, winner)

Usage
-----
Set the environment variable FOOTBALL_API_KEY to your Football-Data.org token,
then run:

    python football_data.py

The database is written to ``soccer.db`` in the current directory by default.
You can override the path via the --db flag:

    python football_data.py --db /path/to/output.db

API reference: https://www.football-data.org/documentation/quickstart
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://api.football-data.org/v4"

# Default competition codes to fetch.  These are available on the free tier.
DEFAULT_COMPETITIONS = ["PL", "PD", "BL1", "SA", "FL1", "CL"]


def load_dotenv(dotenv_path: str = ".env") -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ."""
    env_file = Path(dotenv_path)
    if not env_file.is_file():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get(endpoint: str, api_key: str, params: Optional[dict] = None) -> dict:
    """Make an authenticated GET request to the Football-Data.org API."""
    headers = {"X-Auth-Token": api_key}
    url = f"{BASE_URL}/{endpoint}"
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_competitions(api_key: str) -> list[dict]:
    """Return a list of available competitions."""
    data = _get("competitions", api_key)
    return data.get("competitions", [])


def fetch_teams(api_key: str, competition_code: str) -> list[dict]:
    """Return teams participating in *competition_code*."""
    data = _get(f"competitions/{competition_code}/teams", api_key)
    return data.get("teams", [])


def fetch_matches(api_key: str, competition_code: str) -> list[dict]:
    """Return all matches for *competition_code* (current season)."""
    data = _get(f"competitions/{competition_code}/matches", api_key)
    return data.get("matches", [])


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------

def build_competitions_df(raw_competitions: list[dict]) -> pd.DataFrame:
    """
    Build the *competitions* table.

    Parameters
    ----------
    raw_competitions:
        List of competition objects as returned by the API.

    Returns
    -------
    pd.DataFrame with columns: competition, country, season
    """
    rows = []
    for comp in raw_competitions:
        rows.append(
            {
                "competition": comp.get("code", ""),
                "country": (comp.get("area") or {}).get("name", ""),
                "season": (comp.get("currentSeason") or {}).get("startDate", "")[:4],
            }
        )
    df = pd.DataFrame(rows, columns=["competition", "country", "season"])
    return df.drop_duplicates(subset=["competition", "season"]).reset_index(drop=True)


def build_teams_df(raw_teams_by_comp: dict[str, list[dict]]) -> pd.DataFrame:
    """
    Build the *teams* table.

    Parameters
    ----------
    raw_teams_by_comp:
        Mapping of competition_code -> list of team objects.

    Returns
    -------
    pd.DataFrame with columns: team_id, team_name
    """
    rows = []
    for teams in raw_teams_by_comp.values():
        for team in teams:
            rows.append(
                {
                    "team_id": team.get("id"),
                    "team_name": team.get("name", ""),
                }
            )
    df = pd.DataFrame(rows, columns=["team_id", "team_name"])
    return df.drop_duplicates(subset=["team_id"]).reset_index(drop=True)


def build_matches_df(raw_matches_by_comp: dict[str, list[dict]]) -> pd.DataFrame:
    """
    Build the *matches* table.

    Parameters
    ----------
    raw_matches_by_comp:
        Mapping of competition_code -> list of match objects.

    Returns
    -------
    pd.DataFrame with columns:
        match_id, competition, season, date, matchday,
        home_team_id, away_team_id, home_goals, away_goals, winner
    """
    rows = []
    for comp_code, matches in raw_matches_by_comp.items():
        for match in matches:
            score = match.get("score") or {}
            full_time = score.get("fullTime") or {}
            season = match.get("season") or {}
            rows.append(
                {
                    "match_id": match.get("id"),
                    "competition": comp_code,
                    "season": str(season.get("startDate", ""))[:4],
                    "date": match.get("utcDate", "")[:10],
                    "matchday": match.get("matchday"),
                    "home_team_id": (match.get("homeTeam") or {}).get("id"),
                    "away_team_id": (match.get("awayTeam") or {}).get("id"),
                    "home_goals": full_time.get("home"),
                    "away_goals": full_time.get("away"),
                    "winner": score.get("winner"),
                }
            )
    columns = [
        "match_id", "competition", "season", "date", "matchday",
        "home_team_id", "away_team_id", "home_goals", "away_goals", "winner",
    ]
    df = pd.DataFrame(rows, columns=columns)
    return df.drop_duplicates(subset=["match_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Database writer
# ---------------------------------------------------------------------------

def save_to_database(
    competitions_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    db_path: str = "soccer.db",
) -> None:
    """
    Persist all three DataFrames to a SQLite database.

    Existing tables are replaced on each run so the data is always up to date.
    """
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        competitions_df.to_sql("competitions", conn, if_exists="replace", index=False)
        teams_df.to_sql("teams", conn, if_exists="replace", index=False)
        matches_df.to_sql("matches", conn, if_exists="replace", index=False)
    logger.info("Database written to %s", db_path)
    logger.info("  competitions: %d rows", len(competitions_df))
    logger.info("  teams:        %d rows", len(teams_df))
    logger.info("  matches:      %d rows", len(matches_df))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_database(
    api_key: str,
    competition_codes: list[str] = DEFAULT_COMPETITIONS,
    db_path: str = "soccer.db",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Orchestrate fetching and database creation.

    Returns
    -------
    Tuple of (competitions_df, teams_df, matches_df)
    """
    logger.info("Fetching competitions …")
    raw_comps = fetch_competitions(api_key)
    # Filter to the codes we care about
    raw_comps = [c for c in raw_comps if c.get("code") in competition_codes]

    raw_teams: dict[str, list[dict]] = {}
    raw_matches: dict[str, list[dict]] = {}

    for code in competition_codes:
        logger.info("Fetching teams for %s …", code)
        try:
            raw_teams[code] = fetch_teams(api_key, code)
        except requests.HTTPError as exc:
            logger.warning("Could not fetch teams for %s: %s", code, exc)
            raw_teams[code] = []

        logger.info("Fetching matches for %s …", code)
        try:
            raw_matches[code] = fetch_matches(api_key, code)
        except requests.HTTPError as exc:
            logger.warning("Could not fetch matches for %s: %s", code, exc)
            raw_matches[code] = []

    competitions_df = build_competitions_df(raw_comps)
    teams_df = build_teams_df(raw_teams)
    matches_df = build_matches_df(raw_matches)

    save_to_database(competitions_df, teams_df, matches_df, db_path)
    return competitions_df, teams_df, matches_df


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Fetch Football-Data.org data and build a SQLite database."
    )
    parser.add_argument(
        "--db",
        default="soccer.db",
        help="Path for the output SQLite database (default: soccer.db)",
    )
    parser.add_argument(
        "--competitions",
        nargs="+",
        default=DEFAULT_COMPETITIONS,
        help="Space-separated list of competition codes (default: PL PD BL1 SA FL1 CL)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("FOOTBALL_API_KEY", "")
    if not api_key:
        parser.error(
            "Set FOOTBALL_API_KEY in your environment or in .env."
        )

    build_database(api_key, args.competitions, args.db)


if __name__ == "__main__":
    main()
