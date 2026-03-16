"""
tests/test_database.py
~~~~~~~~~~~~~~~~~~~~~~
Unit tests for football_data.py.

These tests exercise the DataFrame builders and the database writer without
requiring a live Football-Data.org API key.
"""

import os
import sqlite3
import tempfile

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
from football_data import (
    build_competitions_df,
    build_matches_df,
    build_teams_df,
    save_to_database,
)

# ---------------------------------------------------------------------------
# Fixtures – minimal API-shaped payloads
# ---------------------------------------------------------------------------

RAW_COMPETITIONS = [
    {
        "code": "PL",
        "area": {"name": "England"},
        "currentSeason": {"startDate": "2023-08-11"},
    },
    {
        "code": "CL",
        "area": {"name": "Europe"},
        "currentSeason": {"startDate": "2023-09-19"},
    },
]

RAW_TEAMS = {
    "PL": [
        {"id": 57, "name": "Arsenal FC"},
        {"id": 61, "name": "Chelsea FC"},
    ],
    "CL": [
        {"id": 57, "name": "Arsenal FC"},   # duplicate – should be de-duped
        {"id": 86, "name": "Real Madrid CF"},
    ],
}

RAW_MATCHES = {
    "PL": [
        {
            "id": 1001,
            "season": {"startDate": "2023-08-11"},
            "utcDate": "2023-08-12T14:00:00Z",
            "matchday": 1,
            "homeTeam": {"id": 57},
            "awayTeam": {"id": 61},
            "score": {
                "fullTime": {"home": 2, "away": 1},
                "winner": "HOME_TEAM",
            },
        },
    ],
    "CL": [
        {
            "id": 2001,
            "season": {"startDate": "2023-09-19"},
            "utcDate": "2023-09-20T20:00:00Z",
            "matchday": 1,
            "homeTeam": {"id": 86},
            "awayTeam": {"id": 57},
            "score": {
                "fullTime": {"home": 1, "away": 0},
                "winner": "HOME_TEAM",
            },
        },
    ],
}


# ---------------------------------------------------------------------------
# Tests: build_competitions_df
# ---------------------------------------------------------------------------

class TestBuildCompetitionsDf:
    def test_columns(self):
        df = build_competitions_df(RAW_COMPETITIONS)
        assert list(df.columns) == ["competition", "country", "season"]

    def test_row_count(self):
        df = build_competitions_df(RAW_COMPETITIONS)
        assert len(df) == 2

    def test_values(self):
        df = build_competitions_df(RAW_COMPETITIONS)
        pl_row = df[df["competition"] == "PL"].iloc[0]
        assert pl_row["country"] == "England"
        assert pl_row["season"] == "2023"

    def test_deduplication(self):
        dupes = RAW_COMPETITIONS + RAW_COMPETITIONS
        df = build_competitions_df(dupes)
        assert len(df) == 2

    def test_empty_input(self):
        df = build_competitions_df([])
        assert list(df.columns) == ["competition", "country", "season"]
        assert len(df) == 0

    def test_missing_area(self):
        raw = [{"code": "XX", "area": None, "currentSeason": {"startDate": "2023-01-01"}}]
        df = build_competitions_df(raw)
        assert df.iloc[0]["country"] == ""

    def test_missing_season(self):
        raw = [{"code": "YY", "area": {"name": "Test"}, "currentSeason": None}]
        df = build_competitions_df(raw)
        assert df.iloc[0]["season"] == ""


# ---------------------------------------------------------------------------
# Tests: build_teams_df
# ---------------------------------------------------------------------------

class TestBuildTeamsDf:
    def test_columns(self):
        df = build_teams_df(RAW_TEAMS)
        assert list(df.columns) == ["team_id", "team_name"]

    def test_deduplication(self):
        # Arsenal appears in both PL and CL; should only be one row.
        df = build_teams_df(RAW_TEAMS)
        assert len(df[df["team_id"] == 57]) == 1

    def test_total_unique_teams(self):
        df = build_teams_df(RAW_TEAMS)
        # Arsenal(57), Chelsea(61), Real Madrid(86) → 3 unique
        assert len(df) == 3

    def test_empty_input(self):
        df = build_teams_df({})
        assert list(df.columns) == ["team_id", "team_name"]
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Tests: build_matches_df
# ---------------------------------------------------------------------------

class TestBuildMatchesDf:
    def test_columns(self):
        df = build_matches_df(RAW_MATCHES)
        expected = [
            "match_id", "competition", "season", "date", "matchday",
            "home_team_id", "away_team_id", "home_goals", "away_goals", "winner",
        ]
        assert list(df.columns) == expected

    def test_row_count(self):
        df = build_matches_df(RAW_MATCHES)
        assert len(df) == 2

    def test_values(self):
        df = build_matches_df(RAW_MATCHES)
        pl_match = df[df["match_id"] == 1001].iloc[0]
        assert pl_match["competition"] == "PL"
        assert pl_match["season"] == "2023"
        assert pl_match["date"] == "2023-08-12"
        assert pl_match["matchday"] == 1
        assert pl_match["home_team_id"] == 57
        assert pl_match["away_team_id"] == 61
        assert pl_match["home_goals"] == 2
        assert pl_match["away_goals"] == 1
        assert pl_match["winner"] == "HOME_TEAM"

    def test_deduplication(self):
        duped = {"PL": RAW_MATCHES["PL"] + RAW_MATCHES["PL"], "CL": RAW_MATCHES["CL"]}
        df = build_matches_df(duped)
        assert len(df) == 2

    def test_empty_input(self):
        df = build_matches_df({})
        assert len(df) == 0

    def test_null_score(self):
        """Matches not yet played may have null fullTime scores."""
        raw = {
            "PL": [
                {
                    "id": 9999,
                    "season": {"startDate": "2023-08-11"},
                    "utcDate": "2024-05-01T15:00:00Z",
                    "matchday": 38,
                    "homeTeam": {"id": 57},
                    "awayTeam": {"id": 61},
                    "score": {
                        "fullTime": {"home": None, "away": None},
                        "winner": None,
                    },
                }
            ]
        }
        df = build_matches_df(raw)
        row = df.iloc[0]
        assert row["home_goals"] is None or pd.isna(row["home_goals"])
        assert row["winner"] is None


# ---------------------------------------------------------------------------
# Tests: save_to_database
# ---------------------------------------------------------------------------

class TestSaveToDatabase:
    def _sample_dfs(self):
        competitions_df = build_competitions_df(RAW_COMPETITIONS)
        teams_df = build_teams_df(RAW_TEAMS)
        matches_df = build_matches_df(RAW_MATCHES)
        return competitions_df, teams_df, matches_df

    def test_tables_created(self):
        comps, teams, matches = self._sample_dfs()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            save_to_database(comps, teams, matches, db_path)
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
                tables = {row[0] for row in cursor.fetchall()}
            assert {"competitions", "teams", "matches"} == tables
        finally:
            os.unlink(db_path)

    def test_competitions_schema(self):
        comps, teams, matches = self._sample_dfs()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            save_to_database(comps, teams, matches, db_path)
            df = pd.read_sql("SELECT * FROM competitions", sqlite3.connect(db_path))
            assert set(df.columns) == {"competition", "country", "season"}
        finally:
            os.unlink(db_path)

    def test_teams_schema(self):
        comps, teams, matches = self._sample_dfs()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            save_to_database(comps, teams, matches, db_path)
            df = pd.read_sql("SELECT * FROM teams", sqlite3.connect(db_path))
            assert set(df.columns) == {"team_id", "team_name"}
        finally:
            os.unlink(db_path)

    def test_matches_schema(self):
        comps, teams, matches = self._sample_dfs()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            save_to_database(comps, teams, matches, db_path)
            df = pd.read_sql("SELECT * FROM matches", sqlite3.connect(db_path))
            expected_cols = {
                "match_id", "competition", "season", "date", "matchday",
                "home_team_id", "away_team_id", "home_goals", "away_goals", "winner",
            }
            assert set(df.columns) == expected_cols
        finally:
            os.unlink(db_path)

    def test_row_counts_in_db(self):
        comps, teams, matches = self._sample_dfs()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            save_to_database(comps, teams, matches, db_path)
            conn = sqlite3.connect(db_path)
            assert conn.execute("SELECT COUNT(*) FROM competitions").fetchone()[0] == 2
            assert conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0] == 3
            assert conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 2
            conn.close()
        finally:
            os.unlink(db_path)

    def test_replace_on_rerun(self):
        """Calling save_to_database twice should not duplicate rows."""
        comps, teams, matches = self._sample_dfs()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            save_to_database(comps, teams, matches, db_path)
            save_to_database(comps, teams, matches, db_path)
            conn = sqlite3.connect(db_path)
            assert conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 2
            conn.close()
        finally:
            os.unlink(db_path)
