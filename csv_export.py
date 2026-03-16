"""
csv_export.py
~~~~~~~~~~~~~
Export football data tables to CSV files.

Usage
-----
    python csv_export.py [--db /path/to/soccer.db] [--output /path/to/output/]
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, inspect

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def export_tables_from_db(db_path: str, output_dir: str = ".") -> None:
    """
    Export all tables from SQLite database to CSV files.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    output_dir : str
        Directory to write CSV files to (default: current directory).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create SQLAlchemy engine
    engine = create_engine(f"sqlite:///{db_path}")

    # Inspect database to get all table names
    inspector = inspect(engine)
    tables = inspector.get_table_names()

    if not tables:
        logger.warning(f"No tables found in database: {db_path}")
        return

    logger.info(f"Found {len(tables)} table(s): {', '.join(tables)}")

    # Export each table to CSV
    for table_name in tables:
        try:
            df = pd.read_sql_table(table_name, engine)
            csv_path = output_path / f"{table_name}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Exported {table_name} ({len(df)} rows) → {csv_path}")
        except Exception as e:
            logger.error(f"Failed to export {table_name}: {e}")

    engine.dispose()


def export_dataframes_to_csv(
    competitions_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    output_dir: str = ".",
) -> None:
    """
    Export DataFrames to CSV files.

    Parameters
    ----------
    competitions_df : pd.DataFrame
        Competitions table DataFrame.
    teams_df : pd.DataFrame
        Teams table DataFrame.
    matches_df : pd.DataFrame
        Matches table DataFrame.
    output_dir : str
        Directory to write CSV files to (default: current directory).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tables = {
        "competitions": competitions_df,
        "teams": teams_df,
        "matches": matches_df,
    }

    for table_name, df in tables.items():
        csv_path = output_path / f"{table_name}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Exported {table_name} ({len(df)} rows) → {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export football data tables to CSV files."
    )
    parser.add_argument(
        "--db",
        type=str,
        default="soccer.db",
        help="Path to SQLite database file (default: soccer.db).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Output directory for CSV files (default: current directory).",
    )

    args = parser.parse_args()

    # Check if database exists
    db_file = Path(args.db)
    if not db_file.exists():
        logger.error(f"Database file not found: {args.db}")
        exit(1)

    export_tables_from_db(str(db_file), args.output)
