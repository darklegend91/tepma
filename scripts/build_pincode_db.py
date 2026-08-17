#!/usr/bin/env python3
"""Build the local PIN lookup database from the Department of Posts CSV export.

Run this once during setup, and rerun it whenever you want a fresher postal snapshot:

    .venv/bin/python scripts/build_pincode_db.py

The database is built atomically, so an interrupted refresh cannot damage the file
currently used by the application.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import httpx


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "reference" / "india_post_pincodes.sqlite3"
RESOURCE_URL = (
    "https://api.data.gov.in/resource/5c2f62fe-5afa-4119-a499-fec9d604d5bd"
)
PUBLIC_API_KEY = "579b464db66ec23bdd000001cdc3b564546246a772a26393094f5645"
PIN_RE = re.compile(r"^[1-9][0-9]{5}$")
CSV_FIELDS = {
    "circlename",
    "regionname",
    "divisionname",
    "officename",
    "pincode",
    "officetype",
    "delivery",
    "district",
    "statename",
}


def download_csv(destination: Path, api_key: str) -> None:
    """Stream the complete official CSV export to a temporary file."""
    timeout = httpx.Timeout(180.0, connect=15.0)
    with httpx.stream(
        "GET",
        RESOURCE_URL,
        params={
            "api-key": api_key,
            "format": "csv",
            "offset": 0,
            "limit": "all",
        },
        headers={"Accept": "text/csv", "User-Agent": "TePMA-pincode-builder/1.0"},
        timeout=timeout,
        follow_redirects=True,
    ) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_bytes():
                output.write(chunk)

    with destination.open("r", encoding="utf-8-sig", newline="") as source:
        header = next(csv.reader(source), [])
    if not CSV_FIELDS.issubset(set(header)):
        raise RuntimeError("The downloaded file is not the expected postal CSV export")


def clean(row: dict[str, str], field: str) -> str:
    return re.sub(r"\s+", " ", row.get(field) or "").strip()


def build_database(csv_path: Path, output_path: Path) -> tuple[int, int]:
    """Create a pincode-clustered SQLite database and atomically install it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_db = output_path.with_name(f".{output_path.name}.building")
    temporary_db.unlink(missing_ok=True)

    connection = sqlite3.connect(temporary_db)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode = OFF;
            PRAGMA synchronous = OFF;
            PRAGMA temp_store = MEMORY;

            CREATE TABLE post_offices (
                pincode TEXT NOT NULL,
                officename TEXT NOT NULL,
                district TEXT NOT NULL,
                statename TEXT NOT NULL,
                divisionname TEXT NOT NULL,
                circlename TEXT NOT NULL,
                regionname TEXT NOT NULL,
                officetype TEXT NOT NULL,
                delivery TEXT NOT NULL,
                PRIMARY KEY (
                    pincode, officename, district, statename, divisionname,
                    circlename, regionname, officetype, delivery
                )
            ) WITHOUT ROWID;

            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
            """
        )

        insert = """
            INSERT OR IGNORE INTO post_offices (
                pincode, officename, district, statename, divisionname,
                circlename, regionname, officetype, delivery
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        batch: list[tuple[str, ...]] = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames or not CSV_FIELDS.issubset(set(reader.fieldnames)):
                raise RuntimeError("Postal CSV columns are missing or have changed")
            for row in reader:
                pin = clean(row, "pincode")
                office = clean(row, "officename")
                if not PIN_RE.fullmatch(pin) or not office:
                    continue
                batch.append((
                    pin,
                    office,
                    clean(row, "district"),
                    clean(row, "statename"),
                    clean(row, "divisionname"),
                    clean(row, "circlename"),
                    clean(row, "regionname"),
                    clean(row, "officetype"),
                    clean(row, "delivery"),
                ))
                if len(batch) >= 5000:
                    connection.executemany(insert, batch)
                    batch.clear()
            if batch:
                connection.executemany(insert, batch)

        record_count = connection.execute(
            "SELECT COUNT(*) FROM post_offices"
        ).fetchone()[0]
        pin_count = connection.execute(
            "SELECT COUNT(DISTINCT pincode) FROM post_offices"
        ).fetchone()[0]
        if record_count < 100_000 or pin_count < 10_000:
            raise RuntimeError(
                f"Postal snapshot looks incomplete: {record_count} offices, {pin_count} PINs"
            )

        metadata = {
            "source": "Department of Posts via data.gov.in",
            "source_url": RESOURCE_URL,
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "record_count": str(record_count),
            "pincode_count": str(pin_count),
        }
        connection.executemany(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            metadata.items(),
        )
        connection.commit()
        connection.execute("VACUUM")
    except Exception:
        connection.close()
        temporary_db.unlink(missing_ok=True)
        raise
    else:
        connection.close()

    os.replace(temporary_db, output_path)
    return record_count, pin_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        help="Use an existing Department of Posts CSV instead of downloading it.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--api-key",
        default=os.getenv("DATA_GOV_IN_API_KEY", PUBLIC_API_KEY),
        help="data.gov.in API key; defaults to DATA_GOV_IN_API_KEY or the public key.",
    )
    args = parser.parse_args()

    if args.csv:
        records, pins = build_database(args.csv.resolve(), args.output.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="tepma-pincodes-") as directory:
            csv_path = Path(directory) / "pincodes.csv"
            print("Downloading the Department of Posts PIN directory...")
            download_csv(csv_path, args.api_key)
            records, pins = build_database(csv_path, args.output.resolve())

    size_mb = args.output.resolve().stat().st_size / (1024 * 1024)
    print(
        f"Created {args.output.resolve()}\n"
        f"Indexed {records:,} post offices across {pins:,} PINs ({size_mb:.1f} MiB)."
    )


if __name__ == "__main__":
    main()
