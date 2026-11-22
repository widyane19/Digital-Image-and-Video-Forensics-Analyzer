"""Download small samples of public datasets to add to the controlled dataset.

- vision: extra unedited phone photos and videos, added to data/originals/
- csvted: videos with frames deleted, duplicated or inserted by another team, in data/external/csvted/

Run:  python -m src.external_datasets {vision,csvted} --data ../data [--dry-run]
"""
