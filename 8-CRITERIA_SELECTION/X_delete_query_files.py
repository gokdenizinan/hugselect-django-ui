from pathlib import Path

root = Path("/Users/ardacanseradali/Documents/Tez_master/8-CRITERIA_SELECTION/experiments")

for file in root.rglob("query.json"):
    file.unlink()
    print(f"Deleted: {file}")