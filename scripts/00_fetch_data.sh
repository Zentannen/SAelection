#!/usr/bin/env bash
# Downloads the raw inputs (not committed: ~1 GB unpacked).
#  * VD-level IEC results (2016 LGE, 2019 PE, 2021 LGE, 2024 PE), as republished in
#    github.com/xyclone-designs/database.iec (.input/data.zip, sourced from results.elections.org.za)
#  * Municipal Demarcation Board / IEC voting-district + ward boundaries (2021, 2024, 2026),
#    as mirrored (Git LFS) in github.com/garyvdm/demarcation-compare
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"; GEO="$ROOT/data/geo"; TMP="$ROOT/data/cache"
mkdir -p "$RAW" "$GEO" "$TMP"

if [ ! -f "$RAW/2024 PE.csv" ]; then
  [ -d "$TMP/database.iec" ] || GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/xyclone-designs/database.iec "$TMP/database.iec"
  unzip -o -q "$TMP/database.iec/.input/data.zip" "2016 LGE GP.csv" "2021 LGE GP.csv" "2019 PE.csv" "2024 PE.csv" -d "$RAW"
fi

B=https://media.githubusercontent.com/media/garyvdm/demarcation-compare/master/source-data/demarcations
for p in 2026/voting-district.gpkg.zip 2026/ward.gpkg.zip 2024/voting-district.shapefile.zip 2021/voting-district.shapefile.zip; do
  f="$GEO/$(echo "$p" | tr / _)"
  [ -f "$f" ] || curl -sSL --retry 3 -o "$f" "$B/$p"
  d="${f%.zip}"; [ -d "$d" ] || { mkdir -p "$d"; unzip -o -q "$f" -d "$d"; }
done
echo "raw data ready in $RAW and $GEO"
