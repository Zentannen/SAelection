# Live 2026 results go here

Drop IEC voting-district result CSVs for the 2026 Local Government Elections in this folder
(the per-VD download from results.elections.org.za, same columns as the 2021 LGE files:
`Province, Municipality, Ward, VotingDistrict, VotingStationName, RegisteredVoters, BallotType,
SpoiltVotes, PartyName, TotalValidVotes`). Only Gauteng PR-ballot rows are used; every VD that
appears is treated as *counted* and fixed at its actual result. Run `python scripts/run_live.py`
to re-project the uncounted VDs. Multiple files are merged; a later file overrides a VD.

## Partial counts per ward / municipality (no VD detail)

If you only have running totals per ward or municipality, together with the % of VDs counted
(as on the IEC dashboard), save them as `progress_*.csv`. See `progress_wards.csv.example`:

| column | meaning |
|---|---|
| `unit_id` | ward id (e.g. `79800012`) or municipality code (`JHB`, `TSH`, `EKU`, `GT421`, `GT422`, `GT423`, `GT481`, `GT484`, `GT485`) |
| `pct_vds_counted` | % of the unit's VDs counted (0–100) |
| `registered_counted` | optional: registered voters in the counted VDs (adds turnout information) |
| party columns | counted valid PR votes: `DA, ANC, ACTIONSA, MK, EFF, OTHER`. Full IEC party names also work, and any other party is added to OTHER |

The model fits the swing to these partial totals. It allows extra variance for wards where few
VDs have reported, since those VDs may not be representative. The final result for a partially
counted ward is its counted votes plus (1 − % counted) × the model's projection for that ward.
Ward rows take precedence over municipal rows. If VD-level files are present, progress files
are ignored.
