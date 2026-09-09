# Demo presets

The command-line and demo presets `bear/`, `chair/`, and `dog/`
each contain the four training LDR views from the corresponding local
`HDR-NeRF-syn-4V` scene, copied without modification. Only these selected
images and their matching exposure entries are included.

Each `exposure.json` maps image filenames to positive linear exposure times.
The demo converts these values to log2 exposure times when loading a preset.
