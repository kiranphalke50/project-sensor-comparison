# changes from Harry's scripts : added wrapper to concatenate CSVs in different data folders

import csv
import os
from collections import defaultdict
from pathlib import Path

# project folder
bluebear = True
base_folder = Path(r"/rds/projects/k/kowalcau-tms-opm-meg/project-sensor_comparison/data") if bluebear else Path(r"T:\project-sensor_comparison\data")

# subject list
subject_list = ["sub-all", "sub-dipole9Hz", "sub-dipole17Hz", "sub-dipole32Hz", "sub-dipole1kHz"]

# input/output paths (input : base/data/subject/daq/runs, output : base/data/subject/daq/runs/concat)
for subject in subject_list:
    subject_folder = base_folder / subject / "daq"

    for run_input_folder in subject_folder.iterdir():
        if not run_input_folder.is_dir():
            continue

        print(f"Processing run folder: {run_input_folder}")

        run_output_folder = run_input_folder / "concat"
        os.makedirs(run_output_folder, exist_ok=True)

        # Group files by base name (excluding the numeric suffix)
        file_groups = defaultdict(list)
        for fname in sorted(os.listdir(run_input_folder)):
            if fname.endswith(".csv"):
                # Extract base name (everything before _00000, _00001 etc.)
                base = fname.rsplit("_", 1)[0]
                file_groups[base].append(fname)

        # Process each group
        for base, files in file_groups.items():
            output_path = run_output_folder / f"{base}.csv"
            with open(output_path, "w", newline="", encoding="utf-8") as fout:
                for i, fname in enumerate(sorted(files)):
                    print(f"Concatenating {fname}...")
                    fpath = run_input_folder / fname
                    with open(fpath, "r", encoding="utf-8") as fin:
                        reader = csv.reader(fin, delimiter=";")
                        if i == 0:
                            # Write header from the first file
                            header = next(reader)
                            writer = csv.writer(fout, delimiter=";")
                            writer.writerow(header)
                        else:
                            next(reader)  # Skip header in subsequent files
                        for row in reader:
                            writer.writerow(row)

        print(f"Finished processing run folder: {run_input_folder}")
