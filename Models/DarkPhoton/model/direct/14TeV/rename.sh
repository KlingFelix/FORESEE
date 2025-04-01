#!/bin/bash

# Loop over all files matching the pattern "QRAD*"
for file in Brem-FWW*; do
    # Check if the file exists (to avoid errors if no files match)
    [ -e "$file" ] || continue
    
    # Create the new filename by replacing "QRAD" with "Brem"
    new_file="${file/Brem-FWW/Brem_FWW}"
    
    # Rename the file
    mv "$file" "$new_file"
    echo "Renamed: $file -> $new_file"
done
