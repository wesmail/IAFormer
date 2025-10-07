#!/bin/bash

# Usage function
usage() {
    echo "Usage: $0 [copy|move|link]"
    echo "  copy - Copy files to new directories (default)"
    echo "  move - Move files to new directories"
    echo "  link - Create symbolic links to files"
    exit 1
}

# Get operation mode from command line argument
operation="${1:-copy}"

# Validate operation
if [[ ! "$operation" =~ ^(copy|move|link)$ ]]; then
    echo "Error: Invalid operation '$operation'"
    usage
fi

echo "Operation mode: $operation"
echo ""

# Define the class prefixes
classes=("HToBB" "HToCC" "HToGG" "HToWW2Q1L" "HToWW4Q" "TTBar" "TTBarLep" "WToQQ" "ZJetsToNuNu" "ZToQQ")

# Base directory containing train_part_x directories
base_dir="."

# Create train_xx directories and populate them
for i in {0..99}; do
    # Format the directory name with zero padding
    dir_name=$(printf "train_%02d" $i)
    
    # Create the directory
    mkdir -p "$dir_name"
    
    echo "Creating $dir_name..."
    
    # For each class, find and copy/move the corresponding file
    for class in "${classes[@]}"; do
        # Format the file number with zero padding
        file_num=$(printf "%03d" $i)
        file_pattern="${class}_${file_num}.root"
        
        # Search for the file in all train_part_x directories
        found=0
        for part_dir in train_part_{0..9}; do
            if [ -f "$base_dir/$part_dir/$file_pattern" ]; then
                source_path="$base_dir/$part_dir/$file_pattern"
                dest_path="$dir_name/$file_pattern"
                
                # Perform the operation based on mode
                case "$operation" in
                    copy)
                        cp "$source_path" "$dest_path"
                        echo "  Copied $file_pattern from $part_dir"
                        ;;
                    move)
                        mv "$source_path" "$dest_path"
                        echo "  Moved $file_pattern from $part_dir"
                        ;;
                    link)
                        # Use absolute path for symbolic links
                        abs_source_path=$(readlink -f "$source_path")
                        ln -s "$abs_source_path" "$dest_path"
                        echo "  Linked $file_pattern from $part_dir"
                        ;;
                esac
                
                found=1
                break
            fi
        done
        
        if [ $found -eq 0 ]; then
            echo "  WARNING: $file_pattern not found in any train_part_x directory"
        fi
    done
done

echo "Done! Created 100 train_xx directories using '$operation' operation."
