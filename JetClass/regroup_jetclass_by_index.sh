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
operation="${1:-link}"

# Validate operation
if [[ ! "$operation" =~ ^(copy|move|link)$ ]]; then
    echo "Error: Invalid operation '$operation'"
    usage
fi

echo "Operation mode: $operation"
echo ""

# Ask user for the base directory
read -p "Enter the path to the JetClass dataset directory (where train_part_x directories are located): " base_dir

# Remove trailing slash if present
base_dir="${base_dir%/}"

# Validate that the directory exists
if [ ! -d "$base_dir" ]; then
    echo "Error: Directory '$base_dir' does not exist!"
    exit 1
fi

# Check if at least one train_part_x directory exists
found_train_part=0
for part_dir in "$base_dir"/train_part_{0..9}; do
    if [ -d "$part_dir" ]; then
        found_train_part=1
        break
    fi
done

if [ $found_train_part -eq 0 ]; then
    echo "Warning: No train_part_x directories found in '$base_dir'"
    read -p "Do you want to continue anyway? (y/n): " continue_anyway
    if [[ ! "$continue_anyway" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

echo "Using base directory: $base_dir"
echo ""

# Define the class prefixes
classes=("HToBB" "HToCC" "HToGG" "HToWW2Q1L" "HToWW4Q" "TTBar" "TTBarLep" "WToQQ" "ZJetsToNuNu" "ZToQQ")

# Create train_xx directories and populate them
for i in {0..49}; do
    # Format the directory name with zero padding
    dir_name=$(printf "train_%02d" $i)
    
    # Create the directory in the base directory
    mkdir -p "$base_dir/$dir_name"
    
    echo "Creating $dir_name..."
    
    # Calculate the two file indices for this directory
    file_idx1=$((i * 2))
    file_idx2=$((i * 2 + 1))
    
    # Process both file indices
    for file_idx in $file_idx1 $file_idx2; do
        # For each class, find and copy/move/link the corresponding file
        for class in "${classes[@]}"; do
            # Format the file number with zero padding
            file_num=$(printf "%03d" $file_idx)
            file_pattern="${class}_${file_num}.root"
            
            # Search for the file in all train_part_x directories
            found=0
            for part_dir in "$base_dir"/train_part_{0..9}; do
                if [ -f "$part_dir/$file_pattern" ]; then
                    source_path="$part_dir/$file_pattern"
                    dest_path="$base_dir/$dir_name/$file_pattern"
                    
                    # Perform the operation based on mode
                    case "$operation" in
                        copy)
                            cp "$source_path" "$dest_path"
                            echo "  Copied $file_pattern from $(basename $part_dir)"
                            ;;
                        move)
                            mv "$source_path" "$dest_path"
                            echo "  Moved $file_pattern from $(basename $part_dir)"
                            ;;
                        link)
                            # Use absolute path for symbolic links
                            abs_source_path=$(readlink -f "$source_path")
                            ln -s "$abs_source_path" "$dest_path"
                            echo "  Linked $file_pattern from $(basename $part_dir)"
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
done

echo ""
echo "Done! Created 50 train_xx directories with 20 files each using '$operation' operation in '$base_dir'."