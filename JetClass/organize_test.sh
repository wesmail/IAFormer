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
read -p "Enter the path to the JetClass test directory (where test root files are located): " base_dir

# Remove trailing slash if present
base_dir="${base_dir%/}"

# Validate that the directory exists
if [ ! -d "$base_dir" ]; then
    echo "Error: Directory '$base_dir' does not exist!"
    exit 1
fi

# Check if root files exist in the directory
root_count=$(ls "$base_dir"/*.root 2>/dev/null | wc -l)
if [ $root_count -eq 0 ]; then
    echo "Error: No .root files found in '$base_dir'"
    exit 1
fi

echo "Found $root_count root files in '$base_dir'"
echo "Using base directory: $base_dir"
echo ""

# Define the class prefixes
classes=("HToBB" "HToCC" "HToGG" "HToWW2Q1L" "HToWW4Q" "TTBar" "TTBarLep" "WToQQ" "ZJetsToNuNu" "ZToQQ")

# Create test_xx directories and populate them
for i in {0..9}; do
    # Format the directory name with zero padding
    dir_name=$(printf "test_%02d" $i)
    
    # Create the directory in the base directory
    mkdir -p "$base_dir/$dir_name"
    
    echo "Creating $dir_name..."
    
    # Calculate the two file indices for this directory
    # Files are numbered 100-119, so we map: test_00 -> 100-101, test_01 -> 102-103, etc.
    file_idx1=$((100 + i * 2))
    file_idx2=$((100 + i * 2 + 1))
    
    # Process both file indices
    for file_idx in $file_idx1 $file_idx2; do
        # For each class, find and copy/move/link the corresponding file
        for class in "${classes[@]}"; do
            # Format the file number
            file_pattern="${class}_${file_idx}.root"
            
            # Check if file exists in the base directory
            if [ -f "$base_dir/$file_pattern" ]; then
                source_path="$base_dir/$file_pattern"
                dest_path="$base_dir/$dir_name/$file_pattern"
                
                # Perform the operation based on mode
                case "$operation" in
                    copy)
                        cp "$source_path" "$dest_path"
                        echo "  Copied $file_pattern"
                        ;;
                    move)
                        mv "$source_path" "$dest_path"
                        echo "  Moved $file_pattern"
                        ;;
                    link)
                        # Use absolute path for symbolic links
                        abs_source_path=$(readlink -f "$source_path")
                        ln -s "$abs_source_path" "$dest_path"
                        echo "  Linked $file_pattern"
                        ;;
                esac
            else
                echo "  WARNING: $file_pattern not found"
            fi
        done
    done
done

echo ""
echo "Done! Created 10 test_xx directories with 20 files each using '$operation' operation in '$base_dir'."