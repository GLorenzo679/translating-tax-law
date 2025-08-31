#!/bin/bash
set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Translating Tax Law - Dataset Preparation ===${NC}\n"

# Get the project root directory (parent of scripts/)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Define paths
RAW_DIR="$PROJECT_ROOT/raw"
DATASET_DIR="$PROJECT_ROOT/dataset"
SCRIPTS_DIR="$PROJECT_ROOT/src/dataset_generation"
REPO_URL="https://github.com/CatalaLang/catala-examples"
TEMP_REPO="$RAW_DIR/catala-examples-temp"

# Step 1: Create necessary directories
echo -e "${YELLOW}[1/5] Creating directories...${NC}"
mkdir -p "$RAW_DIR"
mkdir -p "$DATASET_DIR"
echo -e "${GREEN}✓ Directories created${NC}\n"

# Step 2: Download .catala_fr files from GitHub
echo -e "${YELLOW}[2/5] Downloading .catala_fr files from catala-examples repository...${NC}"

# Clone the repository if not already cloned
if [ -d "$TEMP_REPO" ]; then
    echo "Repository already exists, pulling latest changes..."
    cd "$TEMP_REPO"
    git pull
else
    echo "Cloning repository..."
    git clone "$REPO_URL" "$TEMP_REPO"
    cd "$TEMP_REPO"
fi

# Find and copy all .catala_fr files to raw/ directory (excluding tests folders)
echo "Finding all .catala_fr files (excluding tests folders)..."
CATALA_FILES=$(find . -name "*.catala_fr" -type f | grep -v "/tests/")
FILE_COUNT=0

for file in $CATALA_FILES; do
    # Get the relative path and create the same directory structure in raw/
    REL_PATH=$(echo "$file" | sed 's|^\./||')
    DEST_DIR="$RAW_DIR/$(dirname "$REL_PATH")"

    mkdir -p "$DEST_DIR"
    cp "$file" "$DEST_DIR/"
    ((FILE_COUNT++))
    echo "  Copied: $REL_PATH"
done

echo -e "${GREEN}✓ Downloaded $FILE_COUNT .catala_fr files${NC}\n"

# Step 3: Run dataset generation pipeline
cd "$PROJECT_ROOT"

echo -e "${YELLOW}[3/5] Generating dataset from .catala_fr files...${NC}"
python "$SCRIPTS_DIR/generate_dataset.py"
echo -e "${GREEN}✓ Dataset generated${NC}\n"

echo -e "${YELLOW}[4/5] Generating metadata...${NC}"
python "$SCRIPTS_DIR/generate_dataset_metadata.py"
echo -e "${GREEN}✓ Metadata generated${NC}\n"

echo -e "${YELLOW}[4/5] Adding metadata to samples...${NC}"
python "$SCRIPTS_DIR/add_metadata_to_samples.py"
echo -e "${GREEN}✓ Metadata added to samples${NC}\n"

echo -e "${YELLOW}[4/5] Collecting dataset...${NC}"
python "$SCRIPTS_DIR/collect_dataset.py"
echo -e "${GREEN}✓ Dataset collected${NC}\n"

echo -e "${YELLOW}[5/5] Splitting into train/test sets...${NC}"
python "$SCRIPTS_DIR/train_test_split.py"
echo -e "${GREEN}✓ Dataset split into train/test${NC}\n"

# Clean up temporary repository
echo -e "${YELLOW}Cleaning up temporary files...${NC}"
rm -rf "$TEMP_REPO"
echo -e "${GREEN}✓ Temporary files removed${NC}\n"

# Summary
echo -e "${GREEN}=== Dataset Preparation Complete! ===${NC}"
echo -e "Raw files: ${RAW_DIR}"
echo -e "Final dataset: ${DATASET_DIR}"
echo -e "\nDataset files created:"
echo -e "  - ${DATASET_DIR}/dataset.json (full dataset)"
echo -e "  - ${DATASET_DIR}/train.json (training set)"
echo -e "  - ${DATASET_DIR}/test.json (test set)"
echo -e "  - ${DATASET_DIR}/metadata.json (construct definitions)"
