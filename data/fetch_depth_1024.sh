#!/usr/bin/env bash
GREEN='\033[0;32m'
NC='\033[0m'

# -- Depth dataset parameters --
TARGET="vg_depth_1024"
GDRIVE_ID="1-BQcGwsnwS-cYHSWIAkmToSCINUnNZQh"
OUTPUT="depth_images_1024"

echo -e "${GREEN}(1/4) -- Downloading VG depth dataset (Target: ${TARGET})... ${NC}"
gdown --id ${GDRIVE_ID} --output ${TARGET}.zip

echo -e "${GREEN}(2/4) -- Extracting the dataset... ${NC}"
unzip -q ${TARGET}.zip
rm -r ${TARGET}.zip

echo -e "${GREEN}(3/4) -- Merging the image folders... ${NC}"
rsync -a ${TARGET}/VG_100K_2/ ${TARGET}/VG_100K
rm ${TARGET}/VG_100K_2 -r

echo -e "${GREEN}(3/4) -- Moving to the specified directory... ${NC}"
mkdir -p visual_genome
mv ${TARGET}/VG_100K visual_genome/${OUTPUT}

echo -e "${GREEN}Done. ${NC}"
