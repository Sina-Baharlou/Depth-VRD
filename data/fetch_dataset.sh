#!/usr/bin/env bash
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}(1/7) -- Downloading Visual-Genome dataset... ${NC}"
wget -c "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip"
wget -c "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip"

echo -e "${GREEN}(2/7) -- Extracting Visual-Genome images... ${NC}"
mkdir -p visual_genome
unzip -q -d visual_genome/ images.zip
unzip -q -d visual_genome/ images2.zip

echo -e "${GREEN}(3/7) -- Deleting the extra zip files... ${NC}"
rm images.zip
rm images2.zip

echo -e "${GREEN}(4/7) -- Merging the image folders... ${NC}"
rsync -a visual_genome/VG_100K_2/ visual_genome/VG_100K
mv visual_genome/VG_100K visual_genome/images
rm visual_genome/VG_100K_2 -r

echo -e "${GREEN}(5/7) -- Downloading Visual-Genome images meta-data... ${NC}"
wget -c "http://cvgl.stanford.edu/scene-graph/VG/image_data.json" -P stanford_filtered/

echo -e "${GREEN}(6/7) -- Downloading Visual-Genome scene graph relations... ${NC}"
wget -c "http://cvgl.stanford.edu/scene-graph/dataset/VG-SGG.h5"  -P stanford_filtered/

echo -e "${GREEN}(7/7) -- Downloading Visual-Genome scene graph information... ${NC}"
wget -c "http://cvgl.stanford.edu/scene-graph/dataset/VG-SGG-dicts.json" -P stanford_filtered/

echo -e "${GREEN}Done. ${NC}"