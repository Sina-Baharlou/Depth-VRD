#!/usr/bin/env bash
GREEN='\033[1;32m'
NC='\033[0m'

echo -e "${GREEN}(1/6) -- Installing the requirements... ${NC}"
pip install -r requirements.txt

echo -e "${GREEN}(2/6) -- Downloading VG dataset... ${NC}"
(cd data && ./fetch_dataset.sh)

echo -e "${GREEN}(3/6) -- Downloading depth dataset... ${NC}"
(cd data && ./fetch_depth_1024.sh)

echo -e "${GREEN}(4/6) -- Downloading the Checkpoints... ${NC}"
(cd checkpoints && ./fetch_checkpoints.sh)

echo -e "${GREEN}(5/6) -- Compiling libraries... ${NC}"
make

echo -e "${GREEN}(6/6) -- Preparing the environment... ${NC}"
mkdir -p checkpoints/vgdet
mkdir -p data/checkpoints

echo -e "${GREEN}Done. ${NC}"
