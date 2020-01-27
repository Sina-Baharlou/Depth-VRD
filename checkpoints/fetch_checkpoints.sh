#!/usr/bin/env bash
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}Downloading Faster-RCNN Checkpoint... ${NC}"
gdown --id 11zKRr2OF5oclFL47kjFYBOxScotQzArX --output vg-faster-rcnn.tar

echo -e "${GREEN}Downloading LCVD Checkpoint... ${NC}"
gdown --id 1ZkHseT3zA2bk1CBxJ6fhI18GuqwUIAwM --output vgrel-lcvd.tar

echo -e "${GREEN}Done. ${NC}"