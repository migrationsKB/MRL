#!/bin/sh

# 1. Run the crawler.
# give parameters: country_iso2code, batch_nr, start_year, end_year
python -m crawler.main_keywords "DE" "batch4" 2021 2022
# output data to output/crawled/DE/batch4/0/....

