#!/bin/sh

# run post processor to merge the results from semantic analyses
python -m postprocessor.merging_results

# get the aggregated results for sentiments in jsonl file
python postprocessor/get_stats_results.py sentiment

# get the aggregated results for hate speeches in jsonl file
python postprocessor/get_stats_results.py hsd


