#!/usr/bin/env python3
"""Quick debug script to check data loading"""

import pandas as pd
import numpy as np

# Load video review
print("Loading video review...")
video_review = pd.read_csv('datasets/NFL-Punt-Analytics-Competition/video_review.csv')
print(f"Original columns: {list(video_review.columns)}")

# Convert columns
video_review.columns = video_review.columns.str.lower().str.replace('_', '')
print(f"Converted columns: {list(video_review.columns)}")

print(f"\nTotal injury records: {len(video_review)}")
print(f"First few rows:")
print(video_review.head())

# Check for primary partner
print(f"\nChecking primarypartnergsisid column...")
print(f"Non-null partners: {video_review['primarypartnergsisid'].notna().sum()}")
print(f"Null partners: {video_review['primarypartnergsisid'].isna().sum()}")

# Check data types
print(f"\nData types:")
print(video_review.dtypes)

# Sample one injury record
print("\nFirst injury with partner:")
injury_with_partner = video_review[video_review['primarypartnergsisid'].notna()].iloc[0]
print(f"Season: {injury_with_partner['seasonyear']}")
print(f"GameKey: {injury_with_partner['gamekey']}")
print(f"PlayID: {injury_with_partner['playid']}")
print(f"GSISID: {injury_with_partner['gsisid']}")
print(f"Partner GSISID: {injury_with_partner['primarypartnergsisid']}")

# Now check NGS data
print("\n" + "="*60)
print("Loading NGS data sample...")
ngs_sample = pd.read_csv('datasets/NFL-Punt-Analytics-Competition/NGS-2016-pre.csv', nrows=10000)
print(f"Original NGS columns: {list(ngs_sample.columns)}")

ngs_sample.columns = ngs_sample.columns.str.lower().str.replace('_', '')
print(f"Converted NGS columns: {list(ngs_sample.columns)}")

print(f"\nNGS data types:")
print(ngs_sample.dtypes)

# Check if we can find the injury play in NGS
print("\n" + "="*60)
print("Looking for injury play in NGS data...")
test_season = injury_with_partner['seasonyear']
test_gamekey = injury_with_partner['gamekey']
test_playid = injury_with_partner['playid']
test_gsisid = injury_with_partner['gsisid']

print(f"Searching for: Season={test_season}, Game={test_gamekey}, Play={test_playid}, Player={test_gsisid}")

# Load more NGS data to find this play
print("\nLoading full NGS 2016 pre-season data...")
ngs_full = pd.read_csv('datasets/NFL-Punt-Analytics-Competition/NGS-2016-pre.csv')
ngs_full.columns = ngs_full.columns.str.lower().str.replace('_', '')

# Try to find the play
matching_plays = ngs_full[
    (ngs_full['seasonyear'] == test_season) &
    (ngs_full['gamekey'] == test_gamekey) &
    (ngs_full['playid'] == test_playid)
]

print(f"Found {len(matching_plays)} records for this play")
if len(matching_plays) > 0:
    print(f"Players in this play: {matching_plays['gsisid'].unique()}")
    print(f"Looking for injured player {test_gsisid}: {'FOUND' if test_gsisid in matching_plays['gsisid'].values else 'NOT FOUND'}")
    print(f"Looking for partner {injury_with_partner['primarypartnergsisid']}: {'FOUND' if injury_with_partner['primarypartnergsisid'] in matching_plays['gsisid'].values else 'NOT FOUND'}")