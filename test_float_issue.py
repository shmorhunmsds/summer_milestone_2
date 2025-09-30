#!/usr/bin/env python3
"""Debug the float conversion issue"""

import pandas as pd
import numpy as np

# Load video review
video_review = pd.read_csv('datasets/NFL-Punt-Analytics-Competition/video_review.csv')
video_review.columns = video_review.columns.str.lower().str.replace('_', '')

# Check primarypartnergsisid values
print("Checking primarypartnergsisid values:")
print(f"Type: {video_review['primarypartnergsisid'].dtype}")
print(f"First 10 values: {list(video_review['primarypartnergsisid'].head(10))}")
print(f"Unique values: {video_review['primarypartnergsisid'].unique()}")

# Convert primarypartnergsisid to numeric
print("\nConverting primarypartnergsisid to numeric...")
video_review['primarypartnergsisid'] = pd.to_numeric(video_review['primarypartnergsisid'], errors='coerce')
print(f"After conversion type: {video_review['primarypartnergsisid'].dtype}")
print(f"Non-null count after conversion: {video_review['primarypartnergsisid'].notna().sum()}")

# Get first injury with valid partner
injury = video_review[video_review['primarypartnergsisid'].notna()].iloc[0]
print(f"\nFirst injury with valid partner:")
print(f"  Season: {injury['seasonyear']}")
print(f"  GameKey: {injury['gamekey']}")
print(f"  PlayID: {injury['playid']}")
print(f"  GSISID: {injury['gsisid']} (type: {type(injury['gsisid'])})")
print(f"  Partner: {injury['primarypartnergsisid']} (type: {type(injury['primarypartnergsisid'])})")

# Load the pre-season NGS data (since game 5 is pre-season)
print("\n" + "="*60)
print("Loading NGS 2016 pre-season data...")
ngs = pd.read_csv('datasets/NFL-Punt-Analytics-Competition/NGS-2016-pre.csv')
ngs.columns = ngs.columns.str.lower().str.replace('_', '')

print(f"NGS gsisid type: {ngs['gsisid'].dtype}")

# Try to find the specific play
play_data = ngs[
    (ngs['seasonyear'] == injury['seasonyear']) &
    (ngs['gamekey'] == injury['gamekey']) &
    (ngs['playid'] == injury['playid'])
]

print(f"\nSearching for Season={injury['seasonyear']}, Game={injury['gamekey']}, Play={injury['playid']}")
print(f"Found {len(play_data)} records for this play")

if len(play_data) > 0:
    unique_players = play_data['gsisid'].unique()
    print(f"Players in this play: {sorted(unique_players)}")
    print(f"Looking for injured {injury['gsisid']}: {injury['gsisid'] in unique_players}")
    print(f"Looking for partner {injury['primarypartnergsisid']}: {injury['primarypartnergsisid'] in unique_players}")

    # Get motion data for both players
    injured_motion = play_data[play_data['gsisid'] == injury['gsisid']]
    partner_motion = play_data[play_data['gsisid'] == injury['primarypartnergsisid']]

    print(f"\nMotion data found:")
    print(f"  Injured player {injury['gsisid']}: {len(injured_motion)} records")
    print(f"  Partner player {injury['primarypartnergsisid']}: {len(partner_motion)} records")