#!/usr/bin/env python3
"""Check data type mismatch issue"""

import pandas as pd
import numpy as np

# Load video review
video_review = pd.read_csv('datasets/NFL-Punt-Analytics-Competition/video_review.csv')
video_review.columns = video_review.columns.str.lower().str.replace('_', '')

print("Video review data types:")
print(video_review[['seasonyear', 'gamekey', 'playid', 'gsisid', 'primarypartnergsisid']].dtypes)
print()

# Get first injury with partner
injury = video_review[video_review['primarypartnergsisid'].notna()].iloc[0]
print(f"Injury record:")
print(f"  Season: {injury['seasonyear']} (type: {type(injury['seasonyear'])})")
print(f"  GameKey: {injury['gamekey']} (type: {type(injury['gamekey'])})")
print(f"  PlayID: {injury['playid']} (type: {type(injury['playid'])})")
print(f"  GSISID: {injury['gsisid']} (type: {type(injury['gsisid'])})")
print(f"  Partner: {injury['primarypartnergsisid']} (type: {type(injury['primarypartnergsisid'])})")

# Try to convert partner to float
try:
    partner_float = float(injury['primarypartnergsisid'])
    print(f"  Partner as float: {partner_float}")
except:
    print(f"  ERROR: Cannot convert partner '{injury['primarypartnergsisid']}' to float")

# Load the correct NGS file (regular season week 1-6)
print("\n" + "="*60)
print("Loading NGS 2016 regular season week 1-6...")
ngs = pd.read_csv('datasets/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk1-6.csv', nrows=100000)
ngs.columns = ngs.columns.str.lower().str.replace('_', '')

print(f"NGS data types:")
print(ngs[['seasonyear', 'gamekey', 'playid', 'gsisid']].dtypes)

# Check unique games
print(f"\nUnique games in this NGS file: {sorted(ngs['gamekey'].unique())[:20]}")

# Try to find game 5
game5_data = ngs[ngs['gamekey'] == 5]
print(f"\nRecords for game 5: {len(game5_data)}")
if len(game5_data) > 0:
    print(f"Plays in game 5: {sorted(game5_data['playid'].unique())[:10]}")

    # Check for our specific play
    play_data = game5_data[game5_data['playid'] == 3129]
    print(f"\nRecords for play 3129: {len(play_data)}")
    if len(play_data) > 0:
        print(f"Players in this play: {sorted(play_data['gsisid'].unique())}")
        print(f"Looking for injured player {injury['gsisid']}: {injury['gsisid'] in play_data['gsisid'].values}")

        # Try converting partner to float for comparison
        try:
            partner_float = float(injury['primarypartnergsisid'])
            print(f"Looking for partner {partner_float}: {partner_float in play_data['gsisid'].values}")
        except:
            print(f"Partner value '{injury['primarypartnergsisid']}' cannot be converted to float")