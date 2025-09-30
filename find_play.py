#!/usr/bin/env python3
"""Find which file contains the play"""

import pandas as pd
import os

# Target play
target_season = 2016
target_gamekey = 5
target_playid = 3129

ngs_files = [
    'datasets/NFL-Punt-Analytics-Competition/NGS-2016-pre.csv',
    'datasets/NFL-Punt-Analytics-Competition/NGS-2016-post.csv',
    'datasets/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk1-6.csv',
    'datasets/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk7-12.csv',
    'datasets/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk13-17.csv',
    'datasets/NFL-Punt-Analytics-Competition/NGS-2017-pre.csv',
    'datasets/NFL-Punt-Analytics-Competition/NGS-2017-post.csv',
    'datasets/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk1-6.csv',
    'datasets/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk7-12.csv',
    'datasets/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk13-17.csv',
]

print(f"Looking for Season={target_season}, GameKey={target_gamekey}, PlayID={target_playid}")
print("="*60)

for file_path in ngs_files:
    if os.path.exists(file_path):
        print(f"\nChecking {os.path.basename(file_path)}...")

        # Read just the columns we need
        df = pd.read_csv(file_path, usecols=['Season_Year', 'GameKey', 'PlayID'])
        df.columns = df.columns.str.lower().str.replace('_', '')

        # Check if play exists
        matching = df[
            (df['seasonyear'] == target_season) &
            (df['gamekey'] == target_gamekey) &
            (df['playid'] == target_playid)
        ]

        if len(matching) > 0:
            print(f"  ✅ FOUND! {len(matching)} records for this play")

            # Load full data for this play
            full_df = pd.read_csv(file_path)
            full_df.columns = full_df.columns.str.lower().str.replace('_', '')
            play_data = full_df[
                (full_df['seasonyear'] == target_season) &
                (full_df['gamekey'] == target_gamekey) &
                (full_df['playid'] == target_playid)
            ]
            print(f"  Players in play: {sorted(play_data['gsisid'].unique())}")
            break
        else:
            # Check what gamekeys exist in this file
            unique_games = df[df['seasonyear'] == target_season]['gamekey'].unique()
            print(f"  No match. 2016 gamekeys in file: {sorted(unique_games)[:10]}...")

print("\n" + "="*60)
print("Checking all injury plays...")

# Check all injury plays
video_review = pd.read_csv('datasets/NFL-Punt-Analytics-Competition/video_review.csv')
video_review.columns = video_review.columns.str.lower().str.replace('_', '')

found_count = 0
not_found = []

for _, injury in video_review.iterrows():
    found = False
    for file_path in ngs_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, usecols=['Season_Year', 'GameKey', 'PlayID'], nrows=1000000)
            df.columns = df.columns.str.lower().str.replace('_', '')

            matching = df[
                (df['seasonyear'] == injury['seasonyear']) &
                (df['gamekey'] == injury['gamekey']) &
                (df['playid'] == injury['playid'])
            ]

            if len(matching) > 0:
                found = True
                found_count += 1
                break

    if not found:
        not_found.append({
            'season': injury['seasonyear'],
            'gamekey': injury['gamekey'],
            'playid': injury['playid']
        })

print(f"\nResults:")
print(f"  Found: {found_count}/{len(video_review)} injury plays in NGS data")
print(f"  Missing: {len(not_found)} plays")

if not_found:
    print(f"\nMissing plays (first 5):")
    for play in not_found[:5]:
        print(f"    Season {play['season']}, Game {play['gamekey']}, Play {play['playid']}")