#!/usr/bin/env python3
"""Find overlap between injury plays and available NGS data"""

import pandas as pd
import os

# Load injury data
video_review = pd.read_csv('datasets/NFL-Punt-Analytics-Competition/video_review.csv')
video_review.columns = video_review.columns.str.lower().str.replace('_', '')
injury_games = set(video_review['gamekey'].unique())

print(f"Total injury records: {len(video_review)}")
print(f"Unique injury games: {len(injury_games)}")

# Load NGS gamekeys from all files
ngs_files = [
    'NGS-2016-pre.csv',
    'NGS-2016-post.csv',
    'NGS-2016-reg-wk1-6.csv',
    'NGS-2016-reg-wk7-12.csv',
    'NGS-2016-reg-wk13-17.csv',
    'NGS-2017-pre.csv',
    'NGS-2017-post.csv',
    'NGS-2017-reg-wk1-6.csv',
    'NGS-2017-reg-wk7-12.csv',
    'NGS-2017-reg-wk13-17.csv',
]

all_ngs_games = set()
for file_name in ngs_files:
    file_path = f'datasets/NFL-Punt-Analytics-Competition/{file_name}'
    if os.path.exists(file_path):
        print(f"\nChecking {file_name}...")
        # Read just gamekey column efficiently
        df = pd.read_csv(file_path, usecols=['GameKey'])
        df.columns = df.columns.str.lower().str.replace('_', '')
        file_games = set(df['gamekey'].unique())
        all_ngs_games.update(file_games)
        print(f"  Found {len(file_games)} unique games")

# Find overlap
overlap = injury_games & all_ngs_games
missing = injury_games - all_ngs_games

print("\n" + "="*60)
print(f"Summary:")
print(f"  Injury games in NGS data: {len(overlap)}/{len(injury_games)}")
print(f"  Missing games: {len(missing)}")
print(f"\nGames with NGS data: {sorted(overlap)}")
print(f"\nMissing games: {sorted(missing)}")

# Check which injuries we can actually process
processable_injuries = video_review[video_review['gamekey'].isin(overlap)]
print(f"\n" + "="*60)
print(f"Processable injuries: {len(processable_injuries)}/{len(video_review)}")

# Check how many have valid partners
processable_with_partner = processable_injuries[processable_injuries['primarypartnergsisid'].notna()]
print(f"Processable injuries with partners: {len(processable_with_partner)}")