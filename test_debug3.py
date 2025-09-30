#!/usr/bin/env python3
"""Find the mapping between game numbers and gamekeys"""

import pandas as pd
import numpy as np

# Load video review
video_review = pd.read_csv('datasets/NFL-Punt-Analytics-Competition/video_review.csv')
video_review.columns = video_review.columns.str.lower().str.replace('_', '')

print("Injury gamekeys from video_review:")
print(sorted(video_review['gamekey'].unique()))

# Check game_data.csv for the mapping
print("\n" + "="*60)
print("Checking game_data.csv for mapping...")
game_data = pd.read_csv('datasets/NFL-Punt-Analytics-Competition/game_data.csv')
print(f"Game data columns: {list(game_data.columns)}")
game_data.columns = game_data.columns.str.lower().str.replace('_', '')

# Show some game data
print(f"\nFirst few games:")
print(game_data[['seasonyear', 'seasontype', 'week', 'gamekey', 'gamedate']].head(10))

# Find game with gamekey 5
game5 = game_data[game_data['gamekey'] == 5]
if len(game5) > 0:
    print(f"\nGame with gamekey=5:")
    print(game5)

# Check all 2016 games
print("\n" + "="*60)
print("All 2016 games:")
games_2016 = game_data[game_data['seasonyear'] == 2016]
print(f"Total 2016 games: {len(games_2016)}")
print(games_2016[['seasontype', 'week', 'gamekey']].head(20))

# Check NGS data to see what gamekeys are actually there
print("\n" + "="*60)
print("Checking NGS data gamekeys...")
ngs_sample = pd.read_csv('datasets/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk1-6.csv', nrows=100000)
ngs_sample.columns = ngs_sample.columns.str.lower().str.replace('_', '')

ngs_gamekeys = sorted(ngs_sample['gamekey'].unique())
print(f"Gamekeys in NGS 2016 reg wk1-6: {ngs_gamekeys[:20]}")

# Cross-reference
print("\n" + "="*60)
print("Cross-referencing injury plays with available NGS data...")
for _, injury in video_review.head(5).iterrows():
    gamekey = injury['gamekey']
    game_info = game_data[game_data['gamekey'] == gamekey]
    if len(game_info) > 0:
        game_info = game_info.iloc[0]
        print(f"Injury: GameKey={gamekey}, Season={injury['seasonyear']}, PlayID={injury['playid']}")
        print(f"  -> Game info: {game_info['seasontype']}, Week {game_info['week']}, Date: {game_info['gamedate']}")
    else:
        print(f"Injury: GameKey={gamekey} - NO GAME DATA FOUND")