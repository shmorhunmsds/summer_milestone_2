#!/usr/bin/env python3
"""Test the validation script with small dataset"""

import pandas as pd
from nfl_collision_validation import NFLCollisionValidator

print("Testing NFL Collision Validator...")
print("="*60)

validator = NFLCollisionValidator()

# Load data
validator.load_data()

# Process just the injury collisions
print("\n" + "="*60)
injury_collisions = validator.analyze_all_injury_collisions()

print(f"\nResults:")
print(f"Successfully extracted {len(injury_collisions)} injury collisions")

if len(injury_collisions) > 0:
    print(f"\nFirst injury collision features:")
    first = injury_collisions.iloc[0]
    for key, value in first.items():
        if not pd.isna(value):
            print(f"  {key}: {value}")

    print(f"\nSummary of collision features:")
    numeric_cols = injury_collisions.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col not in ['seasonyear', 'gamekey', 'playid', 'gsisid', 'injured_player', 'partner_player', 'is_injury']]

    for col in numeric_cols[:5]:
        print(f"  {col}: mean={injury_collisions[col].mean():.3f}, std={injury_collisions[col].std():.3f}")