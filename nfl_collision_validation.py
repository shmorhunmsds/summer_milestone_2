#!/usr/bin/env python3
"""
NFL Collision Analysis - Full Dataset Validation
Processes all available player collisions without ratio limits for model validation.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class NFLCollisionValidator:
    def __init__(self):
        self.collision_features = []
        self.injury_collisions = []
        self.motion_data = None
        self.video_review = None

    def load_data(self):
        """Load all necessary datasets"""
        print("Loading datasets...")

        # Load video review (collision details)
        self.video_review = pd.read_csv('datasets/NFL-Punt-Analytics-Competition/video_review.csv')
        # Convert column names to lowercase and replace underscores
        self.video_review.columns = self.video_review.columns.str.lower().str.replace('_', '')

        # Convert primarypartnergsisid to numeric (it comes as string)
        self.video_review['primarypartnergsisid'] = pd.to_numeric(
            self.video_review['primarypartnergsisid'], errors='coerce'
        )

        print(f"Video review records: {len(self.video_review)}")
        print(f"Records with valid partners: {self.video_review['primarypartnergsisid'].notna().sum()}")

        # Load NGS data
        print("Loading NGS player tracking data...")
        ngs_paths = [
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

        # Load all NGS data
        ngs_dfs = []
        for path in ngs_paths:
            if os.path.exists(path):
                print(f"  Loading {path}...")
                df = pd.read_csv(path)
                # Convert column names to lowercase and replace underscores
                df.columns = df.columns.str.lower().str.replace('_', '')
                ngs_dfs.append(df)
            else:
                print(f"  Warning: {path} not found")

        self.motion_data = pd.concat(ngs_dfs, ignore_index=True)
        print(f"  Loaded {len(self.motion_data):,} motion records")
        print(f"  Total players: {self.motion_data['gsisid'].nunique():,}")
        print(f"  Total plays: {self.motion_data.groupby(['seasonyear', 'gamekey', 'playid']).ngroups:,}")

    def calculate_collision_features(self, player1_motion, player2_motion):
        """Calculate collision-specific features from two players' movement data"""

        # Ensure both have time alignment
        # Find overlapping time period
        player1_motion = player1_motion.copy()
        player2_motion = player2_motion.copy()

        # Convert time to datetime and calculate seconds
        player1_motion['time'] = pd.to_datetime(player1_motion['time'])
        player2_motion['time'] = pd.to_datetime(player2_motion['time'])

        player1_motion = player1_motion.sort_values('time')
        player2_motion = player2_motion.sort_values('time')

        player1_motion['seconds'] = (player1_motion['time'] - player1_motion['time'].min()).dt.total_seconds()
        player2_motion['seconds'] = (player2_motion['time'] - player2_motion['time'].min()).dt.total_seconds()

        max_start_time = max(player1_motion['seconds'].min(), player2_motion['seconds'].min())
        min_end_time = min(player1_motion['seconds'].max(), player2_motion['seconds'].max())

        if max_start_time >= min_end_time:
            return None

        # Interpolate positions to common time points
        common_times = np.arange(max_start_time, min_end_time, 0.1)  # 10Hz

        if len(common_times) < 3:
            return None

        def interpolate_player_data(motion_data, times):
            interp_data = pd.DataFrame({'time': times})
            for col in ['x', 'y', 'dis', 'o', 'dir']:
                if col in motion_data.columns:
                    interp_data[col] = np.interp(times, motion_data['seconds'], motion_data[col])
            return interp_data

        p1_interp = interpolate_player_data(player1_motion, common_times)
        p2_interp = interpolate_player_data(player2_motion, common_times)

        # Calculate collision features
        features = {}

        # 1. Distance over time
        distances = np.sqrt((p1_interp['x'] - p2_interp['x'])**2 +
                           (p1_interp['y'] - p2_interp['y'])**2)

        features['min_distance'] = distances.min()
        features['distance_at_start'] = distances.iloc[0] if len(distances) > 0 else np.nan
        features['distance_at_end'] = distances.iloc[-1] if len(distances) > 0 else np.nan
        features['avg_distance'] = distances.mean()

        # Find closest approach
        min_dist_idx = distances.idxmin()
        features['time_to_closest_approach'] = common_times[min_dist_idx] if not pd.isna(min_dist_idx) else np.nan

        # 2. Relative velocities
        p1_vx = np.gradient(p1_interp['x']) / 0.1
        p1_vy = np.gradient(p1_interp['y']) / 0.1
        p2_vx = np.gradient(p2_interp['x']) / 0.1
        p2_vy = np.gradient(p2_interp['y']) / 0.1

        # Relative velocity (how fast they're approaching)
        rel_vx = p1_vx - p2_vx
        rel_vy = p1_vy - p2_vy
        relative_speed = np.sqrt(rel_vx**2 + rel_vy**2)

        features['max_relative_speed'] = np.nanmax(relative_speed)
        features['avg_relative_speed'] = np.nanmean(relative_speed)
        features['relative_speed_at_closest'] = relative_speed[min_dist_idx] if not pd.isna(min_dist_idx) else np.nan

        # 3. Approach angles
        if not pd.isna(min_dist_idx):
            # Vector from player 2 to player 1 at closest approach
            dx = p1_interp['x'].iloc[min_dist_idx] - p2_interp['x'].iloc[min_dist_idx]
            dy = p1_interp['y'].iloc[min_dist_idx] - p2_interp['y'].iloc[min_dist_idx]
            collision_angle = np.degrees(np.arctan2(dy, dx))

            # Player orientations at collision
            p1_orientation = p1_interp['o'].iloc[min_dist_idx]
            p2_orientation = p2_interp['o'].iloc[min_dist_idx]

            features['collision_angle'] = collision_angle
            features['p1_orientation_at_collision'] = p1_orientation
            features['p2_orientation_at_collision'] = p2_orientation

            # Angle differences (head-on vs side collision)
            features['p1_angle_diff'] = abs(p1_orientation - collision_angle)
            features['p2_angle_diff'] = abs(p2_orientation - collision_angle)

            # Speed at collision
            features['p1_speed_at_collision'] = p1_interp['dis'].iloc[min_dist_idx]
            features['p2_speed_at_collision'] = p2_interp['dis'].iloc[min_dist_idx]
        else:
            # Set default values when min_dist_idx is NaN
            features['p1_speed_at_collision'] = np.nan
            features['p2_speed_at_collision'] = np.nan

        # 4. Speed characteristics
        features['p1_max_speed'] = p1_interp['dis'].max()
        features['p2_max_speed'] = p2_interp['dis'].max()
        features['p1_avg_speed'] = p1_interp['dis'].mean()
        features['p2_avg_speed'] = p2_interp['dis'].mean()

        # 5. Acceleration (change in speed)
        p1_acc = np.gradient(p1_interp['dis']) / 0.1
        p2_acc = np.gradient(p2_interp['dis']) / 0.1
        features['p1_max_acc'] = np.nanmax(np.abs(p1_acc))
        features['p2_max_acc'] = np.nanmax(np.abs(p2_acc))

        # 6. Distance change rate (closing speed)
        distance_gradient = np.gradient(distances) / 0.1
        features['max_closing_speed'] = -np.nanmin(distance_gradient)  # Negative means getting closer
        features['avg_closing_speed'] = -np.nanmean(distance_gradient[distance_gradient < 0]) if any(distance_gradient < 0) else 0

        # 7. Collision timing (when in play it occurs)
        features['collision_timing'] = features['time_to_closest_approach'] / common_times[-1] if common_times[-1] > 0 else 0

        # ======== ENGINEERED FEATURES ========
        # 8. Speed ratios and differences
        features['speed_ratio'] = features['p1_max_speed'] / (features['p2_max_speed'] + 1e-6)
        features['speed_difference'] = abs(features['p1_max_speed'] - features['p2_max_speed'])

        # 9. Speed retention at collision (how much speed maintained)
        if not pd.isna(features.get('p1_speed_at_collision')) and features['p1_max_speed'] > 0:
            features['p1_speed_retention'] = features['p1_speed_at_collision'] / (features['p1_max_speed'] + 1e-6)
        else:
            features['p1_speed_retention'] = np.nan

        if not pd.isna(features.get('p2_speed_at_collision')) and features['p2_max_speed'] > 0:
            features['p2_speed_retention'] = features['p2_speed_at_collision'] / (features['p2_max_speed'] + 1e-6)
        else:
            features['p2_speed_retention'] = np.nan

        # 10. Collision intensity score
        # Combines minimum distance and relative speed into a single metric
        # Higher values indicate more intense collisions
        min_dist_norm = 1 / (features['min_distance'] + 0.1)  # Inverse distance (closer = higher)
        speed_norm = features['max_relative_speed']  # We'll normalize this later when we have all data
        features['collision_intensity_raw'] = min_dist_norm * speed_norm

        return features

    def analyze_all_injury_collisions(self):
        """Analyze all injury cases with known collision partners"""

        print("="*60)
        print("ANALYZING ALL INJURY COLLISIONS")
        print("="*60)

        injury_collision_features = []

        for _, injury in self.video_review.iterrows():
            season = injury['seasonyear']
            gamekey = injury['gamekey']
            playid = injury['playid']
            injured_player = injury['gsisid']

            # Skip if no collision partner
            if pd.isna(injury.get('primarypartnergsisid')):
                continue

            partner_player = injury['primarypartnergsisid']
            impact_type = injury.get('primaryimpacttype', 'Unknown')

            # Get motion data for both players
            play_motion = self.motion_data[
                (self.motion_data['seasonyear'] == season) &
                (self.motion_data['gamekey'] == gamekey) &
                (self.motion_data['playid'] == playid)
            ]

            injured_motion = play_motion[play_motion['gsisid'] == injured_player]
            partner_motion = play_motion[play_motion['gsisid'] == partner_player]

            if len(injured_motion) < 3 or len(partner_motion) < 3:
                continue

            # Calculate collision features
            collision_features = self.calculate_collision_features(injured_motion, partner_motion)

            if collision_features is not None:
                # Add metadata
                collision_features['seasonyear'] = season
                collision_features['gamekey'] = gamekey
                collision_features['playid'] = playid
                collision_features['injured_player'] = injured_player
                collision_features['partner_player'] = partner_player
                collision_features['impact_type'] = impact_type
                collision_features['player_activity'] = injury.get('playeractivityderived', 'Unknown')
                collision_features['partner_activity'] = injury.get('primarypartneractivityderived', 'Unknown')
                collision_features['friendly_fire'] = injury.get('friendlyfire', 'Unknown')
                collision_features['is_injury'] = 1

                injury_collision_features.append(collision_features)

        print(f"✅ Successfully analyzed {len(injury_collision_features)} injury collisions")
        return pd.DataFrame(injury_collision_features)

    def process_all_non_injury_collisions(self, batch_size=1000, max_collisions=None):
        """Process ALL possible non-injury collisions without ratio limits"""

        print("="*60)
        print("PROCESSING ALL NON-INJURY COLLISIONS")
        print("="*60)

        # Get injury plays to exclude
        injury_plays = set()
        for _, injury in self.video_review.iterrows():
            injury_plays.add((injury['seasonyear'], injury['gamekey'], injury['playid']))

        # Get all non-injury plays
        all_plays = self.motion_data.groupby(['seasonyear', 'gamekey', 'playid']).size().reset_index(name='count')
        non_injury_plays = all_plays[~all_plays.apply(
            lambda x: (x['seasonyear'], x['gamekey'], x['playid']) in injury_plays, axis=1
        )]

        print(f"Total non-injury plays available: {len(non_injury_plays):,}")

        non_injury_features = []
        plays_processed = 0
        collisions_found = 0

        # Process plays in batches
        for _, play in non_injury_plays.iterrows():
            if max_collisions and collisions_found >= max_collisions:
                print(f"Reached maximum collision limit: {max_collisions:,}")
                break

            plays_processed += 1

            # Progress update
            if plays_processed % 100 == 0:
                print(f"  Processed {plays_processed:,} plays, found {collisions_found:,} valid collisions...")

            # Get all players in this play
            play_data = self.motion_data[
                (self.motion_data['seasonyear'] == play['seasonyear']) &
                (self.motion_data['gamekey'] == play['gamekey']) &
                (self.motion_data['playid'] == play['playid'])
            ]

            play_players = play_data['gsisid'].unique()

            if len(play_players) < 2:
                continue

            # Process all pairwise combinations
            for i in range(len(play_players)):
                for j in range(i+1, len(play_players)):
                    if max_collisions and collisions_found >= max_collisions:
                        break

                    player1 = play_players[i]
                    player2 = play_players[j]

                    player1_data = play_data[play_data['gsisid'] == player1]
                    player2_data = play_data[play_data['gsisid'] == player2]

                    if len(player1_data) < 3 or len(player2_data) < 3:
                        continue

                    collision_features = self.calculate_collision_features(player1_data, player2_data)

                    if collision_features is not None:
                        # Only count as collision if players got close enough (within 5 yards)
                        if collision_features['min_distance'] < 5.0:
                            # Add metadata
                            collision_features['seasonyear'] = play['seasonyear']
                            collision_features['gamekey'] = play['gamekey']
                            collision_features['playid'] = play['playid']
                            collision_features['injured_player'] = player1
                            collision_features['partner_player'] = player2
                            collision_features['impact_type'] = 'No injury'
                            collision_features['is_injury'] = 0

                            non_injury_features.append(collision_features)
                            collisions_found += 1

                            # Save periodically to avoid memory issues
                            if len(non_injury_features) >= batch_size:
                                # Save batch to file
                                batch_df = pd.DataFrame(non_injury_features)
                                filename = f'capstone_file/punt_analytics/non_injury_batch_{collisions_found//batch_size:04d}.csv'
                                os.makedirs('capstone_file/punt_analytics', exist_ok=True)
                                batch_df.to_csv(filename, index=False)
                                print(f"    Saved batch: {filename} ({len(batch_df)} collisions)")
                                non_injury_features = []  # Clear memory

        # Save any remaining collisions
        if non_injury_features:
            batch_df = pd.DataFrame(non_injury_features)
            filename = f'capstone_file/punt_analytics/non_injury_batch_final.csv'
            batch_df.to_csv(filename, index=False)
            print(f"    Saved final batch: {filename} ({len(batch_df)} collisions)")

        print(f"\n✅ Processing complete!")
        print(f"  - Plays processed: {plays_processed:,}")
        print(f"  - Valid collisions found: {collisions_found:,}")

        return collisions_found

    def combine_validation_batches(self):
        """Combine all batch files into final validation dataset"""

        print("\nCombining validation batches...")

        # Load injury collisions
        injury_df = pd.read_csv('capstone_file/punt_analytics/injury_collisions_all.csv')

        # Load all non-injury batch files
        batch_files = [f for f in os.listdir('capstone_file/punt_analytics') if f.startswith('non_injury_batch_')]

        non_injury_dfs = []
        for batch_file in sorted(batch_files):
            df = pd.read_csv(f'capstone_file/punt_analytics/{batch_file}')
            non_injury_dfs.append(df)

        non_injury_df = pd.concat(non_injury_dfs, ignore_index=True)

        print(f"  Injury collisions: {len(injury_df):,}")
        print(f"  Non-injury collisions: {len(non_injury_df):,}")
        print(f"  Total validation samples: {len(injury_df) + len(non_injury_df):,}")

        # Combine datasets
        full_dataset = pd.concat([injury_df, non_injury_df], ignore_index=True)

        # Normalize collision_intensity feature
        if 'collision_intensity_raw' in full_dataset.columns and 'max_relative_speed' in full_dataset.columns:
            print("\nNormalizing collision intensity feature...")
            max_relative_speed_max = full_dataset['max_relative_speed'].max()
            if max_relative_speed_max > 0:
                min_dist_norm = 1 / (full_dataset['min_distance'] + 0.1)
                speed_norm = full_dataset['max_relative_speed'] / max_relative_speed_max
                full_dataset['collision_intensity'] = min_dist_norm * speed_norm
                # Drop the raw version
                full_dataset.drop('collision_intensity_raw', axis=1, inplace=True)
                print("  ✅ Collision intensity feature normalized")

        # Save the complete dataset
        full_dataset.to_csv('capstone_file/punt_analytics/full_collision_dataset.csv', index=False)

        # Create summary statistics
        print("\nDataset Statistics:")
        print(f"  Injury rate: {len(injury_df)/(len(injury_df) + len(non_injury_df))*100:.2f}%")
        print(f"  Class imbalance ratio: 1:{len(non_injury_df)//len(injury_df)}")

        # Feature statistics
        feature_cols = [col for col in full_dataset.columns if col not in
                       ['season_year', 'gamekey', 'playid', 'injured_player', 'partner_player',
                        'impact_type', 'player_activity', 'partner_activity', 'friendly_fire', 'is_injury']]

        print("\nFeature Summary:")
        for feature in feature_cols[:5]:  # Show top 5 features
            injury_mean = injury_df[feature].mean()
            non_injury_mean = non_injury_df[feature].mean()
            ratio = injury_mean / non_injury_mean if non_injury_mean != 0 else np.inf
            print(f"  {feature}: Injury={injury_mean:.3f}, Normal={non_injury_mean:.3f}, Ratio={ratio:.2f}")

        return full_dataset


def run_validation_processing(max_collisions=100000):
    """Main function to run the validation data processing"""

    print(f"NFL Collision Validation Processing")
    print(f"Started: {datetime.now()}")
    print(f"Max collisions to process: {max_collisions:,}" if max_collisions else "Processing ALL collisions")
    print("="*60)

    validator = NFLCollisionValidator()

    try:
        # Load data
        validator.load_data()

        # Process injury collisions
        print("\n" + "="*60)
        injury_collisions = validator.analyze_all_injury_collisions()

        # Save injury collisions
        os.makedirs('capstone_file/punt_analytics', exist_ok=True)
        injury_collisions.to_csv('capstone_file/punt_analytics/injury_collisions_all.csv', index=False)
        print(f"Saved {len(injury_collisions)} injury collisions to capstone_file/punt_analytics/injury_collisions_all.csv")

        # Process ALL non-injury collisions
        print("\n" + "="*60)
        num_collisions = validator.process_all_non_injury_collisions(
            batch_size=1000,
            max_collisions=max_collisions
        )

        # Combine all batches
        print("\n" + "="*60)
        full_dataset = validator.combine_validation_batches()

        print("\n" + "="*60)
        print(f"🎯 VALIDATION DATASET COMPLETE!")
        print(f"Finished: {datetime.now()}")

        return full_dataset

    except Exception as e:
        print(f"❌ Error in validation processing: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run with a limit for testing, or set to None to process everything
    dataset = run_validation_processing(max_collisions=50000)  # Process up to 50,000 collisions

    if dataset is not None:
        print(f"\nValidation dataset ready: capstone_file/punt_analytics/full_collision_dataset.csv")
        print(f"Total samples: {len(dataset):,}")