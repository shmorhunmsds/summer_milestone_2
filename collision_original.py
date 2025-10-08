import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

class NFLCollisionAnalyzer:
    def __init__(self):
        self.collision_features = []
        self.injury_collisions = []
        
    def load_data(self):
        """Load all necessary datasets"""
        print("Loading datasets...")
        
        # Load video review (collision details)
        self.video_review = pd.read_csv('datasets/NFL-Punt-Analytics-Competition/video_review.csv')
        self.video_review.columns = self.video_review.columns.str.lower()
        
        # Load NGS motion data - ONLY REGULAR SEASON to avoid conflating different game types
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
        
        def load_and_normalize(path):
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip().str.lower()
            return df
        
        print("Loading NGS motion data...")
        self.motion_data = pd.concat([load_and_normalize(p) for p in ngs_paths], ignore_index=True)
        
        print("NGS columns:", list(self.motion_data.columns))
        print("Video review columns:", list(self.video_review.columns))
        
        # CRITICAL: Convert all key columns to float64 to match motion data format
        print("Converting data types to ensure matching...")
        
        # Motion data - ensure consistent types
        self.motion_data['gamekey'] = self.motion_data['gamekey'].astype('float64')
        self.motion_data['playid'] = self.motion_data['playid'].astype('float64')
        self.motion_data['gsisid'] = self.motion_data['gsisid'].astype('float64')
        self.motion_data['season_year'] = self.motion_data['season_year'].astype('int')
        
        # Video review - convert to match motion data types (handle text values)
        self.video_review['gamekey'] = self.video_review['gamekey'].astype('float64')
        self.video_review['playid'] = self.video_review['playid'].astype('float64')
        self.video_review['gsisid'] = self.video_review['gsisid'].astype('float64')
        
        # Handle partner GSISID - convert to numeric, non-numeric becomes NaN
        self.video_review['primary_partner_gsisid'] = pd.to_numeric(
            self.video_review['primary_partner_gsisid'], errors='coerce'
        )
        
        self.video_review['season_year'] = self.video_review['season_year'].astype('int')
        
        # Clean missing data
        self.motion_data.dropna(subset=['gsisid'], inplace=True)
        
        print(f"Loaded {len(self.motion_data)} motion records")
        print(f"Video review: {len(self.video_review)} injury cases")
        
        # Debug: Check play sizes (more than 22 is likely normal due to substitutions)
        print("\nChecking play sizes...")
        play_sizes = self.motion_data.groupby(['season_year', 'gamekey', 'playid'])['gsisid'].nunique()
        print(f"Play size statistics:")
        print(f"  Min players per play: {play_sizes.min()}")
        print(f"  Max players per play: {play_sizes.max()}")
        print(f"  Average players per play: {play_sizes.mean():.1f}")
        print(f"  Plays with exactly 22 players: {(play_sizes == 22).sum()}/{len(play_sizes)}")
        print("  Note: >22 players is normal due to substitutions and players entering/leaving field")
        
        # Verify data types match
        print("\nData type verification:")
        print(f"Motion data types: gamekey={self.motion_data['gamekey'].dtype}, playid={self.motion_data['playid'].dtype}, gsisid={self.motion_data['gsisid'].dtype}")
        print(f"Video review types: gamekey={self.video_review['gamekey'].dtype}, playid={self.video_review['playid'].dtype}, gsisid={self.video_review['gsisid'].dtype}")
        
        # Test first injury case
        if len(self.video_review) > 0:
            first_injury = self.video_review.iloc[0]
            print(f"\nTesting first injury case:")
            print(f"  Game: {first_injury['gamekey']}, Play: {first_injury['playid']}")
            print(f"  Injured: {first_injury['gsisid']}, Partner: {first_injury['primary_partner_gsisid']}")
            
            # Check if we can find the play
            test_play = self.motion_data[
                (self.motion_data['season_year'] == first_injury['season_year']) &
                (self.motion_data['gamekey'] == first_injury['gamekey']) &
                (self.motion_data['playid'] == first_injury['playid'])
            ]
            print(f"  Found {len(test_play)} motion records for this play")
            
            if len(test_play) > 0:
                players_in_play = test_play['gsisid'].unique()
                injured_found = first_injury['gsisid'] in players_in_play
                partner_found = first_injury['primary_partner_gsisid'] in players_in_play
                print(f"  Injured player found: {injured_found}")
                print(f"  Partner player found: {partner_found}")

    def analyze_collision_case(self, season, gamekey, playid, injured_player, partner_player, verbose=False):
        """Analyze a specific collision between two players"""
        
        if verbose:
            print(f"\nAnalyzing collision: Season {season}, Game {gamekey}, Play {playid}")
            print(f"  Injured player: {injured_player} (type: {type(injured_player)})")
            print(f"  Collision partner: {partner_player} (type: {type(partner_player)})")
        
        # Convert to same types as motion data (float64)
        gamekey = float(gamekey)
        playid = float(playid)
        injured_player = float(injured_player)
        
        # Handle partner_player - might be NaN if it was 'Unclear'
        if pd.isna(partner_player):
            if verbose:
                print("  ❌ Partner player is 'Unclear' - cannot analyze collision")
            return None
            
        partner_player = float(partner_player)
        
        if verbose:
            print(f"  Converted - Injured: {injured_player}, Partner: {partner_player}")
        
        # Get motion data for both players with season filter
        play_motion = self.motion_data[
            (self.motion_data['season_year'] == season) &
            (self.motion_data['gamekey'] == gamekey) &
            (self.motion_data['playid'] == playid) &
            (self.motion_data['gsisid'].isin([injured_player, partner_player]))
        ].copy()
        
        if len(play_motion) == 0:
            if verbose:
                print("  ❌ No motion data found for target players")
            return None
            
        # Sort and prepare data
        play_motion['time'] = pd.to_datetime(play_motion['time'])
        play_motion = play_motion.sort_values(['gsisid', 'time'])
        
        # Calculate relative time
        play_motion['seconds'] = play_motion.groupby('gsisid')['time'].transform(
            lambda x: (x - x.min()).dt.total_seconds()
        )
        
        # Separate the two players
        injured_motion = play_motion[play_motion['gsisid'] == injured_player].copy()
        partner_motion = play_motion[play_motion['gsisid'] == partner_player].copy()
        
        if len(injured_motion) == 0 or len(partner_motion) == 0:
            if verbose:
                print("  ❌ Missing motion data for one or both players")
                print(f"    Injured player {injured_player}: {len(injured_motion)} records")
                print(f"    Partner player {partner_player}: {len(partner_motion)} records")
            return None
        
        if len(injured_motion) < 3 or len(partner_motion) < 3:
            if verbose:
                print("  ❌ insufficient motion data points for gradient calculations")
                print(f"    Injured player {injured_player}: {len(injured_motion)} records")
                print(f"    Partner player {partner_player}: {len(partner_motion)} records")
            return None
            
        if verbose:
            print(f"  ✅ Motion records - Injured: {len(injured_motion)}, Partner: {len(partner_motion)}")
        
        # Calculate collision features
        collision_features = self.calculate_collision_features(injured_motion, partner_motion)
        
        return collision_features
    
    def calculate_collision_features(self, player1_motion, player2_motion):
        """Calculate collision-specific features from two players' movement data"""
        
        # Ensure both have time alignment
        # Find overlapping time period
        max_start_time = max(player1_motion['seconds'].min(), player2_motion['seconds'].min())
        min_end_time = min(player1_motion['seconds'].max(), player2_motion['seconds'].max())
        
        if max_start_time >= min_end_time:
            return None
        
        # Interpolate positions to common time points
        common_times = np.arange(max_start_time, min_end_time, 0.1)  # 10Hz
        
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
        
        # 4. Speed characteristics
        features['p1_max_speed'] = p1_interp['dis'].max()
        features['p2_max_speed'] = p2_interp['dis'].max()
        features['p1_avg_speed'] = p1_interp['dis'].mean()
        features['p2_avg_speed'] = p2_interp['dis'].mean()
        
        if not pd.isna(min_dist_idx):
            features['p1_speed_at_collision'] = p1_interp['dis'].iloc[min_dist_idx]
            features['p2_speed_at_collision'] = p2_interp['dis'].iloc[min_dist_idx]
        
        # 5. Acceleration patterns leading up to collision
        if not pd.isna(min_dist_idx) and min_dist_idx > 5:  # Need some history
            pre_collision_indices = max(0, min_dist_idx - 10), min_dist_idx
            
            p1_pre_speeds = p1_interp['dis'].iloc[pre_collision_indices[0]:pre_collision_indices[1]]
            p2_pre_speeds = p2_interp['dis'].iloc[pre_collision_indices[0]:pre_collision_indices[1]]
            
            if len(p1_pre_speeds) > 1:
                features['p1_acceleration_before_collision'] = np.gradient(p1_pre_speeds).mean()
                features['p2_acceleration_before_collision'] = np.gradient(p2_pre_speeds).mean()
        
        # 6. Time-based features
        features['play_duration'] = len(common_times) * 0.1
        features['collision_timing'] = features['time_to_closest_approach'] / features['play_duration'] if features['play_duration'] > 0 else np.nan
        
        return features
    
    def analyze_all_injury_collisions(self):
        """Analyze all injury cases with known collision partners"""
        
        print("="*60)
        print("ANALYZING ALL INJURY COLLISIONS")
        print("="*60)
        
        injury_collision_features = []
        
        for _, injury in self.video_review.iterrows():
            season = injury['season_year']
            gamekey = injury['gamekey']
            playid = injury['playid']
            injured_player = injury['gsisid']
            
            # Skip if no collision partner
            if pd.isna(injury.get('primary_partner_gsisid')):
                print(f"Skipping injury without collision partner: Game {gamekey}, Play {playid}")
                continue
                
            partner_player = injury['primary_partner_gsisid']
            impact_type = injury.get('primary_impact_type', 'Unknown')
            
            print(f"\nCase: Game {gamekey}, Play {playid}")
            print(f"  Impact: {impact_type}")
            print(f"  Injured: {injured_player} vs Partner: {partner_player}")
            
            # Analyze this collision (WITH verbose output)
            collision_features = self.analyze_collision_case(
                season, gamekey, playid, injured_player, partner_player, verbose=True
            )
            
            if collision_features is not None:
                # Add metadata
                collision_features['season_year'] = season
                collision_features['gamekey'] = gamekey
                collision_features['playid'] = playid
                collision_features['injured_player'] = injured_player
                collision_features['partner_player'] = partner_player
                collision_features['impact_type'] = impact_type
                collision_features['player_activity'] = injury.get('player_activity_derived', 'Unknown')
                collision_features['partner_activity'] = injury.get('primary_partner_activity_derived', 'Unknown')
                collision_features['friendly_fire'] = injury.get('friendly_fire', 'Unknown')
                collision_features['is_injury'] = 1
                
                injury_collision_features.append(collision_features)
                print(f"  ✅ Extracted collision features")
            else:
                print(f"  ❌ Could not extract collision features")
        
        return pd.DataFrame(injury_collision_features)
    
    def sample_non_injury_collisions(self, num_samples=10000):
        """Sample non-injury collisions for comparison - OPTIMIZED VERSION"""
        
        print(f"\nSampling {num_samples} non-injury collisions for comparison...")
        
        # Get random plays that are NOT injury plays
        injury_plays = set()
        for _, injury in self.video_review.iterrows():
            injury_plays.add((injury['season_year'], injury['gamekey'], injury['playid']))
        
        # Sample random plays that are NOT injury plays
        all_plays = self.motion_data[['season_year', 'gamekey', 'playid']].drop_duplicates()
        non_injury_plays = all_plays[~all_plays.apply(
            lambda x: (x['season_year'], x['gamekey'], x['playid']) in injury_plays, axis=1
        )].sample(n=min(num_samples*10, len(all_plays)), random_state=42)  # Increased multiplier
        
        print(f"Available non-injury plays to sample from: {len(non_injury_plays)}")
        
        non_injury_features = []
        
        # Add counters for debugging
        plays_processed = 0
        plays_too_few_players = 0
        plays_insufficient_data = 0
        analyze_failures = 0
        
        for _, play in non_injury_plays.iterrows():
            if len(non_injury_features) >= num_samples:
                break
            
            plays_processed += 1
            
            # Get all players in this play
            play_players = self.motion_data[
                (self.motion_data['season_year'] == play['season_year']) &
                (self.motion_data['gamekey'] == play['gamekey']) &
                (self.motion_data['playid'] == play['playid'])
            ]['gsisid'].unique()
            
            if len(play_players) < 2:
                plays_too_few_players += 1
                continue
            
            # Try multiple player pairs per play
            max_attempts_per_play = 10  # Reduced from 20 for efficiency
            success_this_play = False
            
            for attempt in range(max_attempts_per_play):
                if success_this_play:
                    break
                    
                # Pick two random players to analyze their "collision"
                player1, player2 = np.random.choice(play_players, 2, replace=False)
                
                player1_data = self.motion_data[
                    (self.motion_data['season_year'] == play['season_year']) &
                    (self.motion_data['gamekey'] == play['gamekey']) &
                    (self.motion_data['playid'] == play['playid']) &
                    (self.motion_data['gsisid'] == player1)
                ]    
                
                player2_data = self.motion_data[
                    (self.motion_data['season_year'] == play['season_year']) &
                    (self.motion_data['gamekey'] == play['gamekey']) &
                    (self.motion_data['playid'] == play['playid']) &
                    (self.motion_data['gsisid'] == player2)
                ]

                if len(player1_data) < 3 or len(player2_data) < 3:
                    if attempt == 0:  # Only count once per play
                        plays_insufficient_data += 1
                    continue

                # CRITICAL: Turn OFF verbose printing for bulk sampling
                collision_features = self.analyze_collision_case(
                    play['season_year'], play['gamekey'], play['playid'], player1, player2,
                    verbose=False  # ✅ This is the key fix!
                )

                if collision_features is not None:
                    # Add metadata
                    collision_features['season_year'] = play['season_year']
                    collision_features['gamekey'] = play['gamekey']
                    collision_features['playid'] = play['playid']
                    collision_features['injured_player'] = player1
                    collision_features['partner_player'] = player2
                    collision_features['impact_type'] = 'No injury'
                    collision_features['is_injury'] = 0
                    
                    non_injury_features.append(collision_features)
                    success_this_play = True
                else:
                    if attempt == 0:  # Only count once per play
                        analyze_failures += 1
            
            # Progress update every 5000 plays (less frequent to reduce output)
            if plays_processed % 5000 == 0:
                print(f"Progress: {plays_processed} plays processed, {len(non_injury_features)} successful samples")
        
        print(f"\nSampling Results:")
        print(f"  Plays processed: {plays_processed}")
        print(f"  Too few players: {plays_too_few_players}")
        print(f"  Insufficient data: {plays_insufficient_data}")
        print(f"  Analysis failures: {analyze_failures}")
        print(f"  Successful samples: {len(non_injury_features)}")
        print(f"  Success rate: {len(non_injury_features)/plays_processed:.1%}")
        
        return pd.DataFrame(non_injury_features)
    
    def compare_injury_vs_normal_collisions(self, injury_df, normal_df):
        """Compare collision characteristics between injury and normal plays"""
        
        print("="*60)
        print("INJURY vs NORMAL COLLISION COMPARISON")
        print("="*60)
        
        # Collision features to compare
        collision_metrics = [
            'min_distance', 'max_relative_speed', 'avg_relative_speed',
            'p1_max_speed', 'p2_max_speed', 'p1_speed_at_collision', 'p2_speed_at_collision',
            'collision_timing', 'p1_angle_diff', 'p2_angle_diff'
        ]
        
        print(f"Injury collisions: {len(injury_df)}")
        print(f"Normal collisions: {len(normal_df)}")
        
        print(f"\nCollision Feature Comparison:")
        print(f"{'Feature':<25} {'Injury Avg':<12} {'Normal Avg':<12} {'Ratio':<8} {'Significant?'}")
        print("-" * 70)
        
        significant_features = []
        
        for feature in collision_metrics:
            if feature in injury_df.columns and feature in normal_df.columns:
                injury_mean = injury_df[feature].mean()
                normal_mean = normal_df[feature].mean()
                ratio = injury_mean / normal_mean if normal_mean != 0 else np.inf
                
                # Simple significance test (t-test would be better)
                injury_std = injury_df[feature].std()
                normal_std = normal_df[feature].std()
                difference = abs(injury_mean - normal_mean)
                pooled_std = np.sqrt((injury_std**2 + normal_std**2) / 2)
                
                is_significant = difference > pooled_std  # Simple heuristic
                
                print(f"{feature:<25} {injury_mean:<12.3f} {normal_mean:<12.3f} {ratio:<8.2f} {'Yes' if is_significant else 'No'}")
                
                if is_significant:
                    significant_features.append((feature, ratio))
        
        print(f"\nMost Significant Differences:")
        significant_features.sort(key=lambda x: abs(x[1] - 1), reverse=True)
        for feature, ratio in significant_features[:5]:
            direction = "higher" if ratio > 1 else "lower"
            print(f"  {feature}: {ratio:.2f}x {direction} in injury collisions")
        
        return significant_features


def run_collision_analysis():
    """Main function to run the complete collision analysis"""
    
    analyzer = NFLCollisionAnalyzer()
    
    try:
        # Load data
        analyzer.load_data()
        
        # Analyze injury collisions
        injury_collisions = analyzer.analyze_all_injury_collisions()
        
        if len(injury_collisions) == 0:
            print("❌ No injury collisions found with motion data")
            # Return empty DataFrames instead of None
            return pd.DataFrame(), pd.DataFrame(), []
        
        print(f"\n✅ Successfully analyzed {len(injury_collisions)} injury collisions")
        
        # Sample normal collisions for comparison
        normal_collisions = analyzer.sample_non_injury_collisions(num_samples=len(injury_collisions)*10)
        
        # Compare injury vs normal
        significant_features = analyzer.compare_injury_vs_normal_collisions(injury_collisions, normal_collisions)
        
        # Save results
        injury_collisions.to_csv('scripts/punt_analytics/injury_collision_features.csv', index=False)
        normal_collisions.to_csv('scripts/punt_analytics/normal_collision_features.csv', index=False)
        
        print(f"\n🎯 COLLISION ANALYSIS COMPLETE!")
        print(f"Key insights will show whether:")
        print(f"  - Injury collisions have different approach patterns")
        print(f"  - Relative speeds/angles predict injury risk")
        print(f"  - Collision timing matters for injury occurrence")
        
        return injury_collisions, normal_collisions, significant_features
        
    except Exception as e:
        print(f"❌ Error in collision analysis: {e}")
        import traceback
        traceback.print_exc()
        # Return empty results on error
        return pd.DataFrame(), pd.DataFrame(), []


# Main execution
if __name__ == "__main__":
    # Run the collision analysis
    try:
        injury_df, normal_df, features = run_collision_analysis()
        
        if len(injury_df) > 0:
            print(f"\nResults:")
            print(f"- Injury collisions: {len(injury_df)}")
            print(f"- Normal collisions: {len(normal_df)}")
            print(f"- Significant features: {len(features)}")
        else:
            print("No collision data could be analyzed. Check your data files.")
            
    except Exception as e:
        print(f"Error running analysis: {e}")
        import traceback
        traceback.print_exc()