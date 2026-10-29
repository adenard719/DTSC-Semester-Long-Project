import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier


print("Loading datasets...")
df_teams = pd.read_csv('yearly_team_stats_offense.csv')
df_sb = pd.read_csv('superbowl.csv')

# DATA CLEANING
df_teams = df_teams.dropna(subset=['season', 'team', 'wins', 'total_off_points', 'total_def_points']).copy()
df_teams['point_differential'] = df_teams['total_off_points'] - df_teams['total_def_points']

# Re format team names
name_to_abbr = {
    'Kansas City Chiefs': 'KC', 'New England Patriots': 'NE', 'Philadelphia Eagles': 'PHI',
    'Denver Broncos': 'DEN', 'Seattle Seahawks': 'SEA', 'Baltimore Ravens': 'BAL',
    'New York Giants': 'NYG', 'Green Bay Packers': 'GB', 'New Orleans Saints': 'NO',
    'Pittsburgh Steelers': 'PIT', 'Indianapolis Colts': 'IND', 'Tampa Bay Buccaneers': 'TB',
    'Los Angeles Rams': 'LA', 'St. Louis Rams': 'LA', 'San Francisco 49ers': 'SF',
    'Atlanta Falcons': 'ATL', 'Carolina Panthers': 'CAR', 'Arizona Cardinals': 'ARI',
    'Cincinnati Bengals': 'CIN', 'Buffalo Bills': 'BUF', 'Miami Dolphins': 'MIA',
    'Los Angeles Chargers': 'LAC', 'San Diego Chargers': 'LAC', 'Las Vegas Raiders': 'LV',
    'Oakland Raiders': 'LV', 'Dallas Cowboys': 'DAL', 'Washington Redskins': 'WAS',
    'Washington Commanders': 'WAS', 'Chicago Bears': 'CHI', 'Minnesota Vikings': 'MIN',
    'New York Jets': 'NYJ', 'Cleveland Browns': 'CLE', 'Detroit Lions': 'DET', 'Jacksonville Jaguars': 'JAX'
}

df_sb['season'] = pd.to_datetime(df_sb['Date'], format='mixed', errors='coerce').dt.year - 1
df_sb['winner_abbr'] = df_sb['Winner'].map(name_to_abbr)

df_sb_winners = df_sb[['season', 'winner_abbr', 'Winner']].copy()
df_sb_winners.columns = ['season', 'winner_abbr', 'actual_winner_name']
df_sb_winners['won_superbowl'] = 1

# Merge to label historical winners for 2012-2019 training data
df_teams = pd.merge(df_teams, df_sb_winners[['season', 'winner_abbr', 'won_superbowl']], left_on=['season', 'team'], right_on=['season', 'winner_abbr'], how='left')
df_teams['won_superbowl'] = df_teams['won_superbowl'].fillna(0).astype(int)

# Prediction Model Setup
# Train ONLY on historical seasons (2012-2019)
train_df = df_teams[df_teams['season'] <= 2019]

# Predict Super Bowl winners of 2020-present using regular season stats only
pred_df = df_teams[(df_teams['season'] >= 2020) & (df_teams['season_type'] == 'REG')].copy()

features = ['wins', 'win_pct', 'point_differential', 'total_off_points', 'total_def_points', 'passing_yards', 'rushing_yards']
X_train = train_df[features].fillna(0)
y_train = train_df['won_superbowl']

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Forecast on 2020-present data
X_pred = pred_df[features].fillna(0)
pred_df['superbowl_probability'] = model.predict_proba(X_pred)[:, 1]

# Extract top predicted winner for each recent season
future_predictions = pred_df.loc[pred_df.groupby('season')['superbowl_probability'].idxmax()].copy()

print("\n--- TRUE PREDICTION MODEL: SUPER BOWL WINNERS (2020–PRESENT) ---")
print(future_predictions[['season', 'team', 'superbowl_probability', 'wins', 'point_differential']].to_string(index=False))

# Graph Visualization
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.boxplot(
    x='won_superbowl',
    y='point_differential',
    data=df_teams,
    hue='won_superbowl',
    palette='Set2',
    legend=False
)

plt.title('Point Differential Distribution: Super Bowl Winners vs. Other Teams', fontsize=14, fontweight='bold')
plt.xlabel('Won Super Bowl (0 = No, 1 = Yes)', fontsize=12)
plt.ylabel('Point Differential (Offensive - Defensive Points)', fontsize=12)
plt.xticks([0, 1], ['Other Teams', 'Super Bowl Winners'])
plt.tight_layout()

plt.savefig('superbowl_point_differential.png', dpi=300)
plt.show()