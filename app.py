import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from collections import defaultdict
import requests
from datetime import datetime

# API Configuration
API_KEY = "83026149"
BASE_URL = "https://api.leaguerepublic.com/json/"

# Fetch Current Season
def fetch_current_season(api_key):
    url = f"{BASE_URL}getSeasonsForLeague/{api_key}.json"
    response = requests.get(url)
    response.raise_for_status()
    seasons = response.json()
    return next((s for s in seasons if s.get("currentSeason")), None)

# Fetch Fixtures for Season
def fetch_fixtures_for_season(api_key, season_id):
    url = f"{BASE_URL}getFixturesForSeason/{season_id}.json"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

# Extract Results from Fixtures
def extract_results(fixtures):
    results = []
    for match in fixtures:
        home_team = match["homeTeamName"]
        away_team = match["roadTeamName"]
        
        # Exclude teams containing 'esport' (case-insensitive)
        if "esport" in home_team.lower() or "esport" in away_team.lower():
            continue

        if match.get("result"):  # Only include completed matches
            home_player = extract_player_name(home_team)
            away_player = extract_player_name(away_team)
            try:
                home_score = int(match["homeScore"]) if match["homeScore"].isdigit() else None
                away_score = int(match["roadScore"]) if match["roadScore"].isdigit() else None
                if home_score is not None and away_score is not None:  # Ensure scores are valid
                    results.append({
                        "homePlayer": home_player,
                        "awayPlayer": away_player,
                        "homeScore": home_score,
                        "awayScore": away_score,
                        "date": match["fixtureDate"]
                    })
            except ValueError:
                pass
    return results

# Extract player name from team name
def extract_player_name(team_name):
    start = team_name.find("(")
    end = team_name.find(")")
    if start != -1 and end != -1:
        return team_name[start + 1:end].strip()
    return team_name  # Fallback to full name if no brackets found

# Calculate Weights for Matches
def calculate_match_weights(results):
    today = datetime.today()
    weights = []
    for match in results:
        match_date = datetime.strptime(match["date"], "%Y%m%d %H:%M")  # Ensure correct format
        days_diff = (today - match_date).days
        weight = np.exp(-days_diff / 365)  # Exponential decay
        weights.append(weight)
    return weights

# Prepare Data for Classification
def prepare_data_for_classification(results, weights):
    data = []
    for i, match in enumerate(results):
        home_player = match["homePlayer"]
        away_player = match["awayPlayer"]

        # Map outcomes to 0, 1, 2
        if match["homeScore"] > match["awayScore"]:
            outcome = 2  # Home win
        elif match["homeScore"] < match["awayScore"]:
            outcome = 0  # Away win
        else:
            outcome = 1  # Draw

        data.append({"player1": home_player, "player2": away_player, "outcome": outcome, "weight": weights[i]})
    return pd.DataFrame(data)

# Prepare Data for Regression
def prepare_data_for_regression(results, weights):
    data = []
    for i, match in enumerate(results):
        home_player = match["homePlayer"]
        away_player = match["awayPlayer"]
        total_goals = match["homeScore"] + match["awayScore"]
        data.append({"player1": home_player, "player2": away_player, "total_goals": total_goals, "weight": weights[i]})
    return pd.DataFrame(data)

# Train ML Models
def train_ml_models(data_classification, data_regression):
    players = list(set(data_classification["player1"].tolist() + data_classification["player2"].tolist()))
    player_map = {player: i for i, player in enumerate(players)}

    # Encode players for classification and regression
    data_classification["player1_id"] = data_classification["player1"].map(player_map)
    data_classification["player2_id"] = data_classification["player2"].map(player_map)
    data_regression["player1_id"] = data_regression["player1"].map(player_map)
    data_regression["player2_id"] = data_regression["player2"].map(player_map)

    # Classification Model
    X_class = data_classification[["player1_id", "player2_id"]]
    y_class = data_classification["outcome"]
    weights_class = data_classification["weight"]

    X_train_class, X_test_class, y_train_class, y_test_class, train_weights_class, _ = train_test_split(
        X_class, y_class, weights_class, test_size=0.2, random_state=42
    )
    model_class = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model_class.fit(X_train_class, y_train_class, sample_weight=train_weights_class)

    # Regression Model
    X_reg = data_regression[["player1_id", "player2_id"]]
    y_reg = data_regression["total_goals"]
    weights_reg = data_regression["weight"]

    X_train_reg, X_test_reg, y_train_reg, y_test_reg, train_weights_reg, _ = train_test_split(
        X_reg, y_reg, weights_reg, test_size=0.2, random_state=42
    )
    model_reg = XGBRegressor()
    model_reg.fit(X_train_reg, y_train_reg, sample_weight=train_weights_reg)

    return model_class, model_reg, player_map

# Predict Match Outcome
def predict_match_outcome(model, player_map, player1, player2):
    p1_id = player_map.get(player1)
    p2_id = player_map.get(player2)

    if p1_id is None or p2_id is None:
        return None

    prediction = model.predict_proba([[p1_id, p2_id]])[0]
    return prediction

# Predict Total Goals
def predict_total_goals(model, player_map, player1, player2):
    p1_id = player_map.get(player1)
    p2_id = player_map.get(player2)

    if p1_id is None or p2_id is None:
        return None

    prediction = model.predict([[p1_id, p2_id]])[0]
    return prediction

# Streamlit App
def main():
    st.title("FIFA League Match Predictor")
    st.sidebar.title("Select Players for Match Prediction")

    # Fetch Current Season and Fixtures
    st.sidebar.write("Fetching current season and matches...")
    current_season = fetch_current_season(API_KEY)
    if not current_season:
        st.error("Could not fetch current season.")
        return

    season_id = current_season["seasonID"]
    fixtures = fetch_fixtures_for_season(API_KEY, season_id)
    if not fixtures:
        st.error("No fixtures found for the current season.")
        return

    # Extract Results and Calculate Weights
    results = extract_results(fixtures)
    weights = calculate_match_weights(results)
    if not results:
        st.error("No match results available.")
        return

    # Prepare Data and Train Models
    data_classification = prepare_data_for_classification(results, weights)
    data_regression = prepare_data_for_regression(results, weights)
    st.sidebar.write("Training ML models...")
    model_class, model_reg, player_map = train_ml_models(data_classification, data_regression)

    # Player Selection
    players = list(player_map.keys())
    player1 = st.sidebar.selectbox("Select Player 1", players)
    player2 = st.sidebar.selectbox("Select Player 2", players)

    if st.sidebar.button("Predict Outcome"):
        # Predictions
        outcome_prediction = predict_match_outcome(model_class, player_map, player1, player2)
        total_goals_prediction = predict_total_goals(model_reg, player_map, player1, player2)

        st.write("### Predictions")
        if outcome_prediction is not None:
            st.write(f"Win Probability for {player1}: {outcome_prediction[2]*100:.2f}%")
            st.write(f"Draw Probability: {outcome_prediction[1]*100:.2f}%")
            st.write(f"Win Probability for {player2}: {outcome_prediction[0]*100:.2f}%")

        if total_goals_prediction is not None:
            st.write(f"Predicted Total Goals: {total_goals_prediction:.2f}")

        # Reasoning
        st.write("### Reasoning Behind Predictions")
        player1_stats = calculate_player_statistics(results, player1)
        player2_stats = calculate_player_statistics(results, player2)

        st.write(f"**{player1}'s Stats:** {player1_stats}")
        st.write(f"**{player2}'s Stats:** {player2_stats}")

if __name__ == "__main__":
    main()
