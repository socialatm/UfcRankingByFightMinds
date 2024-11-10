# %%
# Code by Eliott B. built on the previous work of Nicolas (datalgo)
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import timedelta


from numpy import nan
from math import exp

import os
import matplotlib.pyplot as plt

folder_name = "ranking"

# Create the folder for saving rankings if it doesn't exist
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# %%#1
# Load the dataset containing UFC fight data
df_fights = pd.read_csv("extraction/ufc_fights.csv", sep=",")

def handle_draw_column(value):
    if isinstance(value, str):
        if value.lower() == 'draw':
            return 'draw'
        elif value.lower() == 'nc':
            return 'nc'
        elif value == '0':
            return 0
    
    if isinstance(value, (int, float)) and value == 0:
        return 0
    
    return value

# Applying the function to the draw column of your dataframe
df_fights['draw'] = df_fights['draw'].apply(handle_draw_column)

# Get the list of fighters that missed weight with the date
df_missed_weights = pd.read_csv('extraction/missed_weight_fighters.csv')
df_missed_weights["Fighter Name"] = df_missed_weights["Fighter Name"].str.strip()
df_missed_weights["Date"] = pd.to_datetime(df_missed_weights["Date"])


# Convert event dates to datetime format and sort the fights by date
df_fights["eventDate"] = pd.to_datetime(df_fights["eventDate"])
df_fights.sort_values(by=["eventDate"], inplace=True)

# Fill any missing values with 0
df_fights.fillna(0, inplace=True)

# Dictionary to track the last non-catch weight class each fighter fought in
last_non_catch_weight = {}

# Store the last score in each weight class for each fighter
fighter_last_weight_class_score = {}

# Dictionary to keep track of fighters and the weight class they won a title in
fighter_title_weight_class = {}

# Loop through the dataset to correct weight classes for "Catch Weight" fights
for index, row in df_fights.iterrows():
    fighters = [(row['winnerHref'], 'winner'), (row['loserHref'], 'loser')]
    for fighter, role in fighters:
        if row['weightClass'] != "Catch Weight":
            last_non_catch_weight[fighter] = row['weightClass']
        else:
            if fighter in last_non_catch_weight:
                df_fights.at[index, 'weightClass'] = last_non_catch_weight[fighter]
            else:
                other_fighter, other_role = (fighters[0] if fighters[1][0] == fighter else fighters[1])
                if other_fighter in last_non_catch_weight:
                    df_fights.at[index, 'weightClass'] = last_non_catch_weight[other_fighter]
                else:
                    df_fights.at[index, 'weightClass'] = "Unknown"

# %%
# Function to calculate the peremption coefficient based on time since the fight
def fight_peremption_coeff(event_date, today):
    decay_rate = 4200  # Decay rate to control the smoothness of the coefficient
    midpoint = 400  # Midpoint in days to control when the coefficient decays
    t = (today - event_date).days
    if t>730:
        return (1 + exp(-midpoint/decay_rate)) / (1 + exp((t - 730 - midpoint)/decay_rate))
    else:
        return 1

# Scores associated with different methods of winning a fight
win_type_scores = {
    "KO/TKO": 1.2,
    "submission": 1.2,
    "unanimousDecision": 1.15,
    "majorityDecision": 1.1,
    "splitDecision": 1.1,
    "DQ": 1,
}

# Map the method of victory to the corresponding score
df_fights["win_type_scores"] = df_fights["method"].map(win_type_scores)

# Apply additional scores based on performance and type of fight
df_fights["performance_of_the_night_score"] = df_fights["performanceOfTheNight"].apply(lambda x: 1.3 if x == 1 else 1).astype(float)
df_fights["fight_of_the_night_score"] = df_fights["fightOfTheNight"].apply(lambda x: 1.3 if x == 1 else 1)
df_fights["belt_score"] = df_fights["belt"].apply(lambda x: 1.5 if x == 1 else 1)

#df_fights['rounds'] = df_fights['rounds'].astype(int)
df_fights["rounds_score"] = df_fights["rounds"].apply(lambda x: 1 + x*0.05 if x != 0 else 1)

opponent_bonus_coeff = 0.5  # Coefficient for bonus points based on the opponent's strength

# %%
# Calculate the peremption coefficient for each fight
df_fights['peremption_coeff'] = df_fights['eventDate'].apply(
        lambda event_date: fight_peremption_coeff(event_date, pd.Timestamp.today())
    )

# Initialize lists to track win and loss streaks for each fighter
fighters = df_fights['winnerHref'].unique().tolist() + df_fights['loserHref'].unique().tolist()
fighters = list(set(fighters))
fighter_win_streaks = {fighter: 0 for fighter in fighters}
fighter_loser_streaks = {fighter: 0 for fighter in fighters}

weight_class_rankings = {}

last_fight_weight_class = {}

# Main loop over each fight to calculate scores
for i, fight in tqdm(df_fights.iterrows(), desc="Iterate over fights", total=len(df_fights)):
    # Select previous fights before the current one
    previous_fights = df_fights[df_fights["eventDate"] < fight["eventDate"]].copy()

    winner_name = fight['winnerName']
    loser_name = fight['loserName']
    fight_date = fight['eventDate']
    weight_class = fight['weightClass']

    # Check if the winner missed weight
    missed_weight_winner = df_missed_weights[
        (df_missed_weights['Date'] == fight_date) &
        (df_missed_weights['Fighter Name'].apply(lambda x: winner_name in x or x in winner_name))
    ]
    
    # Check if the loser missed weight
    missed_weight_loser = df_missed_weights[
        (df_missed_weights['Date'] == fight_date) &
        (df_missed_weights['Fighter Name'].apply(lambda x: loser_name in x or x in loser_name))
    ]

    winner_malus = 1
    loser_malus = 1
    if not missed_weight_winner.empty:
        winner_malus = 0.8

    if not missed_weight_loser.empty:
        loser_malus = 1.2

    df_fights.loc[i, "winner_malus"] = winner_malus
    df_fights.loc[i, "loser_malus"] = loser_malus


    # If either fighter missed weight, check their last weight class
    if not missed_weight_winner.empty or not missed_weight_loser.empty:
        
        # Retrieve the previous weight classes
        winner_last_weight_class = last_fight_weight_class.get(winner_name, weight_class)
        loser_last_weight_class = last_fight_weight_class.get(loser_name, weight_class)
        
        # Check if the current weight class differs from the previous weight class of either fighter
        if winner_last_weight_class != weight_class or loser_last_weight_class != weight_class:
            # Update the current fight's weight class to the last known "usual" weight class
            weight_class = winner_last_weight_class if winner_last_weight_class == loser_last_weight_class else weight_class
            df_fights.at[i, 'weightClass'] = weight_class

    # Calculate the average score for the weight class of the current fight
    weight_class_fights = previous_fights[(previous_fights["weightClass"] == fight["weightClass"])].copy()
    if not weight_class_fights.empty:
        weight_class_fights['score'] = weight_class_fights['winner_points_before_fight'] + weight_class_fights['loser_points_before_fight']
        weight_class_average = weight_class_fights['score'].median()
    else:
        weight_class_average = 0.1

    winner = fight['winnerHref']
    loser = fight['loserHref']

    # Determine the most frequent weight class for the winner based on the last 5 fights
    winner_last_fights = previous_fights[(previous_fights["winnerHref"] == winner) | (previous_fights["loserHref"] == winner)].tail(5)

    # Determine the most frequent weight class for the loser based on the last 5 fights
    loser_last_fights = previous_fights[(previous_fights["loserHref"] == loser) | (previous_fights["winnerHref"] == loser)].tail(5)

    # Handle the winner's score
    if weight_class in fighter_last_weight_class_score.get(winner, {}):
        # Use the stored score and apply its peremption coefficient
        last_score, last_peremption_coeff = fighter_last_weight_class_score[winner][weight_class]
        winner_before_score = last_score * last_peremption_coeff
    else:
        # If no score in any weight class, start with the default
        winner_before_score = 0.01

    # Store the winner's score before the fight
    df_fights.loc[i, "winner_points_before_fight"] = winner_before_score
    
    # Handle the loser's score
    if weight_class in fighter_last_weight_class_score.get(loser, {}):
        # Use the stored score and apply its peremption coefficient
        last_score, last_peremption_coeff = fighter_last_weight_class_score[loser][weight_class]
        loser_before_score = last_score * last_peremption_coeff
    else:
        # If no score in any weight class, start with the default
        loser_before_score = 0.01

    # Store the loser's score before the fight
    df_fights.loc[i, "loser_points_before_fight"] = loser_before_score
    
    # Ensure the winner is initialized
    if winner not in fighter_last_weight_class_score:
        fighter_last_weight_class_score[winner] = {}

    # Ensure the loser is initialized
    if loser not in fighter_last_weight_class_score:
        fighter_last_weight_class_score[loser] = {}

    # Add a bonus if the fighter has been a champion
    if winner in fighter_title_weight_class and weight_class in fighter_title_weight_class[winner]:
        champ_bonus = 1.55
    else:
        champ_bonus = 1

    df_fights.loc[i, "champ_bonus"] = champ_bonus

    winner_last_two_fights = winner_last_fights.tail(1) 

    loser_last_two_fights = loser_last_fights.tail(1)

    winner_last_two_fights = winner_last_fights[winner_last_fights["eventDate"] >= fight["eventDate"] - timedelta(days=548)].tail(1)

    loser_last_two_fights = loser_last_fights[loser_last_fights["eventDate"] >= fight["eventDate"] - timedelta(days=548)].tail(1)

    

    if df_fights.loc[i,'draw']==0:

        # Update fighter_title_weight_class if this is a title fight
        if fight["belt"] == 1 and fight['rounds']==5: 
            fighter_title_weight_class[winner] = weight_class

        percentage_of_average_win = 0.5
        value_for_win = weight_class_average * percentage_of_average_win
        df_fights.loc[i, "value_for_win"] = value_for_win

        # Handle the winning streak bonus
        if winner in fighter_win_streaks:
            fighter_win_streaks[winner] += 1
        else:
            fighter_win_streaks[winner] = 1
        win_streak_boost = 1 + (0.005 * fighter_win_streaks[winner])
        
        # Handle the losing streak malus
        if loser in fighter_loser_streaks:
            fighter_loser_streaks[loser] += 1
        else:
            fighter_loser_streaks[loser] = 1
        lose_streak_boost = 1 + (0.005 * fighter_loser_streaks[loser])

        opponent_bonus = loser_before_score * opponent_bonus_coeff 
        df_fights.loc[i, "opponent_bonus"] = opponent_bonus

        percentage_of_average_loss = 0.2
        value_for_loss = weight_class_average * percentage_of_average_loss
        df_fights.loc[i, "value_for_loss"] = value_for_loss * fight["win_type_scores"]

        fighter_win_streaks[loser] = 0
        fighter_loser_streaks[winner] = 0

        # Calculate final scores for the winner and loser after the fight
        winner_score = max(0,((fight["win_type_scores"] * opponent_bonus + value_for_win) * fight["belt_score"] * fight["performance_of_the_night_score"] * fight["fight_of_the_night_score"] * fight['rounds_score']) * win_streak_boost * champ_bonus * winner_malus) 
        df_fights.loc[i, "winner_score"] = winner_score

        winner_points_after_fight = winner_before_score + winner_score

        loser_score = ((value_for_loss*lose_streak_boost*loser_malus)/fight["fight_of_the_night_score"])/fight["belt_score"]/fight['rounds_score']/champ_bonus

        loser_points_after_fight = loser_before_score - loser_score

        if winner_points_after_fight < loser_points_after_fight:
            winner_points_after_fight = loser_points_after_fight + 0.01


        # Get or initialize the ranking for this weight class
        if weight_class not in weight_class_rankings:
            weight_class_rankings[weight_class] = {}

        # Update the winner and loser scores in the weight class ranking dictionary
        weight_class_rankings[weight_class][winner] = winner_points_after_fight
        weight_class_rankings[weight_class][loser] = loser_points_after_fight

        # Sort the fighters by their points in descending order to get the top 15
        sorted_fighters = sorted(weight_class_rankings[weight_class].items(), key=lambda x: x[1], reverse=True)
        top_15_fighters = sorted_fighters[:15]

        if any(fighter == winner for fighter, _ in top_15_fighters):

            # Current winner's points after the current fight
            current_winner_points = winner_points_after_fight

            # Check the last two fights of the winner to check if their points will exceed the points of an opponent they lost against
            for _, past_fight in winner_last_two_fights.iterrows():

                # Check if the winner lost in this past fight and if the loss was by KO, SUB, or U-DEC
                if past_fight["loserHref"] == winner and past_fight["method"] in ["KO/TKO", "submission", "unanimousDecision"]:
                    opponent_href = past_fight["winnerHref"]
                    opponent_name = past_fight["winnerName"]

                    opponent_history = df_fights[
                        ((df_fights["winnerHref"] == opponent_href) | (df_fights["loserHref"] == opponent_href)) &
                        (df_fights["eventDate"] <= fight["eventDate"]) 
                    ].sort_values(by="eventDate", ascending=False)
                    
                    if not opponent_history.empty:
                        last_fight = opponent_history.iloc[0]
                        if last_fight["winnerHref"] == opponent_href:
                            opponent_points_on_d_day = last_fight["winner_points_after_fight"]
                        else:
                            opponent_points_on_d_day = last_fight["loser_points_after_fight"]
                        
                        if current_winner_points > opponent_points_on_d_day and loser_before_score < opponent_points_on_d_day:
                            print(winner_name)
                            print(f"Opponent: {opponent_name}")
                            print(f"Method of loss: {past_fight['method']}")
                            print(f"Current winner's points: {current_winner_points}")
                            print(f"Opponent's points on D day: {opponent_points_on_d_day}")
                            print(f"Loser before score: {loser_before_score}")

                            winner_points_after_fight = opponent_points_on_d_day - 0.01


        df_fights.loc[i, "winner_points_after_fight"] = winner_points_after_fight

        df_fights.loc[i, "loser_points_after_fight"] = loser_points_after_fight

    # If the fight is a draw
    elif df_fights.loc[i,'draw']=="draw":
        percentage_of_average_draw = 0.2
        value_for_draw = (weight_class_average * percentage_of_average_draw) * fight["belt_score"] * fight["performance_of_the_night_score"] * fight["fight_of_the_night_score"] * fight['rounds_score']

        winner_points_after_fight = winner_before_score + value_for_draw + 0.05 * (max(0,loser_before_score + winner_before_score))
        loser_points_after_fight = loser_before_score + value_for_draw + 0.05 * (max(0,loser_before_score + winner_before_score))

        df_fights.loc[i, "winner_points_after_fight"] = winner_points_after_fight

        df_fights.loc[i, "loser_points_after_fight"] = loser_points_after_fight
        
    # If the fight is a no contest
    elif df_fights.loc[i,'draw']=="nc":
        percentage_of_average_nc = 0.05
        value_for_nc = (weight_class_average * percentage_of_average_nc)

        winner_points_after_fight = winner_before_score + value_for_nc
        loser_points_after_fight = loser_before_score + value_for_nc

        df_fights.loc[i, "winner_points_after_fight"] = winner_points_after_fight

        df_fights.loc[i, "loser_points_after_fight"] = loser_points_after_fight
    
    last_fight_weight_class[winner_name] = weight_class
    last_fight_weight_class[loser_name] = weight_class
    
    # Update the dictionary after the fight
    fighter_last_weight_class_score.setdefault(winner, {})[weight_class] = (winner_points_after_fight, fight['peremption_coeff'])
    fighter_last_weight_class_score.setdefault(loser, {})[weight_class] = (loser_points_after_fight, fight['peremption_coeff'])



df_fights.drop(columns=['Unnamed: 0'], inplace=True)
df_fights.to_csv("fights.csv", sep=",")

# Load the list of active UFC fighters from ufc_fighters.csv
df_active_fighters = pd.read_csv("extraction/ufc_fighters.csv")
active_fighters_set = set(df_active_fighters['Fighter Name'].str.lower())  


# Determine the last two weight classes a fighter fought in
def get_last_two_weight_classes(fighter, fight_history):
    fighter_fights = fight_history[(fight_history['winnerHref'] == fighter) | (fight_history['loserHref'] == fighter)]
    fighter_fights = fighter_fights.sort_values(by='eventDate', ascending=False).head(2)
    return list(fighter_fights['weightClass'].unique())

# Initialize a new dictionary to track the last two weight classes for each fighter
fighter_last_two_weight_classes = {}

# Populate the last two weight classes for each fighter
for fighter in fighters:
    fighter_last_two_weight_classes[fighter] = get_last_two_weight_classes(fighter, df_fights)

# We retrieve the list of champions and interim champions:
champions = {}
interim_champions = {}

# Iterate through the active fighters DataFrame
for index, row in df_active_fighters.iterrows():
    weight_class = row['Weight Class']
    fighter_name = row['Fighter Name']
    status = row['Status']

    # Assign the fighter to the appropriate dictionary based on their status
    if status == 'C':
        champions[weight_class] = fighter_name
    elif status == 'IC':
        interim_champions[weight_class] = fighter_name


# Update the logic when calculating rankings to filter fighters based on their last two weight classes
last_fight_scores_dict = {}

# Populate the dictionary with the last fight scores for each fighter
for index, fight in df_fights.iterrows():
    for role in ['winner', 'loser']:
        fighter_href = fight[f'{role}Href']
        fighter_score = fight[f'{role}_points_after_fight'] * fight["peremption_coeff"]
        fighter_name = fight[f'{role}Name']
        weight_class = fight['weightClass']
        event_date = fight['eventDate']

        fighter_key = (fighter_href, weight_class)

        # Only consider fighters present in the active fighters list
        if fighter_name.lower() in active_fighters_set:
            if weight_class in fighter_last_two_weight_classes.get(fighter_href, []):
                if fighter_key not in last_fight_scores_dict or event_date > last_fight_scores_dict[fighter_key]['date']:
                    last_fight_scores_dict[fighter_key] = {
                        'Href': fighter_href,
                        'Name': fighter_name,
                        'WeightClass': weight_class,
                        'LastFightScore': fighter_score,
                        'date': event_date
                    }

last_fight_scores = pd.DataFrame(list(last_fight_scores_dict.values()))


# Function to assign ranks with grouping logic based on LastFightScore for top 20 fighters
def assign_grouped_ranks_top20(ranking_df, score_column='LastFightScore', rank_column='Rank', tolerance=1):
    ranking_df.sort_values(by=score_column, ascending=False, inplace=True)
    ranking_df.reset_index(drop=True, inplace=True)
    
    rank = 1
    grouped_fighters = set()
    
    top20_indices = set(ranking_df.index[:20]) 

    i = 0 
    while i < len(ranking_df):
        if i in grouped_fighters:
            i += 1
            continue

        ranking_df.at[i, rank_column] = rank
        grouped_fighters.add(i)

        group_size = 1
        
        for j in range(i + 1, len(ranking_df)):
            if j not in grouped_fighters and j in top20_indices and abs(ranking_df.at[i, score_column] - ranking_df.at[j, score_column]) <= tolerance:
                ranking_df.at[j, rank_column] = rank
                grouped_fighters.add(j)
                group_size += 1
        
        i += group_size
        
        rank += group_size

# Iterate over each weight class and apply the ranking logic
for weight_class in last_fight_scores['WeightClass'].unique():
    wc_ranking = last_fight_scores[last_fight_scores['WeightClass'] == weight_class].copy()
    wc_ranking.sort_values(by='LastFightScore', ascending=False, inplace=True)
    wc_ranking.reset_index(drop=True, inplace=True)

    wc = weight_class.lower()

    if wc in champions:
        champ_name = champions[wc]
        champ_row = wc_ranking[wc_ranking['Name'] == champ_name]

        # If the champion is not ranked first, adjust their position and score
        if not champ_row.empty and champ_row.index[0] != 0:
            champ_index = champ_row.index[0]
            max_score = wc_ranking.loc[0, 'LastFightScore']

            wc_ranking.loc[champ_index, 'LastFightScore'] = max_score + 0.1

            wc_ranking = pd.concat([wc_ranking.loc[[champ_index]], wc_ranking.drop(champ_index)]).reset_index(drop=True)

    # Check if the interim champion is present and adjust their position
    if wc in interim_champions:
        ic_name = interim_champions[wc]
        ic_row = wc_ranking[wc_ranking['Name'] == ic_name]

        # If the interim champion is not ranked second, adjust their position and score
        if not ic_row.empty and ic_row.index[0] != 1:
            ic_index = ic_row.index[0]
            second_score = wc_ranking.loc[1, 'LastFightScore'] if len(wc_ranking) > 1 else wc_ranking.loc[0, 'LastFightScore']

            wc_ranking.loc[ic_index, 'LastFightScore'] = second_score + 0.1

            wc_ranking = pd.concat([wc_ranking.iloc[[0]], wc_ranking.loc[[ic_index]], wc_ranking.drop(ic_index)]).reset_index(drop=True)

    # Ensure ranks are sequential after adjustments
    wc_ranking['Rank'] = wc_ranking.index + 1
    top20_ranking = wc_ranking.head(15)
    mean_last_fight_score_top20 = top20_ranking['LastFightScore'].mean()

    # Set tolerance as 3% of the mean of LastFightScore for the top 20 fighters
    tolerance = 0.005 * mean_last_fight_score_top20
    
    assign_grouped_ranks_top20(wc_ranking, score_column='LastFightScore', rank_column='Rank', tolerance=tolerance)
    
    wc_ranking['Rank'] = wc_ranking['Rank'].astype(int)

    wc_max_score = wc_ranking['LastFightScore'].max()
    wc_ranking['ScaledScore'] = (wc_ranking['LastFightScore'] / wc_max_score) * 100
    wc_ranking['ScaledScore'] = wc_ranking['ScaledScore'].round(2)
    
    wc_ranking[['Href', 'Name', 'LastFightScore', 'ScaledScore', 'Rank']].to_csv(f"{folder_name}/rank_{weight_class}.csv", sep=";", index=False)

