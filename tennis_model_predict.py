# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 15:47:01 2022

@author: jyoung
"""
import pandas as pd
import numpy as np
import dill

from sklearn.preprocessing import OrdinalEncoder

from xgboost import XGBClassifier

from trueskillthroughtime import Game, Player
from elo_funcs import elo_predict

# loading dictionaries
folder = "C:/Users/jyoung/Projects/tennis_project"

with open(f'{folder}/tennis_data/tennis_atp-master/trueskill_dict.pickle', "rb") as file:
    ts_dict = dill.load(file)
    
with open(f'{folder}/tennis_model_encoder.pkl', "rb") as file:
    encoder = dill.load(file)
    
with open('general_elo_dict.pkl', 'rb') as file:
    general_elo_dict = dill.load(file)

with open('hard_dict.pkl', 'rb') as file:
    hard_dict = dill.load(file)

with open('clay_dict.pkl', 'rb') as file:
    clay_dict = dill.load(file)

with open('grass_dict.pkl', 'rb') as file:
    grass_dict = dill.load(file)

with open('carpet_dict.pkl', 'rb') as file:
    carpet_dict = dill.load(file)


# loading the encoder and model

with open(f'{folder}/tennis_model_encoder.pkl', "rb") as file:
    encoder = dill.load(file)
    
model = XGBClassifier()
model.load_model('xgb_tennis_model.json')

# loading dataframe 

data_folder = "C:/Users/jyoung/Projects/tennis_project/tennis_data/tennis_atp-master"
df = pd.read_csv(f'{data_folder}/processed_apt_with_ts.csv', parse_dates=['tourney_date'])


# creating the dicts needed for the model from the df

# creating a dict for handedness
hand_dict = df.groupby("winner_name")['winner_hand'].last().to_dict()

# creating a dict for matches played
matches_played_dict = (
    df
    .groupby('winner_name')['winner_mp']
    .last()
    .to_dict()
)

# creating a dict for last tourney date
last_match_dict = (
    df
    .groupby('winner_name')['tourney_date']
    .last()
    .to_dict()
)


def get_model_win_proba(player_1=None, player_2=None, surface=None, round_=None,
                       tourney_level=None):
  

  today = pd.to_datetime('today').normalize()

  player_1_elo = general_elo_dict[player_1]
  player_1_ts = Player(ts_dict[player_1][-1][1])
  player_1_surface_elo = None
  player_1_hand = hand_dict[player_1]
  player_1_mp = matches_played_dict[player_1]
  player_1_rest = (today - last_match_dict[player_1]).days

  player_2_elo = general_elo_dict[player_2]
  player_2_ts = Player(ts_dict[player_2][-1][1])
  player_2_surface_elo = None
  player_2_hand = hand_dict[player_2]
  player_2_mp = matches_played_dict[player_2]
  player_2_rest = (today - last_match_dict[player_2]).days

  ts_proba = Game([[player_1_ts], [player_2_ts]]).evidence

  if surface == 'Hard':
    player_1_surface_elo = hard_dict[player_1]
    player_2_surface_elo = hard_dict[player_2]
  
  elif surface == 'Clay':
    player_1_surface_elo = clay_dict[player_1]
    player_2_surface_elo = clay_dict[player_2]

  elif surface == 'Carpet':
    player_1_surface_elo = carpet_dict[player_1]
    player_2_surface_elo = carpet_dict[player_2]

  else:
    player_1_surface_elo = grass_dict[player_1]
    player_2_surface_elo = grass_dict[player_2]

  
  df_dict = {
      'player_one_elo_proba' : elo_predict(player_1_elo, player_2_elo),
      'player_one_surface_win_proba' : elo_predict(player_1_surface_elo,
                                                   player_2_surface_elo),
      'player_one_hand' : hand_dict[player_1],
      'player_two_hand' : hand_dict[player_2],
      'tourney_level' : tourney_level,
      'round' : round_,
      'player_one_rest' : player_1_rest,
      'player_two_rest' : player_2_rest,
      'player_one_mp' : player_1_mp,
      'player_two_mp' : player_2_mp,
      'player_one_TS_proba' : ts_proba
  }

  pred_df = pd.DataFrame(df_dict, index=[0])
  pred_df['tourney_level'] = [str(x) for x in pred_df['tourney_level']]
  pred_df_cat = pred_df.select_dtypes('object').copy()
  pred_df[['player_one_hand', 'player_two_hand', 'tourney_level']] = encoder.transform(pred_df_cat)

  return model.predict_proba(pred_df)[0][1]
