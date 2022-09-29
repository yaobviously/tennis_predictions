# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 17:29:19 2022

@author: jyoung
"""


def elo_predict(elo_a=1500, elo_b=1500):
  "calculates the win probability of the first element of the pair of elos"

  prob_a = 1 / (1 + 10 ** ((elo_b - elo_a)/400))
  prob_b = 1 - prob_a

  return prob_a


def fiveodds(p3):
  "converts the probability of winning 3 sets into the probability of winning 5"

  p1 = np.roots([-2, 3, 0, -1*p3])[1]
  p5 = (p1**3)*(4 - 3*p1 + (6*(1-p1)*(1-p1)))

  return p5


def player_update_elo(winner='Rafael Nadal', loser='Pete Sampras', bestof=3, base_k=48,
                      n=31, j=8, player_dict=None):

  prematch_winner_elo = player_dict[winner]
  prematch_loser_elo = player_dict[loser]

  exp_a = 1 / (1 + 10 ** ((prematch_loser_elo - prematch_winner_elo)/400))
  exp_b = 1 - exp_a

  if bestof == 5:
    exp_a = fiveodds(exp_a)
    exp_b = 1 - exp_a

  if prematch_winner_elo <= 1500:
    base_k += n

  if prematch_loser_elo <= 1500:
    base_k += n

  if prematch_winner_elo >= 1800:
    base_k -= j

  if prematch_loser_elo >= 1800:
    base_k -= j

  winner_delta = base_k * (1 - exp_a)
  loser_delta = base_k * ((0) - exp_b)

  player_dict[winner] = prematch_winner_elo + winner_delta
  player_dict[loser] = prematch_loser_elo + loser_delta

  return exp_a


def process_elo(df_=None):

  elo_dict = {name: 1500 for name in df_.winner_name.unique()}

  for name in df_.loser_name.unique():
    if name not in elo_dict:
      elo_dict[name] = 1500

  def player_update_elo(winner='Rafael Nadal', loser='Pete Sampras', bestof=3, base_k=42,
                        n=22, j=5):
    "this function must be encapsulated to operate on the dict - create a class"

    prematch_winner_elo = elo_dict[winner]
    prematch_loser_elo = elo_dict[loser]

    exp_a = 1 / (1 + 10 ** ((prematch_loser_elo - prematch_winner_elo)/400))
    exp_b = 1 - exp_a

    if bestof == 5:
      exp_a = fiveodds(exp_a)
      exp_b = 1 - exp_a

    if prematch_winner_elo <= 1500:
      base_k += n

    if prematch_loser_elo <= 1500:
      base_k += n

    if prematch_winner_elo >= 1800:
      base_k -= j

    if prematch_loser_elo >= 1800:
      base_k -= j

    winner_delta = base_k * (1 - exp_a)
    loser_delta = base_k * ((0) - exp_b)

    elo_dict[winner] = prematch_winner_elo + winner_delta
    elo_dict[loser] = prematch_loser_elo + loser_delta

    return exp_a

  winner_probs = []

  for w, l, b in zip(df_.winner_name, df_.loser_name, df_.best_of):
    d = player_update_elo(winner=w, loser=l)
    winner_probs.append(d)

  df_['surface_winner_prob'] = winner_probs

  y = [1] * len(df_)

  return elo_dict, df_
