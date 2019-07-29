import numpy as np

n_try = 10000
decks = np.array([[0]*28 + [1]*4 + [2]*4 + [3]*4] *n_try)

sh_decks = np.apply_along_axis(np.random.permutation, 1, decks)

mask = np.array([1,2,3]*13 + [1])

wins = np.count_nonzero(np.product(sh_decks - mask, axis = 1))
wins
win_prob = wins/n_try
win_prob
#%%

deck = np.arange(1,41)
np.random.shuffle(deck)
deck[::3]
