import pandas as pd
from visualization import plot_map


biden_tweets = pd.read_csv('data/with_prediction_biden.csv', lineterminator='\n', parse_dates=True)
trump_tweets = pd.read_csv('data/with_prediction_trump.csv', lineterminator='\n', parse_dates=True)


fig_biden_state, fig_trump_state, fig_diff_state = plot_map.state(biden_tweets, trump_tweets)
# fig_biden_state.show()
# fig_trump_state.show()
# fig_diff_state.show()

fig_biden_country, fig_trump_country, fig_diff_country = plot_map.country(biden_tweets, trump_tweets, th_biden = 0.4, th_trump = 0.4, th_diff = 0.4)
# fig_biden_country.show()
# fig_trump_country.show()
# fig_diff_country.show()

fig_density = plot_map.density(biden_tweets, trump_tweets)
fig_density.show()