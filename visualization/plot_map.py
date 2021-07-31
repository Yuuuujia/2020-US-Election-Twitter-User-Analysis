import pandas as pd
import plotly


# biden_tweets = pd.read_csv('data/with_prediction_biden.csv', lineterminator='\n', parse_dates=True)
# trump_tweets = pd.read_csv('data/with_prediction_trump.csv', lineterminator='\n', parse_dates=True)


def state(biden_tweets, trump_tweets):

    biden_tweets['country'].replace({'United States': 'United States of America'}, inplace=True)
    trump_tweets['country'].replace({'United States': 'United States of America'}, inplace=True)

    biden_with_state_code = biden_tweets[biden_tweets.state_code.isnull() == False]
    trump_with_state_code = trump_tweets[trump_tweets.state_code.isnull() == False]

    biden_US_state = biden_with_state_code[(biden_with_state_code.country == 'United States of America')]
    trump_US_state = trump_with_state_code[(trump_with_state_code.country == 'United States of America')]

    ML_state_b = biden_US_state.groupby('state_code')['sentiment_ML'].mean().reset_index(name="ML_sentiment_b")
    ML_state_t = trump_US_state.groupby('state_code')['sentiment_ML'].mean().reset_index(name="ML_sentiment_t")

    merge_ML = pd.merge(ML_state_b, ML_state_t, on='state_code')

    merge_ML['sentiment_difference'] = merge_ML['ML_sentiment_t'] - merge_ML['ML_sentiment_b']

    # Show figure in US map
    # Biden
    fig_biden = plotly.graph_objects.Figure(data = plotly.graph_objects.Choropleth(
        locations = ML_state_b['state_code'],
        z = ML_state_b['ML_sentiment_b'].astype(float),
        locationmode = 'USA-states',
        colorscale = 'rdbu',
        colorbar_title = "Sentiment",
    ))
    fig_biden.update_layout(
        title_text = 'The sentiment of each state (Biden)',
        geo_scope = 'usa',
    )

    # Trump
    fig_trump = plotly.graph_objects.Figure(data = plotly.graph_objects.Choropleth(
        locations = ML_state_t['state_code'],
        z = ML_state_t['ML_sentiment_t'].astype(float),
        locationmode = 'USA-states',
        colorscale = 'Bluered',
        colorbar_title = "Sentiment",
    ))
    fig_trump.update_layout(
        title_text = 'The sentiment of each state (Trump)',
        geo_scope = 'usa',
    )

    # Difference
    fig_diff = plotly.graph_objects.Figure(data = plotly.graph_objects.Choropleth(
        locations = merge_ML['state_code'],
        z = merge_ML['sentiment_difference'].astype(float),
        locationmode = 'USA-states',
        colorscale = 'Bluered',
        colorbar_title = "Sentiment difference",
    ))
    fig_diff.update_layout(
        title_text = 'The sentiment difference of each state: S(Trump) - S(Biden)',
        geo_scope = 'usa',
    )

    return fig_biden, fig_trump, fig_diff


def country(biden_tweets, trump_tweets, th_biden = 0.3, th_trump = 0.5, th_diff = 0.4):
    '''
    :param biden_tweets: dataframe of Biden tweets
    :param trump_tweets: dataframe of Trump tweets
    :param th_biden: data whose sentiment lager than this threshold will be treated as outlier and will be dropped
    :param th_trump: same as above
    :param th_diff: same as above
    :return: three figures: fig_biden, fig_trump and fig_diff
    '''

    biden_tweets['country'].replace({'United States': 'United States of America'}, inplace=True)
    trump_tweets['country'].replace({'United States': 'United States of America'}, inplace=True)

    biden_with_country = biden_tweets[biden_tweets.country.isnull() == False]
    trump_with_country = trump_tweets[trump_tweets.country.isnull() == False]

    ML_country_b = biden_with_country.groupby('country')['sentiment_ML'].mean().reset_index(name="ML_sentiment_b")
    ML_country_t = trump_with_country.groupby('country')['sentiment_ML'].mean().reset_index(name="ML_sentiment_t")

    merge_country = pd.merge(ML_country_b, ML_country_t, on='country')

    merge_country['sentiment_difference'] = merge_country['ML_sentiment_t'] - merge_country['ML_sentiment_b']

    ML_country_b_ = ML_country_b[abs(ML_country_b.ML_sentiment_b) < th_biden]  # 0.3
    ML_country_t_ = ML_country_t[abs(ML_country_t.ML_sentiment_t) < th_trump]
    country = merge_country[abs(merge_country.sentiment_difference) < th_diff]  # 0.4

    # Show figure in World map
    # Biden
    fig_biden = plotly.graph_objects.Figure(data = plotly.graph_objects.Choropleth(
        locations = ML_country_b_['country'],  # Spatial coordinates
        z = ML_country_b_['ML_sentiment_b'].astype(float),  # Data to be color-coded
        locationmode = 'country names',  # set of locations match entries in `locations`
        colorscale = 'rdbu',
        colorbar_title = "Sentiment",
    ))
    fig_biden.update_layout(
        title_text = 'World sentiment (Biden)',
        geo = dict(
            showcoastlines = True,
        )
    )

    # Trump
    fig_trump = plotly.graph_objects.Figure(data = plotly.graph_objects.Choropleth(
        locations = ML_country_t_['country'],  # Spatial coordinates
        z = ML_country_t_['ML_sentiment_t'].astype(float),  # Data to be color-coded
        locationmode = 'country names',  # set of locations match entries in `locations`
        colorscale = 'Bluered',
        colorbar_title = "Sentiment",
    ))
    fig_trump.update_layout(
        title_text = 'World sentiment (Trump)',
        geo = dict(
            showcoastlines = True,
        )
    )

    # Difference
    fig_diff = plotly.graph_objects.Figure(data = plotly.graph_objects.Choropleth(
        locations = country['country'],
        z = country['sentiment_difference'].astype(float),
        locationmode = 'country names',
        colorscale = 'Bluered',
        colorbar_title = "Sentiment difference",
    ))
    fig_diff.update_layout(
        title_text = 'World sentiment difference: S(Trump) - S(Biden)',
        geo = dict(
            showcoastlines = True,
        )
    )

    return fig_biden, fig_trump, fig_diff

def density(biden_tweets, trump_tweets):
    biden_tweets['country'].replace({'United States': 'United States of America'}, inplace=True)
    trump_tweets['country'].replace({'United States': 'United States of America'}, inplace=True)

    biden_with_state_code = biden_tweets[biden_tweets.state_code.isnull() == False]
    trump_with_state_code = trump_tweets[trump_tweets.state_code.isnull() == False]

    biden_US_state = biden_with_state_code[(biden_with_state_code.country == 'United States of America')]
    trump_US_state = trump_with_state_code[(trump_with_state_code.country == 'United States of America')]

    biden_counts = pd.DataFrame(biden_US_state.state_code.value_counts().reset_index(name="counts_biden"))
    trump_counts = pd.DataFrame(trump_US_state.state_code.value_counts().reset_index(name="counts_trump"))

    merge_counts = pd.merge(biden_counts, trump_counts, on='index')

    merge_counts['counts'] = merge_counts['counts_biden'] + merge_counts['counts_trump']

    merge_counts['counts'] /= sum(merge_counts['counts'])

    fig = plotly.graph_objects.Figure(data=plotly.graph_objects.Choropleth(
        locations=merge_counts['index'],
        z=merge_counts['counts'].astype(float),
        locationmode='USA-states',
        colorscale='brwnyl',
        colorbar_title="Density",
    ))
    fig.update_layout(
        title_text='Percentage of number of tweets of each state',
        geo_scope='usa',
    )

    return fig
