import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from transformers import BertTokenizer



def distribution_labels(df, label_name):
    # Assuming 'emotion' is the label column in your dataset
    groupby_label = df.groupby(label_name)[label_name].count()

    # Create the bar chart using Plotly
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=list(sorted(groupby_label.index)),
        y=groupby_label.tolist(),
        text=groupby_label.tolist(),
        textposition='auto'
    ))

    # Update the layout of the figure
    fig.update_layout(
        title_text=f'Distribution of {label_name} Labels within Comments [DF]',
        xaxis_title_text=f'{label_name} Label',
        yaxis_title_text='Frequency',
        bargap=0.2,
        bargroupgap=0.2
    )
    # Show the figure
    fig.write_image('reports/distribuation_labels.png')
    fig.show()



def word_count_distribution(df, text_col_name):
    tokenizer = BertTokenizer.from_pretrained('HooshvareLab/bert-fa-base-uncased')
    df['text_len_by_words'] = df[text_col_name].apply(lambda x: len(tokenizer.tokenize(str(x))))

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df['text_len_by_words']
    ))

    fig.update_layout(
        title_text='Distribution of word counts within context',
        xaxis_title_text='Word Count',
        yaxis_title_text='Frequency',
        bargap=0.2,
        bargroupgap=0.2)

    fig.write_image('reports/bert_tokenized.png')
    fig.show()
