import numpy as np
import plotly.express as px


def concatenate_data(prediction_test_label, expected_test_label, prediction_train_label, expected_train_label):
    molecules = ['C₂H₆', 'CH₄', 'CO₂', 'H₂', 'H₂O', 'H₂S', 'N₂', 'N₂O', 'NH₂', 'O₂', 'O₃', 'PH₃']

    prediction_test_label_reshape = prediction_test_label.T.reshape(-1)
    prediction_train_label_reshape = prediction_train_label.T.reshape(-1)
    molecules_row = []
    for i in range(prediction_test_label.shape[1]):
        molecules_row += [molecules[i]] * len(prediction_test_label)
    for i in range(prediction_train_label.shape[1]):
        molecules_row += [molecules[i]] * len(prediction_train_label)
    return {'Prezis': np.concatenate((prediction_test_label_reshape, prediction_train_label_reshape)),
            'Adevărat': np.concatenate((expected_test_label.T.reshape(-1), expected_train_label.T.reshape(-1))),
            'type': ['test'] * np.prod(prediction_test_label_reshape.shape) + ['train'] * np.prod(
                prediction_train_label.shape),
            'moleculă': molecules_row}


def show_plot(p_data_y_test, e_data_y_test, p_data_y_train, e_data_y_train, limit=(1, 1), length=None):
    df = concatenate_data(p_data_y_test, e_data_y_test, p_data_y_train, e_data_y_train)

    fig = px.scatter(df,
                     x='Adevărat', y='Prezis', color='type', animation_frame='moleculă',
                     marginal_x='histogram', marginal_y='histogram', trendline='ols', labels={'value': 'Prediction'},
                     range_x=[0, limit[0]], range_y=[0, limit[1]],
                     height=length, width=length,
                     )
    fig.update_traces(histnorm='probability', selector={'type': 'histogram'})
    fig.add_shape(
        type="line", line=dict(dash='dash'),
        x0=0, y0=0,
        x1=1, y1=1
    )

    fig["layout"].pop("updatemenus")
    s = fig['layout'].sliders[0]
    for step in s['steps']:
        step['args'][1]['frame']['redraw'] = True

    fig.show()


def show_plot_neat(nn, limit, train_feature, train_label, test_feature, test_label):
    train_predict = np.array([np.array(nn.activate(train_feature[j])) for j in range(len(train_feature))])
    test_predict = np.array([np.array(nn.activate(test_feature[j])) for j in range(len(test_feature))])
    show_plot(test_predict, test_label, train_predict, train_label, limit, 600)
