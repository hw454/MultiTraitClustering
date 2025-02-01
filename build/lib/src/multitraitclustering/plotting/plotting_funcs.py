import altair as alt
import numpy as np
import pandas as pd

def chart_clusters_multi(data, title, color_var, tooltip, xcol = None,
                        palette = None, clust_groups = None, col_list = []):
    if xcol is None:
        col1 = data.columns[0]
    else: col1 = xcol
    chart_dict = {}
    for col2 in col_list:
        if palette is not None:
            chart_dict[col2] = alt.Chart(data, title=title).mark_circle(size=60).encode(
                x = col1,
                y = col2,
                color = alt.Color(color_var, scale=alt.Scale(domain=clust_groups, range=palette)),
                tooltip = tooltip
            )
        else:
            chart_dict[col2] = alt.Chart(data, title=title).mark_circle(size=60).encode(
                x = col1,
                y = col2,
                color = color_var,
                tooltip = tooltip
            )
    return(chart_dict)

def chart_cluster_compare(data_array, xlabels, ylabels, x_lab, y_lab, z_lab, text_precision = ".0f"):
    # Convert this grid to columnar data expected by Altair
    ylen = data_array.shape[0]
    xlen = data_array.shape[1]
    x, y = np.meshgrid(range(0, xlen), range(0, ylen))
    x_clust = {int(np.where(xlabels == cn)[0]) : cn for cn in xlabels}
    y_clust = {int(np.where(ylabels == cn)[0]) : cn for cn in ylabels}
    z = data_array
    source = pd.DataFrame({x_lab: x.ravel(),
                     y_lab: y.ravel(),
                     z_lab: z.ravel()})
    for i in range(0, xlen):
        source.loc[source[x_lab] == i, x_lab] = x_clust[i]
    for j in range(0, ylen):
        source.loc[source[y_lab] == j, y_lab] = y_clust[j]
    chart = alt.Chart(source).mark_rect().encode(
        alt.X(x_lab+":O").title(x_lab),
        alt.Y(y_lab+":O").title(y_lab),
        alt.Color(z_lab+":Q").title(z_lab)
        ).properties(
            width=400,
            height=400
        )
    text = chart.mark_text(baseline='middle').encode(
    alt.Text(z_lab+':Q', format = text_precision),
    color=alt.condition(
        alt.datum[z_lab]> 40,
        alt.value('white'),
        alt.value('black')
    )
)
    return(chart+text)