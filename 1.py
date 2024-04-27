
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import json
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

thresholds = {
    'nitrate': 10,
    'oxygen': 8,
    'dissolved oxygen': 8,
    'calcium': 100,
    'magnesium': 30,
    'arsenic': 0.01
}




comb_df = pd.read_csv('5ele.csv')
comb_df['Date'] = pd.to_datetime(comb_df['ActivityStartDate'])
comb_df['Year'] = comb_df['Date'].dt.year
comb_df['Month'] = comb_df['Date'].dt.month

print(comb_df['Year'].unique())

avg_ph = pd.read_csv('avg_ph.csv')
with open('KS_Counties.geojson', 'r') as f:
    kansas_geojson = json.load(f)
all_kansas_counties = [feature['properties']['COUNTY'] for feature in kansas_geojson['features']]
all_kansas_counties_df = pd.DataFrame(all_kansas_counties, columns=['County Name'])
avg_ph['County Name'] = avg_ph['County Name'].str.title()
complete_counties = all_kansas_counties_df.merge(avg_ph, on='County Name', how='left')

population_df = pd.read_excel('population.xlsx')


population_df['County'] = population_df['County'].str.replace(r"\.|\sCounty, Kansas", "", regex=True).str.strip()


population_long_df = population_df.melt(id_vars='County', var_name='Year', value_name='Population')
population_long_df['Year'] = population_long_df['Year'].astype(int)


# Load the income data
income_df = pd.read_csv('kansas-income.csv')

# Rename the income columns to keep only the year
income_df.rename(columns=lambda x: x.strip().replace('Income ', ''), inplace=True)

# Transform the income data into a long format
income_long_df = income_df.melt(id_vars=['County Name', 'Rank in State 2022', 'Percent Change 2021', 'Percent Change 2022', 'Rank of Percent Change 2022', 'State'], 
                                var_name='Year', value_name='Income')

# Make sure 'Year' is an integer
income_long_df['Year'] = income_long_df['Year'].astype(int)


comb_df['Year'] = comb_df['Year'].astype(int)


unique_counties = comb_df['County Name'].unique()


population_filtered_df = population_long_df[population_long_df['County'].isin(unique_counties)]


combined_df = pd.merge(comb_df, population_filtered_df, left_on=['County Name', 'Year'], right_on=['County', 'Year'], how='inner')

combined_income_df = pd.merge(combined_df, income_long_df, 
                              left_on=['County Name', 'Year'], 
                              right_on=['County Name', 'Year'], 
                              how='inner')

def scale_values(group):
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(group['ResultMeasureValue'].values.reshape(-1, 1))
    group['ScaledValue'] = scaled_values.flatten()
    return group


df_scaled = comb_df.groupby('CharacteristicName', as_index=False).apply(scale_values)
df_scaled = df_scaled.reset_index(drop=True)


grouped_df = df_scaled.groupby(['Year', 'Month', 'CharacteristicName'], as_index=False)['ScaledValue'].mean()


grouped_df['Date'] = pd.to_datetime(grouped_df[['Year', 'Month']].assign(DAY=1))




st.title("Water Quality Analysis Dashboard")
vis_type = st.selectbox("Select the visualization type:", ["Average Scaled Value", "Combined Average Scaled Values", "Tests Pass/Fail Stacked Bar", "Avg pH choropleth",
                                                           "Population Plot", "Income Plot"])


if vis_type == "Average Scaled Value":
    
    selected_characteristic = st.selectbox("Select Characteristic Name:", options=list(thresholds.keys()))
    filtered_grouped_df = grouped_df[grouped_df['CharacteristicName'].str.lower() == selected_characteristic.lower()]
    if not filtered_grouped_df.empty:
        fig = px.line(filtered_grouped_df, x='Date', y='ScaledValue', color='CharacteristicName',
                      title=f'Monthly Average Scaled Values by CharacteristicName for {selected_characteristic}',
                      labels={'ScaledValue': 'Average Scaled Value', 'Date': 'Date'}, markers=True)
        fig.update_layout(xaxis_title='Date', yaxis_title='Average Scaled Value', hovermode='x unified')
        st.plotly_chart(fig)
    else:
        st.write("No data available for the selected characteristic.")
elif vis_type == "Combined Average Scaled Values":
    grouped_df = df_scaled.groupby(['Year', 'Month', 'CharacteristicName'])['ScaledValue'].mean().reset_index()


    grouped_df['Date'] = pd.to_datetime(grouped_df[['Year', 'Month']].assign(DAY=1))


    fig = px.line(grouped_df, x='Date', y='ScaledValue', color='CharacteristicName',
              title='Monthly Average Scaled Values by CharacteristicName for Kansas State',
              labels={'ScaledValue': 'Average Scaled Value', 'Date': 'Date'},
              markers=True)


    fig.update_layout(xaxis_title='Date', yaxis_title='Average Scaled Value',
                  hovermode='x unified')

    st.plotly_chart(fig)
elif vis_type == "Tests Pass/Fail Stacked Bar":
    selected_characteristic = st.selectbox("Select Characteristic Name:", options=list(thresholds.keys()))
    df_filtered = comb_df[comb_df['CharacteristicName'].str.lower() == selected_characteristic.lower()]
    
    threshold = thresholds[selected_characteristic]
    df_filtered['Status'] = df_filtered['ResultMeasureValue'].apply(lambda x: 'Pass' if x <= threshold else 'Fail')
    
 
    counts = df_filtered.groupby(['Year', 'Status'])['CharacteristicName'].count().reset_index()
    
 
    fig = px.bar(counts, x='Year', y='CharacteristicName', color='Status', barmode='stack', 
                 labels={'CharacteristicName': 'Number of Tests'}, title=f'Test Results for {selected_characteristic}')
    fig.update_xaxes(
        title_text="Year",
        tickvals=counts['Year'].unique(), 
        dtick=1 
    )
    st.plotly_chart(fig)
elif vis_type == "Avg pH choropleth":
    complete_counties['ResultMeasureValue'].fillna(-1, inplace=True)

    complete_counties['ResultMeasureValue'] = complete_counties['ResultMeasureValue'].astype(float)


    min_val = complete_counties[complete_counties['ResultMeasureValue'] > -1]['ResultMeasureValue'].min()
    max_val = complete_counties['ResultMeasureValue'].max()


    step = (max_val - min_val) / 5
    custom_color_scale = [
        [0.0, "#D3D3D3"],
        [(min_val - -1) / (max_val - -1), "#D3D3D3"],
        [(min_val + step - -1) / (max_val - -1), px.colors.sequential.Plasma[0]],
        [(min_val + 2*step - -1) / (max_val - -1), px.colors.sequential.Plasma[1]],
        [(min_val + 3*step - -1) / (max_val - -1), px.colors.sequential.Plasma[2]],
        [(min_val + 4*step - -1) / (max_val - -1), px.colors.sequential.Plasma[3]],
        [1.0, px.colors.sequential.Plasma[-1]],
    ]

    fig = px.choropleth(
        complete_counties,
        geojson=kansas_geojson,
        locations='County Name',
        featureidkey="properties.COUNTY",
        color='ResultMeasureValue',
        color_continuous_scale=custom_color_scale,
        range_color=[-1, max_val],
        labels={'ResultMeasureValue': 'Average pH'},
        title='Average pH Levels by County in Kansas',
    )


    fig.update_layout(
        coloraxis_colorbar=dict(
            title='Average pH',
            tickvals=np.linspace(-1, max_val, num=7),
            ticktext=[f"{val:.2f}" for val in np.linspace(-1, max_val, num=7)]
        )
    )

    fig.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig)
elif vis_type == "Population Plot":
 
    selected_characteristic = st.selectbox("Select Characteristic Name:", options=list(thresholds.keys()))
    county_options = combined_df['County Name'].unique()
    selected_county = st.selectbox("Select a County:", options=county_options)

  
    filtered_data = combined_df[(combined_df['County Name'] == selected_county) & 
                                (combined_df['CharacteristicName'].str.lower() == selected_characteristic.lower())]
    

    avg_metric_per_year = filtered_data.groupby('Year')['ResultMeasureValue'].mean().reset_index()


    county_population = population_filtered_df[population_filtered_df['County'].isin([selected_county]) & 
                                               population_filtered_df['Year'].isin(avg_metric_per_year['Year'])]


    fig = make_subplots(specs=[[{"secondary_y": True}]])


    fig.add_trace(go.Scatter(x=avg_metric_per_year['Year'], y=avg_metric_per_year['ResultMeasureValue'],
                             name=f'Avg {selected_characteristic}', mode='lines+markers'), secondary_y=False)


    fig.add_trace(go.Scatter(x=county_population['Year'], y=county_population['Population'],
                             name='Population', mode='lines+markers'), secondary_y=True)


    fig.update_layout(title_text=f"Average {selected_characteristic} and Population Over Time for {selected_county}")

  
    fig.update_xaxes(title_text="Year",
                    tickvals=county_population['Year'].unique(),
                    dtick=1)


    fig.update_yaxes(title_text=f"Average {selected_characteristic}", secondary_y=False)
    fig.update_yaxes(title_text="Population", secondary_y=True)

    st.plotly_chart(fig)



elif vis_type == "Income Plot":
    selected_county = st.selectbox("Select a County for Income Data:", options=combined_income_df['County Name'].unique())
    
    selected_characteristic = st.selectbox("Select Characteristic Name:", options=list(thresholds.keys()))
    
    filtered_income_data = combined_income_df[combined_income_df['County Name'] == selected_county]
    filtered_characteristic_data = comb_df[(comb_df['County Name'] == selected_county) & 
                                           (comb_df['CharacteristicName'].str.lower() == selected_characteristic.lower())]

    avg_income_per_year = filtered_income_data.groupby('Year')['Income'].mean().reset_index()
    avg_characteristic_per_year = filtered_characteristic_data.groupby('Year')['ResultMeasureValue'].mean().reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=avg_income_per_year['Year'], y=avg_income_per_year['Income'],
                             name='Average Income', mode='lines+markers'), secondary_y=False)

    fig.add_trace(go.Scatter(x=avg_characteristic_per_year['Year'], y=avg_characteristic_per_year['ResultMeasureValue'],
                             name=f'Avg {selected_characteristic}', mode='lines+markers'), secondary_y=True)

    fig.update_layout(title_text=f"Income and {selected_characteristic} Over Time for {selected_county}")
    fig.update_xaxes(title_text="Year",
                    tickvals=avg_characteristic_per_year['Year'].unique(),
                    dtick=1)
    fig.update_yaxes(title_text="Average Income", secondary_y=False)
    fig.update_yaxes(title_text=f"Average {selected_characteristic}", secondary_y=True)

    st.plotly_chart(fig)



