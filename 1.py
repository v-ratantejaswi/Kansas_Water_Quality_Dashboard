# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from sklearn.preprocessing import StandardScaler

# # Assuming 'comb_df' is your combined DataFrame and already prepared
# comb_df = pd.read_csv('5ele.csv')
# comb_df['Date'] = pd.to_datetime(comb_df['ActivityStartDate'])
# comb_df['Year'] = comb_df['Date'].dt.year
# comb_df['Month'] = comb_df['Date'].dt.month
# # Function to scale values within each group, adjusted with error handling
# def scale_values(df, column_name='ResultMeasureValue'):
#     if df.empty:
#         return df  # Return the empty DataFrame as is if no rows match
#     scaler = StandardScaler()
#     scaled_values = scaler.fit_transform(df[[column_name]])
#     df['ScaledValue'] = scaled_values
#     return df

# Define threshold values for each characteristic
thresholds = {
    'nitrate': 10,
    'oxygen': 8,
    'dissolved oxygen': 8,
    'calcium': 100,
    'magnesium': 30,
    'arsenic': 0.01
}

# # Streamlit UI
# st.title("Water Quality Analysis Dashboard")

# # Visualization type selection
# vis_type = st.selectbox("Select the visualization type:", ["Average Scaled Value", "Tests Pass/Fail Stacked Bar"])

# # Characteristic name selection
# selected_characteristic = st.selectbox("Select Characteristic Name:", options=list(thresholds.keys()))

# if vis_type == "Average Scaled Value":
#     # Filter DataFrame by selected characteristic
#     df_filtered = comb_df[comb_df['CharacteristicName'].str.lower() == selected_characteristic.lower()]
#     # Scale values if DataFrame is not empty
#     if not df_filtered.empty:
#         df_scaled = scale_values(df_filtered)
#         # Plotting
#         fig = px.line(df_scaled, x='Year', y='ScaledValue', color='CharacteristicName', title=f"Average Scaled Values for {selected_characteristic}")
#         st.plotly_chart(fig)
#     else:
#         st.write("No data available for the selected characteristic.")

# elif vis_type == "Tests Pass/Fail Stacked Bar":
#     # Implement the logic for the stacked bar chart, similar to the previous discussions
#     pass  # Placeholder for the actual implementation

# # Ensure to adjust 'comb_df', column names, and other placeholders to fit your actual data structure


import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import json
import geopandas as gpd
import plotly.express as px

# Load and prepare the DataFrame
comb_df = pd.read_csv('5ele.csv')
comb_df['Date'] = pd.to_datetime(comb_df['ActivityStartDate'])
comb_df['Year'] = comb_df['Date'].dt.year
comb_df['Month'] = comb_df['Date'].dt.month

avg_ph = pd.read_csv('avg_ph.csv')
with open('KS_Counties.geojson', 'r') as f:
    kansas_geojson = json.load(f)
all_kansas_counties = [feature['properties']['COUNTY'] for feature in kansas_geojson['features']]
all_kansas_counties_df = pd.DataFrame(all_kansas_counties, columns=['County Name'])
avg_ph['County Name'] = avg_ph['County Name'].str.title()
complete_counties = all_kansas_counties_df.merge(avg_ph, on='County Name', how='left')

# Function to scale values within each group
def scale_values(group):
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(group['ResultMeasureValue'].values.reshape(-1, 1))
    group['ScaledValue'] = scaled_values.flatten()  # Assign scaled values
    return group

# Apply scaling function and ensure 'CharacteristicName' is a column
df_scaled = comb_df.groupby('CharacteristicName', as_index=False).apply(scale_values)
df_scaled = df_scaled.reset_index(drop=True)  # Reset index to avoid 'CharacteristicName' ambiguity

# Group by Year, Month, and CharacteristicName to calculate monthly average scaled values
grouped_df = df_scaled.groupby(['Year', 'Month', 'CharacteristicName'], as_index=False)['ScaledValue'].mean()

# Convert Year and Month into a Date for plotting
grouped_df['Date'] = pd.to_datetime(grouped_df[['Year', 'Month']].assign(DAY=1))

# Streamlit UI for visualization selection and characteristic name selection
st.title("Water Quality Analysis Dashboard")
vis_type = st.selectbox("Select the visualization type:", ["Average Scaled Value", "Tests Pass/Fail Stacked Bar", "Avg pH choropleth"])
selected_characteristic = st.selectbox("Select Characteristic Name:", options=list(thresholds.keys()))

if vis_type == "Average Scaled Value":
    # Filter based on selected characteristic and plot
    filtered_grouped_df = grouped_df[grouped_df['CharacteristicName'].str.lower() == selected_characteristic.lower()]
    if not filtered_grouped_df.empty:
        fig = px.line(filtered_grouped_df, x='Date', y='ScaledValue', color='CharacteristicName',
                      title=f'Monthly Average Scaled Values by CharacteristicName for {selected_characteristic}',
                      labels={'ScaledValue': 'Average Scaled Value', 'Date': 'Date'}, markers=True)
        fig.update_layout(xaxis_title='Date', yaxis_title='Average Scaled Value', hovermode='x unified')
        st.plotly_chart(fig)
    else:
        st.write("No data available for the selected characteristic.")
elif vis_type == "Tests Pass/Fail Stacked Bar":
    df_filtered = comb_df[comb_df['CharacteristicName'].str.lower() == selected_characteristic.lower()]
    # Determine pass/fail based on threshold
    threshold = thresholds[selected_characteristic]
    df_filtered['Status'] = df_filtered['ResultMeasureValue'].apply(lambda x: 'Pass' if x <= threshold else 'Fail')
    
    # Count the number of Pass and Fail by Year
    counts = df_filtered.groupby(['Year', 'Status'])['CharacteristicName'].count().reset_index()
    
    # Create stacked bar chart
    fig = px.bar(counts, x='Year', y='CharacteristicName', color='Status', barmode='stack', 
                 labels={'CharacteristicName': 'Number of Tests'}, title=f'Test Results for {selected_characteristic}')
    st.plotly_chart(fig)
elif vis_type == "Avg pH choropleth":
    complete_counties['ResultMeasureValue'].fillna(-1, inplace=True)

    fig = px.choropleth(complete_counties,
                    geojson=kansas_geojson,  # Use your GeoJSON data here
                    locations='County Name',
                    featureidkey="properties.COUNTY",  # Adjusted to match your GeoJSON file
                    color='ResultMeasureValue',
                    color_continuous_scale="Viridis",
                    scope="usa",
                    labels={'ResultMeasureValue':'Average pH'},
                    title='Average pH Levels by County in Kansas',
                    hover_name='County Name',  # Show county names on hover
                    # Use a range color to set the special value (-1) to white
                    range_color=[0, max(complete_counties['ResultMeasureValue'])]
                   )

    fig.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig) 
