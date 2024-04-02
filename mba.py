import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from mlxtend.frequent_patterns import apriori

# Function to load data and cache it
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Function to find frequent itemsets using Apriori algorithm
def find_frequent_items(transactions, min_support):
    try:
        # Convert transactional data to one-hot encoded format
        one_hot_encoded_df = pd.get_dummies(transactions)
        
        # Find frequent itemsets using Apriori algorithm
        frequent_itemsets = apriori(one_hot_encoded_df, min_support=min_support, use_colnames=True)
        
        return frequent_itemsets
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")

# Main function
def main():
    st.title("Frequent Items Analysis")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.write(df.head())  # Display first few rows of the dataframe

        # Slider widget for minimum support level
        min_support = st.slider("Select Minimum Support Level", min_value=0.0, max_value=1.0, step=0.01, value=0.1)

        # Button to trigger frequent itemset computation
        if st.button("Find Frequent Itemsets"):
            frequent_itemsets = find_frequent_items(df['Item Purchased'], min_support)
            if frequent_itemsets is not None:
                # Filter frequent itemsets based on support level
                frequent_itemsets_filtered = frequent_itemsets[frequent_itemsets['support'] >= min_support]
                if not frequent_itemsets_filtered.empty:
                    st.write(frequent_itemsets_filtered)
                else:
                    st.warning("No frequent itemsets found with the selected support level.")
            else:
                st.error("No frequent itemsets found.")

# Function to display the "Customer Behavior by Season" section
def show_customer_behavior_by_season(df, season):
    st.title("Customer Behavior by Season")
    if 'Season' in df.columns:
        season_df = df[df['Season'] == season]
        season_counts = season_df['Item Purchased'].value_counts()
        fig = px.bar(season_counts, x=season_counts.index, y=season_counts.values,
                     labels={'x': 'Item Purchased', 'y': 'Count'}, title=f"Customer Behavior in {season} Season")
        st.plotly_chart(fig)
    else:
        st.error("The DataFrame does not contain the 'Season' column.")

# Function to display the "Detailed Representation" section
def show_detailed_representation(df, selected_column, selected_season):
    st.title("Detailed Representation")

    # Check if 'Season' column exists in the DataFrame
    if 'Season' not in df.columns:
        st.error("The DataFrame does not contain the 'Season' column.")
        return

    # Check if a season is selected
    if selected_season:
        # Filter DataFrame based on selected season and column
        selected_df = df[[selected_column, 'Season']][df['Season'] == selected_season]
        st.write(selected_df)

# Function to display the "About" section
def show_about():
    st.subheader("About Us")
    st.write(
        "We are a team of data analysts dedicated to helping businesses understand their customers better "
        "and make informed decisions based on data-driven insights. With our expertise in machine learning "
        "and data mining techniques, we provide valuable insights into customer behavior and preferences."
    )
    
    st.subheader("Our Approach")
    st.write(
        "Our approach involves analyzing transactional data using the Apriori algorithm, a classic algorithm "
        "in data mining used for association rule learning. By identifying frequent itemsets and association "
        "rules from transaction data, we can uncover hidden patterns and relationships in customer behavior."
    )
    
    st.subheader("How It Works")
    st.write(
        "1. **Data Collection**: We gather transactional data from your business, including information "
        "such as customer IDs, product IDs, and purchase timestamps."
        "\n\n"
        "2. **Data Preprocessing**: We preprocess the data to remove noise and prepare it for analysis. "
        "This may include tasks such as handling missing values, encoding categorical variables, and scaling "
        "numeric features."
        "\n\n"
        "3. **Apriori Algorithm**: We apply the Apriori algorithm to the preprocessed data to discover frequent "
        "itemsets and association rules."
        "\n\n"
        "4. **Insights Generation**: We interpret the results of the Apriori algorithm to generate insights "
        "into customer behavior and seasonal purchase patterns. These insights can inform business strategies "
        "such as product recommendations, marketing campaigns, and inventory management."
    )
    
    st.subheader("Contact Us")
    st.write(
        "If you're interested in leveraging our expertise to gain valuable insights into your business, "
        "please feel free to reach out to us. We'd love to discuss how we can help you achieve your goals."
        "\n\n"
        "Email: contact@example.com\n"
        "Phone: +1 (123) 456-7890"
    )

# Main function
def main():
    st.markdown("# Consumer Behavior and Shopping Habits Dataset:")
    st.write("E-Commerce Transaction Trends: A Comprehensive Dataset:")

    st.title("Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        dashboard_selectbox = st.sidebar.selectbox('Select Dashboard View:',
                                                   ('Description', 'Frequent items', 'Seasonal Shopping', 'About'))

        # Main content
        if dashboard_selectbox == 'Description':
            st.header('Description')
            st.subheader("Dataset Description:")
            st.write(df.describe())

            st.write("3D Animated Graphs:")
            for column in df.columns:
                if df[column].dtype in ['int64', 'float64'] and column != 'id':
                    st.write(f"Column: {column}")
                    fig = px.scatter_3d(df, x=df.columns[0], y=df.columns[1], z=df.columns[2], animation_frame=column,
                                         title=f"3D Animated Plot for {column}", template="plotly_dark")
                    st.plotly_chart(fig)

        elif dashboard_selectbox == 'Frequent items':
            st.header('Frequent items')
            
            # Slider widget to adjust minimum support level
            min_support = st.slider("Select Minimum Support Level", min_value=0.0, max_value=1.0, step=0.01, value=0.1)
            
            # Find frequent itemsets
            if 'Item Purchased' in df.columns:
                frequent_itemsets = find_frequent_items(df['Item Purchased'], min_support)
                # Display frequent itemsets
                st.subheader("Frequent Itemsets:")
                st.write(frequent_itemsets)
            else:
                st.error("The DataFrame does not contain the 'Item Purchased' column.")

        elif dashboard_selectbox == 'Seasonal Shopping':
            st.header('Seasonal Shopping')
            if 'Season' in df.columns:
                season_options = df['Season'].unique().tolist()
                selected_season = st.selectbox("Select Season", season_options)
                show_customer_behavior_by_season(df, selected_season)
                
                # Selectbox widget for selecting columns
                selected_column = st.selectbox("Select Column", df.columns.tolist())
                show_detailed_representation(df, selected_column, selected_season)
            else:
                st.error("The DataFrame does not contain the 'Season' column.")

        elif dashboard_selectbox == 'About':
            show_about()

# Run the app
if __name__ == "__main__":
    main()
