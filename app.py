import streamlit as st
import pandas as pd
import joblib

# --- Model and Data Loading ---
@st.cache_resource
def load_models_and_data():
    try:
        yield_model = joblib.load('models/yield_predictor.pkl')
        # Load data for dropdowns
        df = pd.read_csv('data/crop_yield.csv')
        crops = sorted([crop.strip() for crop in df['Crop'].unique()])
        seasons = sorted([season.strip() for season in df['Season'].unique()])
        states = sorted([state.strip() for state in df['State'].unique()])
        return yield_model, crops, seasons, states
    except FileNotFoundError as e:
        st.error(f"Error loading a required file: {e}. Please ensure all necessary files are present.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during model loading: {e}")
        st.stop()

model, CROPS, SEASONS, STATES = load_models_and_data()

# --- UI Configuration ---
st.set_page_config(page_title="AgriAI Crop Suggester", layout="wide", initial_sidebar_state="expanded")

# --- UI Setup ---
st.title("🌾 AgriAI Crop Suggester")
st.markdown("Get crop suggestions with yield predictions based on your farming conditions.")

# --- Input Form ---
st.sidebar.title("Input Your Conditions")
with st.sidebar.form(key='suggestion_input_form'):
    st.subheader("Location and Time")
    state = st.selectbox("Select State", STATES, index=STATES.index("Assam"))
    season = st.selectbox("Select Season", SEASONS, index=SEASONS.index("Kharif"))
    crop_year = st.number_input("Crop Year", min_value=1997, max_value=2040, value=2024)

    st.subheader("Farming Inputs")
    area = st.number_input("Area (in Hectares)", min_value=1.0, value=100.0)
    annual_rainfall = st.number_input("Annual Rainfall (in mm)", min_value=0.0, value=1500.0)
    fertilizer = st.number_input("Fertilizer Usage (in Tons)", min_value=0.0, value=5000.0)
    pesticide = st.number_input("Pesticide Usage (in Tons)", min_value=0.0, value=100.0)
    
    submit_button = st.form_submit_button('Suggest Best Crops')

# --- Main Processing and Output ---
if submit_button:
    with st.spinner("Analyzing conditions and predicting yields for all crops... This may take a moment."):
        try:
            # --- Iterative Prediction Logic ---
            predictions = []
            base_input = {
                'Crop_Year': [crop_year],
                'Season': [season],
                'State': [state],
                'Area': [area],
                'Annual_Rainfall': [annual_rainfall],
                'Fertilizer': [fertilizer],
                'Pesticide': [pesticide]
            }
            
            for crop in CROPS:
                # Create a dataframe for the current crop
                current_input = base_input.copy()
                current_input['Crop'] = [crop]
                input_df = pd.DataFrame(current_input)
                
                # Ensure column order matches training
                training_columns = ['Crop_Year', 'Season', 'State', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Crop']
                input_df = input_df[training_columns]
                
                # Predict yield and store it
                predicted_yield = model.predict(input_df)[0]
                # Avoid negative yield predictions
                if predicted_yield > 0:
                    predictions.append({'Crop': crop, 'Predicted_Yield': predicted_yield})

            if not predictions:
                st.error("Could not generate any valid crop suggestions for the provided inputs.")
            else:
                # Sort crops by predicted yield
                top_crops = sorted(predictions, key=lambda x: x['Predicted_Yield'], reverse=True)[:5]
                top_crops_df = pd.DataFrame(top_crops)
                
                # --- Display Results ---
                st.header("Crop Suggestions")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Top Recommendation")
                    top_crop_name = top_crops_df.iloc[0]['Crop']
                    top_crop_yield = top_crops_df.iloc[0]['Predicted_Yield']
                    st.success(f"**{top_crop_name.upper()}**")
                    st.metric(
                        label="Predicted Yield",
                        value=f"{top_crop_yield:.2f} tons/hectare"
                    )
                
                with col2:
                    st.subheader("Top 5 Suggestions by Yield")
                    st.dataframe(top_crops_df,
                                 column_config={
                                     "Crop": st.column_config.TextColumn("Crop"),
                                     "Predicted_Yield": st.column_config.ProgressColumn(
                                         "Predicted Yield (tons/hectare)",
                                         format="%.2f",
                                         min_value=0,
                                         max_value=top_crops_df['Predicted_Yield'].max(),
                                     ),
                                 },
                                 hide_index=True,
                                 use_container_width=True)

                # Placeholder for RAG advisory for the top crop
                st.subheader(f"Detailed Advisory for {top_crop_name.upper()}")
                st.info(f"Advisory details for {top_crop_name} would be displayed here using the RAG system.")

        except Exception as e:
            st.error(f"An error occurred during the suggestion process: {e}")

else:
    st.info("Enter your farming conditions in the sidebar to get crop suggestions.")
