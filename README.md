# AgriAI - Intelligent Agricultural Assistant

AgriAI is a comprehensive AI-driven platform designed to empower farmers and agricultural enthusiasts with data-backed insights. It leverages machine learning to predict crop yields and suggests the most suitable crops for specific environmental conditions.

## 🌟 Key Features

### 1. Crop Yield Prediction
- **Predictive Modeling:** Uses a Random Forest Regressor to estimate crop yield (tons per hectare).
- **Multi-Factor Analysis:** Takes into account Location (State), Season, Year, Area, Annual Rainfall, Fertilizer usage, and Pesticide usage.
- **Top Recommendations:** Automatically identifies and ranks the top 5 crops likely to produce the highest yield for the given conditions.

### 2. Retrieval-Augmented Generation (RAG) Advisory (In Progress)
- **Knowledge Base:** Built from specialized markdown documents containing detailed information on various crops (e.g., Rice, Wheat, Maize, Mango).
- **Semantic Search:** Uses HuggingFace embeddings (`all-MiniLM-L6-v2`) and FAISS for efficient retrieval of agricultural advisory content.
- **Contextual Insights:** Designed to provide specific cultivation tips and best practices for selected crops.

### 3. Interactive Dashboard
- **User-Friendly Interface:** Built with Streamlit for real-time interaction.
- **Data Visualization:** Displays yield predictions in interactive tables and metrics.

---

## 🛠️ Technical Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Machine Learning:** [Scikit-learn](https://scikit-learn.org/), [Pandas](https://pandas.pydata.org/), [Joblib](https://joblib.readthedocs.io/)
- **AI/LLM Pipeline:** [LangChain](https://www.langchain.com/), [FAISS](https://github.com/facebookresearch/faiss), [HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Data:** Custom CSV datasets for crop recommendations and yields.

---

## 📂 Project Structure

```text
AgriAI/
├── app.py                  # Main Streamlit application
├── data/
│   ├── Crop_recommendation.csv # Dataset for NPK-based recommendation
│   ├── crop_yield.csv          # Dataset for yield prediction
│   └── crop_info/              # Markdown files for RAG knowledge base
├── models/
│   ├── yield_predictor.pkl     # Trained yield prediction model
│   ├── faiss_index.pkl         # FAISS vector store for RAG
│   ├── scaler.pkl              # Scaler for data normalization
│   └── crop_recommender.pkl    # NPK-based crop classifier
├── src/
│   ├── train_yield_model.py    # Script to train the yield predictor
│   └── build_rag_pipeline.py   # Script to build the FAISS index
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Prepare Models & Index
If models are not present, train them:
```bash
python src/train_yield_model.py
python src/build_rag_pipeline.py
```

### 3. Run the Application
```bash
streamlit run app.py
```

---

## 🔮 Future Roadmap
- [ ] **Full RAG Integration:** Connect the FAISS index to an LLM (OpenAI/Gemini) to provide natural language advice.
- [ ] **Soil-Based Recommendations:** Integrate the `crop_recommender.pkl` to suggest crops based on NPK values and soil pH.
- [ ] **Real-time Weather Integration:** Fetch live weather data to provide dynamic recommendations.
- [ ] **Market Price Prediction:** Add features to estimate potential revenue based on current market trends.

---

## 📄 License
This project is licensed under the MIT License.
