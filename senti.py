import streamlit as st 
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import re
import speech_recognition as sr
import os

nltk.download('punkt')

# Aspect configuration with strict rules
ASPECT_RULES = {
    "food": {
        "keywords": ["food", "taste", "dish", "meal", "flavor", "cuisine"],
        "negative_triggers": ["bad", "poor", "awful", "tasteless", "bland", 
                             "inedible", "disgusting", "salty", "cold", "mediocre",
                             "could be better", "wasn't impressive", "lacked flavor"]
    },
    "staff": {
        "keywords": ["staff", "service", "waiter", "manager", "server"],
        "negative_triggers": ["rude", "slow", "poor", "horrible", "inattentive",
                             "unhelpful", "ignored", "had to wait", "unprofessional"]
    },
    "ambience": {
        "keywords": ["ambience", "atmosphere", "environment", "restaurant", "clean"],
        "negative_triggers": ["noisy", "too noisy", "uncomfortable", "not clean", "needs work", "shabby"]
    },
    "price": {
        "keywords": ["price", "cost", "value"],
        "negative_triggers": ["overpriced", "too high", "expensive"]
    }
}

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Please speak your review...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            st.success(f"📝 Recognized Text: {text}")
            return text
        except sr.UnknownValueError:
            st.error("❌ Could not understand the audio.")
            return None
        except sr.RequestError:
            st.error("❌ Could not request results, check your internet connection.")
            return None
        except sr.WaitTimeoutError:
            st.warning("⏳ No speech detected, please try again.")
            return None

def add_review_to_csv(df, hotel_name, review_text):
    if df.empty:
        new_bill_number = 1
    else:
        df['bill_number'] = pd.to_numeric(df['bill_number'], errors='coerce')
        new_bill_number = int(df['bill_number'].max()) + 1 if not df.empty else 1
    
    city = df[df['hotel_name'] == hotel_name]['city'].iloc[0] if not df[df['hotel_name'] == hotel_name].empty else "Unknown"
    
    new_review = pd.DataFrame({
        'bill_number': [new_bill_number],
        'hotel_name': [hotel_name],
        'city': [city],
        'review': [review_text]
    })
    
    return pd.concat([df, new_review], ignore_index=True)

def analyze_review(review):
    results = {aspect: False for aspect in ASPECT_RULES}
    sentences = sent_tokenize(str(review))
    
    for sentence in sentences:
        parts = re.split(r' but |, | however | although | though | and ', sentence, flags=re.IGNORECASE)
        for part in parts:
            for aspect, rules in ASPECT_RULES.items():
                kw_present = any(re.search(rf'\b{k}\b', part.lower()) for k in rules["keywords"])
                neg_present = any(re.search(rf'\b{t}\b', part.lower()) for t in rules["negative_triggers"])
                
                if aspect == "food" and "overpriced" in part.lower():
                    results["price"] = True
                elif kw_present and neg_present:
                    results[aspect] = True
    return results

# Streamlit UI
st.title("🏨 Hotel Review Sentiment Analysis")
st.write("Analyze negative sentiments in hotel reviews across different aspects.")

# Initialize or load data
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=['bill_number', 'hotel_name', 'city', 'review'])

# File uploader
uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])
if uploaded_file:
    st.session_state.df = pd.read_csv(uploaded_file)

# Voice review input
st.subheader("🎙️ Add a Voice Review")
hotel_name_voice = st.text_input("Enter the hotel name for the voice review:")
if st.button("Record Voice Review"):
    if hotel_name_voice:
        review_text = recognize_speech()
        if review_text:
            st.session_state.df = add_review_to_csv(st.session_state.df, hotel_name_voice, review_text)
            st.success("✅ Review added successfully!")
    else:
        st.warning("⚠️ Please enter a hotel name before recording.")

# Analysis section
if not st.session_state.df.empty:
    # Perform analysis
    results = {}
    hotel_cities = st.session_state.df.groupby('hotel_name')['city'].first().to_dict()
    
    for hotel in st.session_state.df['hotel_name'].unique():
        hotel_reviews = st.session_state.df[st.session_state.df['hotel_name'] == hotel]
        total_reviews = len(hotel_reviews)
        aspect_counts = {aspect: 0 for aspect in ASPECT_RULES}
        
        for review in hotel_reviews['review']:
            review_result = analyze_review(review)
            for aspect, is_negative in review_result.items():
                if is_negative:
                    aspect_counts[aspect] += 1
        
        percentages = {aspect: (count/total_reviews)*100 for aspect, count in aspect_counts.items()}
        results[hotel] = percentages

    # Display results
    st.subheader("📊 Negative Sentiment Analysis")
    for hotel, aspects in results.items():
        st.write(f"**{hotel} ({hotel_cities.get(hotel, '')})**")
        cols = st.columns(4)
        for i, (aspect, percent) in enumerate(aspects.items()):
            cols[i].metric(label=aspect.capitalize(), value=f"{percent:.0f}%")

    # Visualizations
    st.subheader("📉 Aspect-wise Analysis")
    aspect_order = ["food", "staff", "ambience", "price"]
    colors = ['#FF1493', '#00BFFF', '#9ACD32', '#FFD700']
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, hotel in enumerate(results.keys()):
        if i >= 5:  # Only show first 5 hotels (adjust as needed)
            break
            
        values = [results[hotel][aspect] for aspect in aspect_order]
        axes[i].bar(aspect_order, values, color=colors)
        axes[i].set_title(f"{hotel}\n{hotel_cities.get(hotel, '')}")
        axes[i].set_ylim(0, 100)
        
        for j, val in enumerate(values):
            axes[i].text(j, val + 2, f"{val:.0f}%", ha='center')

    # Total negative sentiment
    axes[5].bar(
        list(results.keys()), 
        [sum(aspects.values())/4 for aspects in results.values()],
        color='#a64ac9'
    )
    axes[5].set_title("Total Negative Sentiment")
    axes[5].set_ylim(0, 100)
    for i, val in enumerate([sum(aspects.values())/4 for aspects in results.values()]):
        axes[5].text(i, val + 2, f"{val:.1f}%", ha='center')

    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("ℹ️ Please upload a CSV file or add voice reviews to begin analysis.")