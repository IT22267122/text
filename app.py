import streamlit as st
from test import summarize_large_text, analyze_sentiment, extract_keywords_from_large_text, topic_modeling_on_large_texts

# Streamlit app
st.title("Text Summarization and Analysis")

# Add an image
st.image("images.jpg", use_column_width=True)

# Text input
text = st.text_area("Enter your text here:", height=300)

# Slider for summary length selection
summary_length = st.slider("Select Summary Length:", 1, 3, 2)
length_category = {1: "Short", 2: "Medium", 3: "Long"}[summary_length]
st.write(f"Summary Length: {length_category}")

# Original text length (display word count instead of character count)
if text:
    word_count = len(text.split())  # Count words in the original text
    st.write(f"Original Text Length: {word_count} words")

# Create three columns for buttons
col1, col2, col3, col4 = st.columns([1,1,1,1])

if col1.button("Summarize"):
    if text:
        if word_count < 60:
            st.warning("Text is too short to summarize. Must be at least 60 words.")
        else:
            # Restrict medium summary if word count is less than 120
            if summary_length == 2 and word_count < 120:
                st.warning("Text is too short for a medium summary. Must be at least 120 words.")
            # Restrict long summary if word count is less than 150
            elif summary_length == 3 and word_count < 180:
                st.warning("Text is too short for a long summary. Must be at least 180 words.")
            else:
                # Map summary_length to summary type
                summary_type = {1: "short", 2: "medium", 3: "long"}[summary_length]
                summary = summarize_large_text(text, summary_type=summary_type)  # Pass the summary_type
                summary_word_count = len(summary.split())  # Count words in the summarized text

                st.subheader("Summarized Text")
                st.write(summary)
                st.write(f"Summarized Text Length: {summary_word_count} words")  # Display word count instead of character count

if col2.button("Analyze Sentiment"):
    if text:
        sentiment = analyze_sentiment(text)
        st.subheader("Sentiment")
        st.write(f"The sentiment of the text is: {sentiment}")

if col3.button("Extract Keywords"):
    if text:
        keywords = extract_keywords_from_large_text(text)
        st.subheader("Keywords")
        st.write(", ".join(keywords))

if col4.button("Topic Modeling"):
    if text:
        topics = topic_modeling_on_large_texts([text])
        st.subheader("Topics")
        for topic in topics:
            st.write(topic)
