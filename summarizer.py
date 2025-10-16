import streamlit as st
import spacy
from collections import Counter
from string import punctuation
from heapq import nlargest
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import re
import time
import PyPDF2
import io

# Configure page
st.set_page_config(
    page_title="AI Document Summarizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

# Load abstractive summarization model
@st.cache_resource
def load_abstractive_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

def extractive_summarize(text, num_sentences=3, detail_level="Intermediate"):
    """Extractive summarization using spaCy and TF-IDF"""
    nlp = load_spacy_model()
    doc = nlp(text)
    
    # Filter out stop words and punctuation
    keywords = [token.text.lower() for token in doc if token.text.lower() not in nlp.Defaults.stop_words and token.text not in punctuation]
    word_frequencies = Counter(keywords)
    
    if not word_frequencies:
        return "Unable to generate summary from the provided text."
    
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency

    sentence_scores = {}
    for sent in doc.sents:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores:
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]
    
    if not sentence_scores:
        return "Unable to generate summary from the provided text."
    
    # Adjust num_sentences based on detail_level for extractive
    if detail_level == "Beginner":
        num_sentences = max(1, int(num_sentences * 0.5))
    elif detail_level == "Expert":
        num_sentences = min(len(list(doc.sents)), int(num_sentences * 1.5))

    # Select top sentences
    summary_sentences = nlargest(min(num_sentences, len(sentence_scores)), sentence_scores, key=sentence_scores.get)
    summary = " ".join([sent.text for sent in summary_sentences])
    return summary

def abstractive_summarize(text, detail_level="Intermediate"):
    """Abstractive summarization using BART"""
    try:
        summarizer = load_abstractive_model()
        max_input_length = 1024
        if len(text.split()) > max_input_length:
            words = text.split()
            text = " ".join(words[:max_input_length])
        
        if detail_level == "Beginner":
            min_length = 20
            max_length = 80
        elif detail_level == "Expert":
            min_length = 50
            max_length = 200
        else:
            min_length = 30
            max_length = 130

        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        return f"Error in abstractive summarization: {str(e)}"

def extract_text_from_url(url):
    """Extract text content from a URL"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        for script in soup(["script", "style"]):
            script.decompose()
        
        main_content = None
        content_selectors = [
            "article", "[role=\"main\"]", ".content", ".main-content",
            ".article-content", ".post-content", "#content", "#main"
        ]
        
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.find("body")
        
        if main_content:
            text = main_content.get_text()
            text = re.sub(r"\s+", " ", text).strip()
            title = soup.title.string if soup.title else "Extracted Content"
            return text, title
        else:
            return None, "Could not extract content from the page"
            
    except Exception as e:
        return None, f"Error fetching URL: {str(e)}"

def extract_text_from_pdf(pdf_file):
    """Extract text content from a PDF file"""
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
        return text, pdf_file.name
    except Exception as e:
        return None, f"Error extracting text from PDF: {str(e)}"

def main():
    st.title("🧠 AI Document Summarizer")
    st.markdown("Transform lengthy documents into concise, intelligent summaries using advanced NLP techniques.")
    
    st.sidebar.header("⚙️ Configuration")
    
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["📝 Text Input", "🔗 URL Input", "📄 PDF Upload"]
    )
    
    method = st.sidebar.selectbox(
        "Summarization Method:",
        ["extractive", "abstractive"],
        format_func=lambda x: "🔍 Extractive (Key Sentences)" if x == "extractive" else "🤖 Abstractive (AI Generated)"
    )
    
    detail_level = st.sidebar.selectbox(
        "Summary Detail Level:",
        ["Beginner", "Intermediate", "Expert"]
    )

    if method == "extractive":
        num_sentences = st.sidebar.slider(
            "Number of sentences:",
            min_value=1,
            max_value=10,
            value=3
        )
    else:
        num_sentences = 3
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📄 Input")
        
        text_to_summarize = ""
        title = ""
        
        if input_method == "📝 Text Input":
            text_to_summarize = st.text_area(
                "Enter your text here:",
                height=300,
                placeholder="Paste your document text here..."
            )
            title = "User Input"
            
        elif input_method == "🔗 URL Input":
            url = st.text_input(
                "Enter URL:",
                placeholder="https://example.com/article"
            )
            
            if url and st.button("🔄 Fetch Content"):
                with st.spinner("Fetching content from URL..."):
                    text_to_summarize, title = extract_text_from_url(url)
                    
                if text_to_summarize:
                    st.success(f"✅ Successfully extracted content from: {title}")
                    with st.expander("📖 View extracted content"):
                        st.text_area("Extracted text:", text_to_summarize, height=200, disabled=True)
                else:
                    st.error(f"❌ {title}")
        
        else:
            pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
            if pdf_file:
                with st.spinner("Extracting text from PDF..."):
                    text_to_summarize, title = extract_text_from_pdf(pdf_file)
                
                if text_to_summarize:
                    st.success(f"✅ Successfully extracted content from: {title}")
                    with st.expander("📖 View extracted content"):
                        st.text_area("Extracted text:", text_to_summarize, height=200, disabled=True)
                else:
                    st.error(f"❌ {title}")

        if text_to_summarize and st.button("✨ Generate Summary", type="primary"):
            if len(text_to_summarize.strip()) < 50:
                st.warning("⚠️ Please provide more text for better summarization results.")
            else:
                with st.spinner(f"Generating {method} summary..."):
                    start_time = time.time()
                    
                    if method == "extractive":
                        summary = extractive_summarize(text_to_summarize, num_sentences, detail_level)
                    else:
                        summary = abstractive_summarize(text_to_summarize, detail_level)
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                
                st.header("📊 Summary Results")
                
                col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                
                with col_metric1:
                    st.metric("Original Length", f"{len(text_to_summarize):,} chars")
                
                with col_metric2:
                    st.metric("Summary Length", f"{len(summary):,} chars")
                
                with col_metric3:
                    compression_ratio = (len(summary) / len(text_to_summarize)) * 100
                    st.metric("Compression", f"{compression_ratio:.1f}%")
                
                with col_metric4:
                    st.metric("Processing Time", f"{processing_time:.2f}s")
                
                st.subheader("📝 Generated Summary")
                st.info(summary)
                
                st.subheader("📋 Copy Summary")
                st.code(summary, language=None)
    
    with col2:
        st.header("ℹ️ About")
        
        st.markdown("""
        ### 🔍 Extractive Summarization
        - Selects key sentences from the original text
        - Preserves original wording
        - Fast and reliable
        - Good for factual content
        
        ### 🤖 Abstractive Summarization
        - Generates new sentences
        - More human-like summaries
        - Takes longer to process
        - Better for creative content
        
        ### 📊 Features
        - ✅ Text input support
        - ✅ URL content extraction
        - ✅ PDF file upload
        - ✅ Real-time processing
        - ✅ Compression metrics
        - ✅ Easy copy functionality
        """)
        
        st.markdown("---")
        st.markdown("**💡 Tips:**")
        st.markdown("""
        - For best results, use texts with at least 100 words
        - URLs should point to articles or blog posts
        - Extractive method is faster for long documents
        - Abstractive method provides more coherent summaries
        """)

if __name__ == "__main__":
    main()
