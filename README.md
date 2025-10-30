# Ex.No.6 Development of Python Code Compatible with Multiple AI Tools

# Date:30/10/25
# Register no:212223090008

# Aim: 
Write and implement Python code that integrates with multiple AI tools to automate the task of interacting with APIs, comparing outputs, and generating actionable insights with Multiple AI Tools


### **AI Tools Required:**

1. **OpenAI GPT models** (for natural language understanding and generation)
2. **Hugging Face Transformers** (for text summarization and sentiment analysis)
3. **Google Generative AI / Gemini** (for reasoning and factual response comparison)

---

## **Explanation:**

Artificial Intelligence (AI) tools provide diverse capabilities such as text generation, code assistance, image recognition, and decision-making. However, using multiple AI tools together helps to:

* **Compare outputs** and improve reliability.
* **Automate complex workflows** that require different AI strengths.
* **Generate actionable insights** by combining different AI responses.

In this experiment, Python is used as the integration medium since it provides strong libraries for API handling, automation, and data processing. We will develop a program that connects to **multiple AI tools**, collects outputs, compares them, and produces a final summarized insight.

---

## **Concept and Approach:**

1. **Integration via APIs:**
   Each AI platform (OpenAI, Hugging Face, Gemini, etc.) provides APIs that allow developers to send text or data and receive processed results.

2. **Comparison of Results:**
   By calling different AI models on the same input, we can analyze their responses for consistency, creativity, or accuracy.

3. **Automation of Insights:**
   The code uses logic to select the best output or combine responses into an actionable summary.

---

## **Example 1: Sentiment Analysis using Multiple AI Tools**

This example compares sentiment analysis results from **OpenAI GPT** and **Hugging Face** sentiment models.

```python
# Example 1: Compare sentiment from OpenAI GPT and Hugging Face

from transformers import pipeline
import openai

# Initialize Hugging Face pipeline
sentiment_model = pipeline("sentiment-analysis")

# OpenAI API key setup (replace with your key)
openai.api_key = "your_openai_api_key"

# Input text
text = "I love using AI tools because they make my work easier!"

# Get Hugging Face result
hf_result = sentiment_model(text)[0]['label']

# Get OpenAI GPT result
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "system", "content": "Analyze sentiment of the text."},
              {"role": "user", "content": text}]
)
gpt_result = response.choices[0].message.content

# Print comparison
print("Input:", text)
print("Hugging Face Sentiment:", hf_result)
print("OpenAI GPT Sentiment:", gpt_result)
```

**Analysis:**
Both AI tools interpret emotions differently. Hugging Face provides structured labels like *Positive/Negative*, while GPT gives human-like reasoning (e.g., *“The sentiment is positive because it expresses enthusiasm.”*).
Combining both helps validate emotional tone accurately.

---

## **Example 2: Text Summarization using Hugging Face and Gemini API**

This example generates summaries using **two AI models** and then compares them.

```python
# Example 2: Text Summarization using Multiple AI Tools

from transformers import pipeline
import google.generativeai as genai

# Initialize Hugging Face summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Configure Gemini API key
genai.configure(api_key="your_gemini_api_key")

text = """
Artificial Intelligence is revolutionizing industries by automating decision-making processes, 
improving accuracy, and enabling personalized services. The adoption of AI technologies 
continues to grow in healthcare, education, and manufacturing sectors.
"""

# Summarize using Hugging Face
hf_summary = summarizer(text, max_length=50, min_length=25, do_sample=False)[0]['summary_text']

# Summarize using Google Gemini
gemini_model = genai.GenerativeModel('gemini-pro')
gemini_response = gemini_model.generate_content("Summarize this: " + text)

# Print comparison
print("Original Text:\n", text)
print("\nHugging Face Summary:\n", hf_summary)
print("\nGemini Summary:\n", gemini_response.text)
```

**Analysis:**

* Hugging Face generates concise factual summaries.
* Gemini provides context-aware summaries with a more natural tone.
  Combining both can enhance readability and factual correctness.

---

## **Example 3: AI Insight Generator using Combined Models**

This example integrates multiple tools to **compare outputs and create a final decision**.

```python
# Example 3: Combined AI Insight Generator

import openai
from transformers import pipeline
import google.generativeai as genai

# API configurations
openai.api_key = "your_openai_api_key"
genai.configure(api_key="your_gemini_api_key")

# Input question
query = "What are the potential impacts of AI on healthcare?"

# Step 1: OpenAI response
gpt_response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": query}]
)
gpt_output = gpt_response.choices[0].message.content

# Step 2: Hugging Face summarizer to shorten GPT output
summarizer = pipeline("summarization")
summary = summarizer(gpt_output, max_length=60, min_length=30, do_sample=False)[0]['summary_text']

# Step 3: Gemini validation
gemini_model = genai.GenerativeModel('gemini-pro')
validation = gemini_model.generate_content("Check if this summary is accurate and useful:\n" + summary)

# Step 4: Final output
print("\nOriginal Query:\n", query)
print("\nGPT Detailed Response:\n", gpt_output)
print("\nSummarized Insight (Hugging Face):\n", summary)
print("\nGemini Validation:\n", validation.text)
```

**Analysis:**

* The GPT model provides deep reasoning.
* Hugging Face refines it into a short, focused version.
* Gemini verifies and enhances readability.
  This workflow demonstrates **collaborative AI integration** to produce refined insights.

---

## **Discussion:**

By integrating multiple AI tools:

* Developers can **compare outputs** to ensure accuracy and diversity.
* The workflow becomes **modular**, allowing flexible substitution of models.
* Python serves as a universal bridge due to its vast library ecosystem and API support.

This approach is particularly useful in:

* **Healthcare analytics** (cross-checking AI predictions).
* **Educational chatbots** (multi-source content verification).
* **Research automation** (summarizing and validating literature).

---

## **Conclusion:**

In this experiment, Python was used to integrate and automate interactions among multiple AI tools such as **OpenAI**, **Hugging Face**, and **Google Gemini**.
The developed codes demonstrated:

* API-based communication with different AI systems.
* Comparison of AI-generated outputs.
* Generation of final actionable insights through automation.

---

## **Result:**

The corresponding Python programs were executed successfully.
The experiment demonstrated the ability of Python to integrate multiple AI tools effectively, compare their outputs, and generate meaningful insights automatically.


