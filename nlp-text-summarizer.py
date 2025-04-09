
import sys
print(sys.version)

import networkx as nx
import spacy
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, pipeline

# Specify the model name
model_name = "google/pegasus-xsum"

# Load pretrained tokenizer and model
pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

# Load spaCy model
nlp = spacy.load('en_core_web_lg')

def textrank(text):
    doc = nlp(text)
    graph = nx.Graph()
    for sentence in doc.sents:
        for token in sentence:
            graph.add_node(token.text)
            for child in token.children:
                graph.add_edge(token.text, child.text)
    scores = nx.pagerank(graph)
    return scores

example_text = """Deep learning algorithms
Deep learning algorithms are typically trained on large datasets of labeled data. The algorithms learn to associate features in the data with the correct labels. For example, in an image recognition task, the algorithm might learn to associate certain features in an image (such as the shape of an object or the color of an object) with the correct label (such as "dog" or "cat").

Once a deep learning algorithm has been trained, it can be used to make predictions on new data. For example, a deep learning algorithm that has been trained to recognize images of dogs can be used to identify dogs in new images.

How does deep learning work?
Deep learning works by using artificial neural networks to learn from data. Neural networks are made up of layers of interconnected nodes, and each node is responsible for learning a specific feature of the data.  Building on our previous example with images – in an image recognition network, the first layer of nodes might learn to identify edges, the second layer might learn to identify shapes, and the third layer might learn to identify objects.

As the network learns, the weights on the connections between the nodes are adjusted so that the network can better classify the data. This process is called training, and it can be done using a variety of techniques, such as supervised learning, unsupervised learning, and reinforcement learning.

Once a neural network has been trained, it can be used to make predictions with new data it’s received.

Deep learning vs. machine learning
Both deep learning and machine learning are branches of artificial intelligence, with machine learning being a broader term encompassing various techniques, including deep learning. Both machine learning and deep learning algorithms can be trained on labeled or unlabeled data, depending on the task and algorithm.

Machine learning and deep learning are both applicable to tasks such as image recognition, speech recognition, and natural language processing. However, deep learning often outperforms traditional machine learning in complex pattern recognition tasks like image classification and object detection due to its ability to learn hierarchical representations of data."""

scores = textrank(example_text)
print(scores)

def get_top_sentences(text, n=2):
    doc = nlp(text)
    scores = textrank(text)
    top_sentences = sorted([(sent.text, sum(scores.get(word, 0) for word in sent)) for sent in doc.sents], key=lambda x: x[1], reverse=True)
    return top_sentences[:n]

top_sentences = get_top_sentences(example_text)
for sent, score in top_sentences:
    print(sent)

def get_top_phrases(text, n=10):
    doc = nlp(text)
    scores = textrank(text)
    phrases = []
    for sent in doc.sents:
        for token in sent:
            if token.pos_ == "NOUN" or token.pos_ == "PROPN":
                phrase = token.text
                score = scores.get(phrase, 0)
                phrases.append((phrase, score))
    top_phrases = sorted(phrases, key=lambda x: x[1], reverse=True)
    return top_phrases[:n]

top_phrases = get_top_phrases(example_text)
for phrase, score in top_phrases:
    print(f"{phrase}: {score}")

# Load pretrained tokenizer and model
pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)
pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name)

# Create tokens
tokens = pegasus_tokenizer(example_text, truncation=True, padding="longest", return_tensors="pt")
tokens

# Summarize text
encoded_summary = pegasus_model.generate(**tokens)
decoded_summary = pegasus_tokenizer.decode(encoded_summary[0], skip_special_tokens=True)
print(decoded_summary)

# Define summarization pipeline
summarizer = pipeline("summarization", model=model_name, tokenizer=pegasus_tokenizer, framework="pt")

# Create summary
summary = summarizer(example_text, min_length=30, max_length=150)
summary[0]["summary_text"]
