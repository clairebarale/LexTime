import pandas as pd
import spacy
import re

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")  # Use a larger model if needed (e.g., en_core_web_trf)

files = [
    "prompting/dataset_implicit_events/qa_tr.csv", 
    "tracie_prompting/tracie_questions.csv"
]

# 1General Time Adverbs
general_time_adverbs = [
    "now", "then", "before", "after", "later", "soon", "earlier", "previously",
    "recently", "already", "immediately", "eventually", "finally", "formerly", "subsequently"
]

# Frequency Adverbs
frequency_adverbs = [
    "always", "usually", "often", "sometimes", "occasionally", "rarely", "seldom", "never",
    "frequently", "constantly", "continuously", "periodically", "infrequently", "repeatedly"
]

# Relative Temporal Expressions
relative_temporal_expressions = [
    "yesterday", "today", "tomorrow", "tonight", "this morning", "this afternoon",
    "this evening", "this week", "this month", "this year", "last night", "last week", 
    "last month", "last year", "next week", "next month", "next year",
    "the day before yesterday", "the day after tomorrow"
]

# Specific Time Expressions
specific_time_expressions = [
    "at noon", "at midnight", "at dawn", "at dusk", "at sunrise", "at sunset",
    "at 3 PM", "at 5:30 AM"
]

# Calendar-Based Expressions
calendar_based_expressions = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "this Monday", "next Friday", "last Sunday", "January", "February", "March", "April",
    "May", "June", "July", "August", "September", "October", "November", "December",
    "this January", "next December", "last March", "spring", "summer", "autumn", "fall", "winter"
]

# Durations
durations = [
    "for a moment", "for a while", "for an hour", "for a day", "for a week", "for a month", 
    "for a year", "for decades", "for centuries", "since morning", "since last night", 
    "since yesterday", "since last week", "since last month", "since the beginning", 
    "since 1990", "since childhood"
]

# Time Conjunctions & Prepositions
time_conjunctions_prepositions = [
    "until", "till", "since", "while", "when", "whenever", "as soon as", "as long as",
    "afterwards", "meanwhile", "in the meantime", "up to now", "up until now"
]

# Temporal Comparisons
temporal_comparisons = [
    "sooner", "later", "earliest", "latest", "the other day", "one day", "someday", "some time"
]

# Hypothetical & Speculative Temporal Expressions
hypothetical_temporal_expressions = [
    "eventually", "someday", "in the near future", "in the distant future", "not yet", "yet",
    "ever", "once upon a time"
]

# Historical & Long-Term Expressions
historical_temporal_expressions = [
    "in ancient times", "in the past", "in the future", "in those days", "in modern times",
    "back in the day", "ages ago", "eons ago"
]

# Time-Related Idiomatic Expressions
time_related_idioms = [
    "in no time", "all of a sudden", "out of the blue", "once in a blue moon", "from time to time",
    "time after time", "over time", "in due time", "at the last minute", "at the right moment",
    "for the time being", "at the moment", "on time", "just in time"
]

def extract_trees_and_modals(text):
    """
    Extracts dependency trees from text and collects modal verbs and negations.
    """
    doc = nlp(text)
    
    # Find root nodes (heads of independent trees)
    roots = [token for token in doc if token.dep_ == "ROOT"]
    
    # Collect words belonging to each root tree
    trees_data = []
    
    for root in roots:
        tree_tokens = set()

        def collect_subtree(token):
            """Recursively collect all tokens in the subtree of a given token."""
            if token not in tree_tokens:
                tree_tokens.add(token)
                for child in token.children:
                    collect_subtree(child)
        
        collect_subtree(root)
        
        # Sort tokens based on sentence order
        sorted_tokens = sorted(tree_tokens, key=lambda x: x.i)
        tree_text = " ".join([t.text for t in sorted_tokens])

        # Clean up the tree text (useful for the tracie data)
        tree_text = re.sub(r'"', '', tree_text)
        tree_text = re.sub(r'^\s+', '', tree_text)
        tree_text = re.sub(r'\s([.,;:!?])', r'\1', tree_text)
        tree_text = re.sub(r'\t', '', tree_text)
        
        # Extract modal verbs and negations
        modal_verbs = [token.text for token in sorted_tokens if token.pos_ == "AUX" and token.dep_ == "aux"]
        negations = [token.text for token in sorted_tokens if token.dep_ == "neg"]

        # Prepare empty columns for properties
        properties = {
            "tree": tree_text,
            "modal_verbs": ", ".join(modal_verbs) if modal_verbs else "",
            "negations": ", ".join(negations) if negations else "",
            "nouns": "",
            "verbs": "",
            "passive_voice_events": "",
            "adjectives": "",
            "nominalized_events": "",
            "conditional_events": "",
            "subordinate_clause_events": "",
            "evidential_marker_events": "",
            "counterfactual_events": "",
            "direct_speech": "",
            "indirect_speech": "",
            "legal_citations": "",
            "citation_in_speech": "",
            "paraphrased_summaries": "",
            "dates": "",
            "tense": "",
            "aspectual_markers": "",
            "future_speculative_events": ""
        }

        trees_data.append(properties)
    
    return pd.DataFrame(trees_data)

def extract_temporal_properties(text):
    """
    Extracts all temporal expressions from a paragraph and categorizes them.
    """
    doc = nlp(text)

    temporal_data = {
        "paragraph": text,
        "general_time_adverbs": [],
        "frequency_adverbs": [],
        "relative_temporal_expressions": [],
        "specific_time_expressions": [],
        "calendar_based_expressions": [],
        "durations": [],
        "time_conjunctions_prepositions": [],
        "temporal_comparisons": [],
        "hypothetical_temporal_expressions": [],
        "historical_temporal_expressions": [],
        "time_related_idioms": []
    }

    for token in doc:
        word = token.text.lower()
        if word in general_time_adverbs:
            temporal_data["general_time_adverbs"].append(token.text)
        elif word in frequency_adverbs:
            temporal_data["frequency_adverbs"].append(token.text)
        elif word in relative_temporal_expressions:
            temporal_data["relative_temporal_expressions"].append(token.text)
        elif word in specific_time_expressions:
            temporal_data["specific_time_expressions"].append(token.text)
        elif word in calendar_based_expressions:
            temporal_data["calendar_based_expressions"].append(token.text)
        elif word in durations:
            temporal_data["durations"].append(token.text)
        elif word in time_conjunctions_prepositions:
            temporal_data["time_conjunctions_prepositions"].append(token.text)
        elif word in temporal_comparisons:
            temporal_data["temporal_comparisons"].append(token.text)
        elif word in hypothetical_temporal_expressions:
            temporal_data["hypothetical_temporal_expressions"].append(token.text)
        elif word in historical_temporal_expressions:
            temporal_data["historical_temporal_expressions"].append(token.text)
        elif word in time_related_idioms:
            temporal_data["time_related_idioms"].append(token.text)

    # Convert lists to comma-separated strings
    for key in temporal_data:
        if isinstance(temporal_data[key], list):
            temporal_data[key] = ", ".join(temporal_data[key]) if temporal_data[key] else ""

    return pd.DataFrame([temporal_data])

# Load dataset
mydf = pd.read_csv(files[1], usecols=["paragraph"])

# Process first dataframe (trees, modals, negations)
df_trees = pd.concat([extract_trees_and_modals(row["paragraph"]) for _, row in mydf.iterrows()], ignore_index=True)
# If col tree is empty, remove the row
df_trees = df_trees[df_trees["tree"] != ""]

# Process second dataframe (temporal expressions)
df_temporal = pd.concat([extract_temporal_properties(row["paragraph"]) for _, row in mydf.iterrows()], ignore_index=True)

# Save the dataframes
output_folder = "language_analysis/"
df_trees.to_csv(output_folder + "tracie_trees_modals_negations.csv", index=False)
df_temporal.to_csv(output_folder + "tracie_temporal_expressions.csv", index=False)
