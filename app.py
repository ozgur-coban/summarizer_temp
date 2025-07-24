# Cell 1: Install all necessary libraries
# !pip install -q transformers sentencepiece torch accelerate
# !pip install -q spacy
# In a new cell
# !pip install -q rouge_score evaluate
# !python -m spacy download en_core_web_sm
# !pip install -q yake
# Install spaCy and its small English model
# The Definitive "Ensemble Generation" Script - Scaffolding

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import time
import spacy
from collections import Counter
import re
import math
import os

os.system("python -m spacy download en_core_web_sm")

# ==============================================================================
# --- ⚙️ YOUR CONTROL PANEL ⚙️ ---
# ==============================================================================
# The path to your unzipped, fine-tuned model folder

# ==============================================================================


# Helper function to generate a single summary (NOW CORRECTED)
def generate_single_summary(text, summarizer_pipeline, generation_params):
    """
    Helper to generate a single summary.
    Includes robust chunking logic for long articles.
    """
    tokenizer = summarizer_pipeline.tokenizer
    model_max_length = summarizer_pipeline.model.config.max_position_embeddings

    token_ids = tokenizer(text, truncation=False, padding=False)["input_ids"]
    total_tokens = len(token_ids)

    if total_tokens <= model_max_length:
        result = summarizer_pipeline(text, **generation_params)
        return result[0]["summary_text"] if result else None
    else:
        print(f"    - (Input text is {total_tokens} tokens long. Applying chunking...)")
        max_chunk_tokens = 900
        overlap = 100

        # Calculate the total number of chunks that will be processed
        num_chunks = math.ceil(total_tokens / (max_chunk_tokens - overlap))
        chunk_summaries = []

        for i in range(0, total_tokens, max_chunk_tokens - overlap):
            chunk_token_ids = token_ids[i : i + max_chunk_tokens]
            chunk_text = tokenizer.decode(chunk_token_ids, skip_special_tokens=True)

            # --- FIX: Now using num_chunks in the print statement for clarity ---
            print(
                f"      - Summarizing chunk {len(chunk_summaries) + 1}/{num_chunks}..."
            )
            # -----------------------------------------------------------------

            chunk_params = generation_params.copy()
            chunk_params["max_length"] = 140
            chunk_params["min_length"] = 40

            result = summarizer_pipeline(chunk_text, **chunk_params)
            if result:
                chunk_summaries.append(result[0]["summary_text"])

        return "\n".join(chunk_summaries)


# Helper function to generate multiple summaries (NOW CORRECTED)
def generate_multiple_summaries(text, summarizer_pipeline, generation_params):
    """
    Helper to generate a list of summaries.
    Includes robust chunking logic for long articles.
    """
    tokenizer = summarizer_pipeline.tokenizer
    model_max_length = summarizer_pipeline.model.config.max_position_embeddings

    token_ids = tokenizer(text, truncation=False, padding=False)["input_ids"]
    total_tokens = len(token_ids)

    if total_tokens <= model_max_length:
        results = summarizer_pipeline(text, **generation_params)
        return [res["summary_text"] for res in results] if results else []
    else:
        print(
            f"    - (Input text is {total_tokens} tokens long. Applying chunking to create an intermediate summary first...)"
        )

        # 1. Create the intermediate summary using a reliable beam search
        intermediate_summary = generate_single_summary(
            text,
            summarizer_pipeline,
            {"max_length": 150, "min_length": 30, "num_beams": 4},
        )

        if not intermediate_summary:
            return []

        # 2. Now, generate multiple creative options from the intermediate summary
        print("    - (Generating multiple options from the intermediate summary...)")
        results = summarizer_pipeline(intermediate_summary, **generation_params)
        return [res["summary_text"] for res in results] if results else []


def module_1_baseline_generation(input_text, pipeline_beam, pipeline_sample):
    """
    Generates one factual baseline summary (Beam Search) and one creative
    baseline summary (Sampling) from the plain input text.
    """
    print("\n--> Running Module 1: Baseline Generation...")
    summaries = []

    # --- A) The Beam Search Baseline ---
    # Define the parameters for the factual, beam search summary.
    beam_search_params = {
        "max_length": 150,
        "min_length": 30,
        "do_sample": False,
        "num_beams": 4,
        "early_stopping": True,
        "repetition_penalty": 1.2,
    }

    # Generate the single best summary using the beam search pipeline.
    summary_beam = generate_single_summary(
        input_text, pipeline_beam, beam_search_params
    )

    if summary_beam:
        summaries.append(summary_beam)
        print("    - 'Beam Search Baseline' candidate generated.")

    # --- B) The Sampling Baseline ---
    # Define the parameters for the creative, sampling summary.
    sampling_params = {
        "max_length": 150,
        "min_length": 30,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.8,
    }

    # Generate the single summary using the sampling pipeline.
    summary_sample = generate_single_summary(
        input_text, pipeline_sample, sampling_params
    )

    if summary_sample:
        summaries.append(summary_sample)
        print("    - 'Sampling Baseline' candidate generated.")

    return summaries


def module_2_simple_prompters(input_text, pipeline_beam):
    """
    Generates summaries guided by direct, simple instructions using the
    "Simple Command" and "Chain of Thought" methods.
    """
    print("\n--> Running Module 2: Simple Prompters...")
    summaries = []

    # --- A) The "Simple Command" Prompt ---
    # Define the simple, direct instruction.
    prompt_simple = f"Summarize this article in one concise sentence:\n\n{input_text}"

    # Define the generation parameters based on your successful test.
    simple_command_params = {
        "max_length": 60,
        "min_length": 10,
        "num_beams": 4,  # Using beam search for factual output
        "early_stopping": True,
    }

    # Generate the summary using the beam search pipeline.
    summary_simple = generate_single_summary(
        prompt_simple, pipeline_beam, simple_command_params
    )

    if summary_simple:
        summaries.append(summary_simple)
        print("    - 'Simple Command' candidate generated.")

    # --- B) The "Chain of Thought" Prompt ---
    # Define the reasoning-based instruction.
    prompt_cot = f"First, identify the key entities and the main event in the following article. Then, based on that analysis, provide a concise, abstractive summary.\n\nArticle:\n{input_text}"

    # Define generation parameters based on your successful test.
    cot_params = {
        "max_length": 150,
        "min_length": 35,  # Using 35 as per your test script
        "num_beams": 4,  # Using beam search
        "repetition_penalty": 1.2,
        "early_stopping": True,
    }

    # Generate the summary using the beam search pipeline.
    summary_cot = generate_single_summary(prompt_cot, pipeline_beam, cot_params)

    if summary_cot:
        summaries.append(summary_cot)
        print("    - 'Chain of Thought' candidate generated.")

    return summaries


def module_3_metadata_guided(input_text, guiding_tags, pipeline_beam, pipeline_sample):
    """
    Generates summaries that are explicitly focused on pre-identified key topics
    provided by the user.
    """
    print("\n--> Running Module 3: Metadata-Guided Summarizers...")
    summaries = []

    # This module will only run if tags are provided.
    if not guiding_tags:
        print("    - Skipping module: No GUIDING_TAGS provided.")
        return summaries

    # Create the guided prompt from the human-provided tags.
    tags_str = (
        ", ".join(guiding_tags) if isinstance(guiding_tags, list) else guiding_tags
    )
    prompt_tags = f"""
    Summarize the following news article. It is important that the summary focuses on the key topics of: {tags_str}.

    Article:
    {input_text}
    """

    # --- A) The Beam Search Guided Summary ---
    # Define the parameters for the factual, guided summary.
    beam_guided_params = {
        "max_length": 150,
        "min_length": 30,
        "do_sample": False,
        "num_beams": 4,
        "early_stopping": True,
    }

    summary_beam_guided = generate_single_summary(
        prompt_tags, pipeline_beam, beam_guided_params
    )

    if summary_beam_guided:
        summaries.append(summary_beam_guided)
        print("    - 'Tag-Guided Beam Search' candidate generated.")

    # --- B) The Sampling Guided Summary ---
    # Define the parameters for the creative, guided summary.
    sampling_guided_params = {
        "max_length": 150,
        "min_length": 30,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
    }

    summary_sample_guided = generate_single_summary(
        prompt_tags, pipeline_sample, sampling_guided_params
    )

    if summary_sample_guided:
        summaries.append(summary_sample_guided)
        print("    - 'Tag-Guided Sampling' candidate generated.")

    return summaries


def module_4_angled_summarizers(
    input_text, guiding_tags, pipeline_beam, pipeline_sample
):
    """
    Generates a general overview (Beam + Sampling) and specific deep-dive
    summaries for each provided tag.
    """
    print("\n--> Running Module 4: Angled Summarizers...")
    summaries = []

    # This module will only run if tags are provided.
    if not guiding_tags:
        print("    - Skipping module: No GUIDING_TAGS provided.")
        return summaries

    # --- A) Generate the "General Summaries" from this method ---
    # These are generated without a complex prompt.
    general_beam_params = {
        "max_length": 150,
        "min_length": 30,
        "do_sample": False,
        "num_beams": 4,
        "early_stopping": True,
    }
    summary_general_beam = generate_single_summary(
        input_text, pipeline_beam, general_beam_params
    )
    if summary_general_beam:
        summaries.append(summary_general_beam)
        print("    - 'Angled - General Beam' candidate generated.")

    general_sample_params = {
        "max_length": 150,
        "min_length": 30,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
    }
    summary_general_sample = generate_single_summary(
        input_text, pipeline_sample, general_sample_params
    )
    if summary_general_sample:
        summaries.append(summary_general_sample)
        print("    - 'Angled - General Sampling' candidate generated.")

    # --- B) Generate the specific "Angle-Focused" summaries ---
    # Loop through each human-provided tag.
    for tag in guiding_tags:
        print(f"    - Generating angled summary for tag: '{tag}'...")
        # Create a prompt that forces the model to focus on this one tag.
        angle_prompt = f"Summarize the following article with a specific focus on '{tag}'.\n\nArticle:\n{input_text}"

        # Use reliable Beam Search for these factual, focused summaries.
        angle_params = {
            "max_length": 80,
            "min_length": 15,
            "do_sample": False,
            "num_beams": 4,
        }

        summary_angle = generate_single_summary(
            angle_prompt, pipeline_beam, angle_params
        )

        if summary_angle:
            summaries.append(summary_angle)
            print(f"      - Candidate for '{tag}' generated.")

    return summaries


# def module_5_multiple_sampling_runs(input_text, num_options, pipeline_sample):
#     """
#     Generates a "menu" of diverse, creative options using sampling.
#     """
#     print("\n--> Running Module 5: Multiple Sampling Runs...")

#     # Define the parameters for generating multiple, high-quality samples.
#     multi_sample_params = {
#         "max_length": 150,
#         "min_length": 30,
#         "do_sample": True,
#         "num_return_sequences": num_options,  # Use the value from the control panel
#         "top_k": 50,
#         "top_p": 0.95,
#         "temperature": 0.9,
#         "repetition_penalty": 1.2,
#         "no_repeat_ngram_size": 3,
#     }

#     # Generate a list of creative summaries using the sampling pipeline.
#     # We use the generate_multiple_summaries helper because we expect a list.
#     summaries = generate_multiple_summaries(
#         input_text, pipeline_sample, multi_sample_params
#     )

#     if summaries:
#         print(f"    - {len(summaries)} 'Creative Sampling Options' generated.")


#     return summaries
# --- This is the corrected function. Paste this over your old Module 5. ---


def module_5_beam_and_sampling_options(
    input_text, num_sampling_options, pipeline_beam, pipeline_sample
):
    """
    Generates one factual baseline (Beam Search) and a "menu" of diverse,
    creative options (Sampling), using a loop for true isolation and correctness.
    """
    print("\n--> Running Module 5: Beam + Multiple Sampling Options...")
    summaries = []

    # --- A) The Beam Search Option (Unchanged) ---
    print("    - Generating 'Beam Search Option' candidate...")
    beam_search_params = {
        "max_length": 150,
        "min_length": 30,
        "do_sample": False,
        "num_beams": 4,
        "early_stopping": True,
        "repetition_penalty": 1.2,
    }
    # This part still uses the original, reliable helper function.
    summary_beam = generate_single_summary(
        input_text, pipeline_beam, beam_search_params
    )
    if summary_beam:
        summaries.append(summary_beam)
        print("      - Candidate generated.")

    # --- B) The Creative Options (via Sampling in a Loop) ---
    print(f"    - Generating {num_sampling_options} 'Creative Sampling Options'...")

    # --- THIS IS THE FIX ---
    # Define parameters for generating ONE high-quality sample at a time.
    sampling_params_single = {
        "max_length": 150,
        "min_length": 30,
        "do_sample": True,
        "num_return_sequences": 1,  # Generate one at a time to avoid conflicts
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.9,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
    }

    # This loop is the definitive fix, inspired by your other working scripts.
    sample_summaries = []
    for i in range(num_sampling_options):
        print(f"      - Generating creative option #{i + 1}...")
        # Each call is a new, independent run of the sampling algorithm.
        # It uses the dedicated pipeline_sample for true isolation.
        summary_sample = generate_single_summary(
            input_text, pipeline_sample, sampling_params_single
        )
        if summary_sample:
            sample_summaries.append(summary_sample)

    if sample_summaries:
        summaries.extend(sample_summaries)
        print(f"      - {len(sample_summaries)} creative candidates generated.")
    # --- END OF FIX ---

    return summaries


def module_6_auto_keyword_guided_yake(input_text, pipeline_beam, pipeline_sample):
    """
    Generates summaries guided by keywords automatically extracted by the YAKE library.
    """
    print("\n--> Running Module 6: Automated Keyword-Guided Summarizers (YAKE)...")
    summaries = []

    try:
        import yake
    except ImportError:
        print(
            "    - Skipping module: 'yake' library not installed. Please run '!pip install -q yake'."
        )
        return summaries

    # --- A) Automatically Extract Key Topics using YAKE ---
    print("    - Extracting key topics with YAKE...")
    kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=5)
    keywords_yake = [kw for kw, score in kw_extractor.extract_keywords(input_text)]
    print(f"      - Identified Keywords: {keywords_yake}")

    # --- B) Create the dynamically guided prompt ---
    if not keywords_yake:
        print("    - Skipping generation: YAKE extracted no keywords from the text.")
        return summaries

    tags_str_yake = ", ".join(keywords_yake)
    prompt_yake = f"""
    Summarize the following news article. It is important that the summary focuses on the primary topics of: {tags_str_yake}. Focus on the main event described.

    Article:
    {input_text}
    """

    # --- C) Run the prompt through both Beam Search and Sampling pipelines ---

    # Beam Search version
    yake_beam_params = {
        "max_length": 150,
        "min_length": 30,
        "do_sample": False,
        "num_beams": 4,
    }
    summary_yake_beam = generate_single_summary(
        prompt_yake, pipeline_beam, yake_beam_params
    )
    if summary_yake_beam:
        summaries.append(summary_yake_beam)
        print("    - 'YAKE-Guided Beam Search' candidate generated.")

    # Sampling version
    yake_sample_params = {
        "max_length": 150,
        "min_length": 30,
        "do_sample": True,
        "top_k": 50,
    }
    summary_yake_sample = generate_single_summary(
        prompt_yake, pipeline_sample, yake_sample_params
    )
    if summary_yake_sample:
        summaries.append(summary_yake_sample)
        print("    - 'YAKE-Guided Sampling' candidate generated.")

    return summaries


def module_7_keyword_reranker_simple(
    input_text, num_candidates, pipeline_sample, nlp_ner
):
    """
    Generates a pool of candidates and selects the best one based on a
    simple count of how many key topics are covered.
    """
    print("\n--> Running Module 7: Keyword-Based Reranker (Simple Count)...")

    # --- Step 1: Extract keywords using NER for scoring ---
    print("    - Extracting key topics with spaCy NER...")
    doc = nlp_ner(input_text)
    target_labels = ["ORG", "PERSON", "PRODUCT", "EVENT", "GPE"]
    entities = [ent.text.lower() for ent in doc.ents if ent.label_ in target_labels]
    keywords_for_reranking = [item for item, count in Counter(entities).most_common(5)]
    print(f"      - Identified Keywords for Scoring: {keywords_for_reranking}")

    # --- Step 2: Generate a pool of diverse candidates ---
    print(f"    - Generating {num_candidates} candidates for internal reranking...")
    rerank_gen_params = {
        "max_length": 150,
        "min_length": 30,
        "do_sample": True,
        "num_return_sequences": 1,  # Generate one at a time in a loop
    }

    internal_candidates = []
    for _ in range(num_candidates):
        summary = generate_single_summary(
            input_text, pipeline_sample, rerank_gen_params
        )
        if summary:
            internal_candidates.append(summary)

    # --- Step 3: Score, rerank, and select the best summary ---
    if not internal_candidates:
        print("    - Skipping: No internal candidates were generated for reranking.")
        return []  # Return an empty list if no candidates were made

    ranked_candidates = []
    for candidate in internal_candidates:
        # The score is the number of unique keywords present in the summary.
        score = sum(1 for kw in set(keywords_for_reranking) if kw in candidate.lower())
        ranked_candidates.append({"summary": candidate, "score": score})

    # Sort to find the best candidate
    ranked_candidates.sort(key=lambda x: x["score"], reverse=True)

    # We return ONLY the single best summary from this entire process.
    best_summary = ranked_candidates[0]["summary"]
    print("    - 'Keyword Reranker' best candidate selected.")

    return [best_summary]


def module_8_weighted_keyword_reranker(
    input_text, num_candidates, pipeline_sample, nlp_ner
):
    """
    A more advanced reranker that scores candidates based on topic importance (frequency).
    This is a self-contained module with the CORRECT scoring logic.
    """
    print("\n--> Running Module 8: Weighted Keyword Reranker...")

    # --- Step 1: Extract entities and calculate their importance weights ---
    print("    - Extracting key topics and calculating importance weights...")
    doc = nlp_ner(input_text)
    target_labels = ["ORG", "PERSON", "PRODUCT", "EVENT", "GPE"]
    all_entities = [ent.text.lower() for ent in doc.ents if ent.label_ in target_labels]
    keyword_weights = Counter(all_entities)
    print(f"      - Identified Topic Weights: {keyword_weights}")

    # --- Step 2: Generate a fresh pool of diverse candidates ---
    print(f"    - Generating {num_candidates} new candidates for weighted reranking...")
    rerank_gen_params = {
        "max_length": 150,
        "min_length": 30,
        "do_sample": True,
        "num_return_sequences": 1,
    }
    internal_candidates = []
    for _ in range(num_candidates):
        summary = generate_single_summary(
            input_text, pipeline_sample, rerank_gen_params
        )
        if summary:
            internal_candidates.append(summary)

    # --- Step 3: Score, rerank, and select the best summary using weights ---
    if not internal_candidates:
        print("    - Skipping: No candidates available for weighted reranking.")
        return []

    ranked_candidates_weighted = []
    for candidate in internal_candidates:
        # --- THIS IS THE CORRECT, SUPERIOR SCORING LOGIC ---
        score = 0
        found_keywords = set()
        # Iterate through the full dictionary of weights, which may contain multi-word keys
        for keyword, weight in keyword_weights.items():
            # Check if the keyword (e.g., "tayfun block-4") exists as a substring
            if keyword in candidate.lower() and keyword not in found_keywords:
                score += weight  # Add the keyword's importance score
                found_keywords.add(keyword)
        # ----------------------------------------------------
        ranked_candidates_weighted.append({"summary": candidate, "score": score})

    ranked_candidates_weighted.sort(key=lambda x: x["score"], reverse=True)
    best_summary_weighted = ranked_candidates_weighted[0]["summary"]
    print("    - 'Weighted Reranker' best candidate selected.")

    return [best_summary_weighted]


# --- IMPORTANT: Paste the article you want to summarize here ---

# print("--- Starting Ensemble Generation Process ---")
# start_time_total = time.time()
# try:
#     # ======================================================================
#     # --- STAGE 1: SETUP AND INITIALIZATION ---
#     # ======================================================================
#     print("\nStep 1: Loading all required models...")
#     device = 0 if torch.cuda.is_available() else -1

#     tokenizer = AutoTokenizer.from_pretrained(f"./{MODEL_PATH}")
#     model = AutoModelForSeq2SeqLM.from_pretrained(f"./{MODEL_PATH}")

#     # Create isolated pipelines for different strategies
#     pipeline_beam = pipeline(
#         "summarization", model=model, tokenizer=tokenizer, device=device
#     )
#     pipeline_sample = pipeline(
#         "summarization", model=model, tokenizer=tokenizer, device=device
#     )

#     # Load the NER model for keyword extraction
#     nlp_ner = spacy.load("en_core_web_sm")

#     print("✅ All models loaded successfully.")

#     # ======================================================================
#     # --- STAGE 2: GENERATION MODULES ---
#     # ======================================================================
#     print("\nStep 2: Generating candidate summaries from modules...")
#     labeled_candidates = {}

#     module_1_results = module_1_baseline_generation(
#         INPUT_TEXT, pipeline_beam, pipeline_sample
#     )
#     for summary in module_1_results:
#         if summary not in labeled_candidates:
#             labeled_candidates[summary] = "Module 1: Baseline Generation"

#     module_2_results = module_2_simple_prompters(INPUT_TEXT, pipeline_beam)
#     for summary in module_2_results:
#         if summary not in labeled_candidates:
#             labeled_candidates[summary] = "Module 2: Simple Prompters"
#     module_3_results = module_3_metadata_guided(
#         INPUT_TEXT, GUIDING_TAGS, pipeline_beam, pipeline_sample
#     )
#     for summary in module_3_results:
#         if summary not in labeled_candidates:
#             labeled_candidates[summary] = "Module 3: Metadata-Guided Summarizers"
#     module_4_results = module_4_angled_summarizers(
#         INPUT_TEXT, GUIDING_TAGS, pipeline_beam, pipeline_sample
#     )
#     for summary in module_4_results:
#         if summary not in labeled_candidates:
#             labeled_candidates[summary] = "Module 4: Angled Summarizers"
#     module_5_results = module_5_beam_and_sampling_options(
#         INPUT_TEXT, NUM_Candidates_FOR_RERANKING, pipeline_beam, pipeline_sample
#     )
#     for summary in module_5_results:
#         if summary not in labeled_candidates:
#             labeled_candidates[summary] = "Module 5: Beam + Multiple Sampling Options"
#     module_6_results = module_6_auto_keyword_guided_yake(
#         INPUT_TEXT, pipeline_beam, pipeline_sample
#     )
#     for summary in module_6_results:
#         if summary not in labeled_candidates:
#             labeled_candidates[summary] = (
#                 "Module 6: Automated Keyword-Guided Summarizers (YAKE)"
#             )
#     module_7_results = module_7_keyword_reranker_simple(
#         INPUT_TEXT, NUM_Candidates_FOR_RERANKING, pipeline_sample, nlp_ner
#     )
#     for summary in module_7_results:
#         if summary not in labeled_candidates:
#             labeled_candidates[summary] = (
#                 "Module 7: Keyword-Based Reranker (Simple Count)"
#             )
#     module_8_results = module_8_weighted_keyword_reranker(
#         INPUT_TEXT, NUM_Candidates_FOR_RERANKING, pipeline_sample, nlp_ner
#     )
#     for summary in module_8_results:
#         if summary not in labeled_candidates:
#             labeled_candidates[summary] = "Module 8: Weighted Keyword Reranker"

#     # ======================================================================
#     # --- STAGE 3: COLLECTION AND FINAL OUTPUT ---
#     # ======================================================================
#     # Remove any exact duplicate summaries that may have been generated
#     end_time_total = time.time()

#     print("\n" + "=" * 60)
#     print("      FINAL POOL OF CANDIDATE SUMMARIES")
#     print("=" * 60)
#     final_candidates_list = list(labeled_candidates.items())
#     if not final_candidates_list:
#         print(
#             "No candidates were generated. The generation modules are currently empty."
#         )
#     else:
#         print(f"Generated a total of {len(final_candidates_list)} unique candidates.")
#         for i, (summary, module_name) in enumerate(final_candidates_list, 1):
#             print(f"\n--- Candidate #{i} ---")
#             print(f"Generated by: {module_name}")
#             print(f"Summary: {summary}")

#     print("\n" + "=" * 60)
#     print(f"(Total generation time: {end_time_total - start_time_total:.2f} seconds)")

# except Exception as e:
#     print(f"\n❌ An unexpected error occurred: {e}")

# print("\n--- Script Finished ---")


# ... [PASTE ALL YOUR ORIGINAL MODULE CODE HERE]
# (No edits or deletions—just copy as is, everything except the "main()" and "if __name__ == '__main__': main()" lines)

# ======= Streamlit UI Wrapper =======

import streamlit as st

st.set_page_config(page_title="AA News Ensemble Summarizer", layout="wide")
st.title("AA News Ensemble Summarizer")
st.write(
    "Paste a news article below and generate summaries with all ensemble modules. All your code and logic are preserved."
)

# ---- User inputs ----
article = st.text_area("Paste News Article Here", height=320, value="")
num_candidates = st.slider(
    "Number of creative candidates for reranker modules", 1, 10, 5
)

generate_clicked = st.button("Generate Summaries")

if generate_clicked:
    if article.strip():
        with st.spinner("Running all modules. Please wait..."):
            import time

            start_time = time.time()

            # MODEL_PATH = "bart-english-news-summarizer"
            GUIDING_TAGS = []
            INPUT_TEXT = article
            NUM_Candidates_FOR_RERANKING = num_candidates

            # Load models and pipelines
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
            import spacy

            device = 0 if torch.cuda.is_available() else -1
            # ozgur-coban/bart-english-news-summarizer/
            # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            # model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
            tokenizer = AutoTokenizer.from_pretrained(
                "ozgur-coban/bart-english-news-summarizer"
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                "ozgur-coban/bart-english-news-summarizer"
            )

            pipeline_beam = pipeline(
                "summarization", model=model, tokenizer=tokenizer, device=device
            )
            pipeline_sample = pipeline(
                "summarization", model=model, tokenizer=tokenizer, device=device
            )

            nlp_ner = spacy.load("en_core_web_sm")

            labeled_candidates = {}

            # [your module calls go here, unchanged]
            module_1_results = module_1_baseline_generation(
                INPUT_TEXT, pipeline_beam, pipeline_sample
            )
            for summary in module_1_results:
                if summary not in labeled_candidates:
                    labeled_candidates[summary] = "Module 1: Baseline Generation"

            module_2_results = module_2_simple_prompters(INPUT_TEXT, pipeline_beam)
            for summary in module_2_results:
                if summary not in labeled_candidates:
                    labeled_candidates[summary] = "Module 2: Simple Prompters"

            module_3_results = module_3_metadata_guided(
                INPUT_TEXT, GUIDING_TAGS, pipeline_beam, pipeline_sample
            )
            for summary in module_3_results:
                if summary not in labeled_candidates:
                    labeled_candidates[summary] = (
                        "Module 3: Metadata-Guided Summarizers"
                    )

            module_4_results = module_4_angled_summarizers(
                INPUT_TEXT, GUIDING_TAGS, pipeline_beam, pipeline_sample
            )
            for summary in module_4_results:
                if summary not in labeled_candidates:
                    labeled_candidates[summary] = "Module 4: Angled Summarizers"

            module_5_results = module_5_beam_and_sampling_options(
                INPUT_TEXT, NUM_Candidates_FOR_RERANKING, pipeline_beam, pipeline_sample
            )
            for summary in module_5_results:
                if summary not in labeled_candidates:
                    labeled_candidates[summary] = (
                        "Module 5: Beam + Multiple Sampling Options"
                    )

            module_6_results = module_6_auto_keyword_guided_yake(
                INPUT_TEXT, pipeline_beam, pipeline_sample
            )
            for summary in module_6_results:
                if summary not in labeled_candidates:
                    labeled_candidates[summary] = (
                        "Module 6: Automated Keyword-Guided Summarizers (YAKE)"
                    )

            module_7_results = module_7_keyword_reranker_simple(
                INPUT_TEXT, NUM_Candidates_FOR_RERANKING, pipeline_sample, nlp_ner
            )
            for summary in module_7_results:
                if summary not in labeled_candidates:
                    labeled_candidates[summary] = (
                        "Module 7: Keyword-Based Reranker (Simple Count)"
                    )

            module_8_results = module_8_weighted_keyword_reranker(
                INPUT_TEXT, NUM_Candidates_FOR_RERANKING, pipeline_sample, nlp_ner
            )
            for summary in module_8_results:
                if summary not in labeled_candidates:
                    labeled_candidates[summary] = "Module 8: Weighted Keyword Reranker"

            end_time = time.time()
            st.subheader(f"Candidate Summaries ({len(labeled_candidates)})")
            if not labeled_candidates:
                st.warning(
                    "No candidates were generated. The generation modules are currently empty."
                )
            else:
                for i, (summary, module_name) in enumerate(
                    labeled_candidates.items(), 1
                ):
                    st.markdown(f"**{i}.** *({module_name})*\n\n{summary}\n")
                st.info(f"Total generation time: {end_time - start_time:.2f} seconds.")
    else:
        st.warning("Please paste a news article first.")


# (Optional: You can add a footer, copy-to-clipboard, etc. later if you want.)
