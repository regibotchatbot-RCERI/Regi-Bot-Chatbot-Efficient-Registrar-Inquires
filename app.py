from flask import Flask, render_template, request, jsonify
# ============================================================
# IMPORTS PARA SA DATA LOADING AT NLP
# ============================================================
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import os
import threading
import time
import re
from thefuzz import fuzz, process
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import numpy as np


# ============================================================
# FLASK APP INITIALIZATION
# ============================================================
app = Flask(__name__)

# ==================== LOAD ENV ====================
load_dotenv()
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
CREDENTIALS_FILE_PATH = os.getenv("GOOGLE_CREDENTIALS_FILE")
USE_GOOGLE_SHEETS = os.getenv("USE_GOOGLE_SHEETS", "True") == "True"
EXCEL_FILE_PATH = os.getenv("EXCEL_FILE_PATH", "knowledge_base.xlsx")

IS_RENDER = os.getenv("RENDER", "0") == "1"

# ============================================================
# GLOBAL VARIABLES
# ============================================================
knowledge_base = {}
acronym_to_full_word = {}
last_modified_time = 0
current_topic = None

print("🔄 Loading NLP model for smarter understanding...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("✅ NLP model loaded successfully!")

knowledge_texts = []
knowledge_embeddings = None

# ============================================================
# LANGUAGE DETECTION KEYWORDS
# ============================================================
tagalog_keywords = [
    "ano", "paano", "saan", "kailan", "kelan", "magkano", "po", "ko", "ba",
    "kausapin", "requirements", "kung", "tungkol", "magdrop", "nag-drop",
    "mag-enroll", "enroll", "anong", "rekwirements", "pagaaral", "bayarin",
    "tuition"
]

english_keywords = [
    "what", "how", "where", "when", "how much", "hello", "hi", "about", "who",
    "drop", "ask", "enroll", "enrollment", "requirements", "study", "tuition",
    "fee", "pay"
]

# ============================================================
# CONTEXTUAL QUESTION TAGGING
# ============================================================
QUESTION_TAGS = {
    "WHERE_TAG": ["saan", "where is", "lokasyon", "location", "locate", "nasaan"],
    "WHEN_TAG": ["kailan", "when", "oras", "hours", "schedule", "opening", "closing"],
    "HOW_MUCH_TAG": ["magkano", "how much", "tuition", "fee", "bayad", "presyo"],
    "WHO_TAG": ["sino", "who is", "contact person", "kausapin", "may hawak"],
    "GENERAL_TAG": ["ano", "paano", "what is", "how to", "paliwanag", "explain"]
}

# ============================================================
# TOPIC CATEGORIES
# ============================================================
REGISTRAR_TOPICS = ["registrar", "records", "transkript", "tor", "form 137", "form 138"]
ACCOUNTING_TOPICS = ["accounting", "bayarin", "tuition", "fee", "bayad", "utang", "balance"]
GENERAL_INFO_TOPICS = ["office hours", "oras", "lokasyon"]

# ============================================================
# KNOWLEDGE BASE LOADING FUNCTIONS
# ============================================================
def load_knowledge_base_from_google_sheets():
    global knowledge_base, acronym_to_full_word, knowledge_texts, knowledge_embeddings

    try:
        print("🔄 Reloading knowledge base from Google Sheets...")
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE_PATH, scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(GOOGLE_SHEET_ID)
        worksheet = sheet.get_worksheet(0)
        all_data = worksheet.get_all_records()

        new_knowledge_base = {}
        new_acronym_to_full_word = {}

        for row in all_data:
            keywords_str = str(row.get('Keyword', '')).strip()
            if not keywords_str:
                continue

            keywords = [kw.strip().lower() for kw in keywords_str.split(',') if kw.strip()]
            tagalog_response = str(row.get('Tagalog_Response', '')).strip() or "Walang sagot sa Tagalog."
            english_response = str(row.get('English_Response', '')).strip() or "No English response."

            for keyword in keywords:
                new_knowledge_base[keyword] = {
                    "tagalog": tagalog_response,
                    "english": english_response
                }

            if len(keywords) > 1:
                for i in range(len(keywords)):
                    for j in range(len(keywords)):
                        if i != j:
                            combo = f"{keywords[i]} {keywords[j]}".strip()
                            if combo not in new_knowledge_base:
                                new_knowledge_base[combo] = {
                                    "tagalog": tagalog_response,
                                    "english": english_response
                                }

                new_acronym_to_full_word[keywords[0]] = keywords[1]

        knowledge_base = new_knowledge_base
        acronym_to_full_word = new_acronym_to_full_word
        knowledge_texts = list(knowledge_base.keys())
        print(f"🔢 Total keywords before encoding: {len(knowledge_texts)}")

        knowledge_embeddings = model.encode(knowledge_texts, convert_to_tensor=True)
        print("✅ NLP embeddings generated successfully.")
        return True

    except Exception as e:
        print(f"⚠️ Error loading from Google Sheets: {e}")
        return False


def load_knowledge_base_from_excel(file_path):
    global knowledge_base, acronym_to_full_word
    try:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return False

        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            print("Error: Unsupported file format.")
            return False

        new_knowledge_base = {}
        new_acronym_to_full_word = {}

        for _, row in df.iterrows():
            keywords_str = str(row['Keyword']).strip()
            keywords = [kw.strip().lower() for kw in keywords_str.split(',') if kw.strip()]
            tagalog_response = str(row['Tagalog_Response']).strip() if pd.notna(row['Tagalog_Response']) else "Walang sagot sa Tagalog."
            english_response = str(row['English_Response']).strip() if pd.notna(row['English_Response']) else "No English response."

            for keyword in keywords:
                new_knowledge_base[keyword] = {
                    "tagalog": tagalog_response,
                    "english": english_response
                }

            if len(keywords) > 1:
                acronym = keywords[0]
                full_word_phrase = keywords[1]
                new_acronym_to_full_word[acronym] = full_word_phrase

        knowledge_base = new_knowledge_base
        acronym_to_full_word = new_acronym_to_full_word
        print("✅ Knowledge base reloaded successfully.")
        return True

    except Exception as e:
        print(f"⚠️ Error loading from Excel: {e}")
        return False


def check_for_changes_and_reload():
    global last_modified_time
    if USE_GOOGLE_SHEETS:
        load_knowledge_base_from_google_sheets()
    else:
        try:
            current_modified_time = os.path.getmtime(EXCEL_FILE_PATH)
            if current_modified_time > last_modified_time:
                if load_knowledge_base_from_excel(EXCEL_FILE_PATH):
                    last_modified_time = current_modified_time
        except FileNotFoundError:
            pass

        # Background auto-checker every 2 seconds
def auto_reload_worker():
    while True:
        try:
            check_for_changes_and_reload()
        except Exception as e:
            print("Auto reload error:", e)
        time.sleep(2)

        if not IS_RENDER:
        # start auto-reload thread only in local/dev environments
            reload_thread = threading.Thread(target=auto_reload_worker, daemon=True)
            reload_thread.start()
        else:
            print("⚠️ Auto-reload thread disabled on Render (RENDER=1) to save memory.")



# ============================================================
# LANGUAGE DETECTION
# ============================================================
def detect_language(text):
    text_lower = text.lower()
    tagalog_count = sum(1 for k in tagalog_keywords if re.search(r'\b' + re.escape(k) + r'\b', text_lower))
    english_count = sum(1 for k in english_keywords if re.search(r'\b' + re.escape(k) + r'\b', text_lower))

    if tagalog_count > english_count:
        return "tagalog"
    elif english_count > tagalog_count:
        return "english"
    return "tagalog"

# ============================================================
# NLP SUPPORT FUNCTIONS
# ============================================================
def expand_acronyms(user_message_lower):
    expanded_message = user_message_lower
    for acronym, full_word in acronym_to_full_word.items():
        if 2 <= len(acronym) <= 5:
            expanded_message = re.sub(r'\b' + re.escape(acronym) + r'\b', full_word, expanded_message)
    return expanded_message


def get_best_nlp_match(user_message):
    global current_topic
    expanded_message = expand_acronyms(user_message)
    user_embedding = model.encode(expanded_message, convert_to_tensor=True)
    cosine_scores = util.cos_sim(user_embedding, knowledge_embeddings)[0]
    best_match_idx = int(cosine_scores.argmax())
    best_score = float(cosine_scores[best_match_idx])
    best_keyword = knowledge_texts[best_match_idx]

    if current_topic and current_topic != "multiple":
        if current_topic in best_keyword or best_keyword in current_topic:
            best_score = min(best_score + 0.10, 1.0)

    return best_keyword, best_score


def find_best_match(query, choices, threshold=85):
    if not choices:
        return None, 0
    matches = process.extractBests(query, choices, scorer=fuzz.partial_ratio, score_cutoff=threshold)
    if matches:
        matches.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
        return matches[0][0], matches[0][1]
    return None, 0

def log_unmatched_query(query):
    with open("unmatched_queries.txt", "a") as f:
        f.write(query + "\n")

# ============================================================
# MAIN BOT RESPONSE FUNCTION
# ============================================================
def get_bot_response(user_message):
    global current_topic

    import difflib  #  for typo / fuzzy spelling correction

    user_message_lower = user_message.lower().strip()
    detected_lang = detect_language(user_message)
    found_responses = {}
    keyword_pattern = ["School ID", " lost ID", " shift course", "course"] # to avoid duplicate matches

    # ============================================================
    # SPELLING CORRECTION PREPROCESSING
    # ============================================================
    # automatically corrects words close to known keywords
    all_keywords = [k.lower() for k in knowledge_base.keys()]
    user_words = user_message_lower.split()

    corrected_words = []
    for word in user_words:
        # find close match (tolerance for misspelled words)
        close = difflib.get_close_matches(word, all_keywords, n=1, cutoff=0.82)
        corrected_words.append(close[0] if close else word)

    user_message_lower = " ".join(corrected_words)
    # ============================================================


    # STEP 1: GREETINGS
    if any(re.search(r'\b' + w + r'\b', user_message_lower) for w in ["hi", "hello", "kumusta", "kamusta"]):
        current_topic = "general"
        return "Hello! Kumusta? Ano ang maipaglilingkod ko sa iyo ngayon?" if detected_lang == "tagalog" else "Hello! How can I help you today?"

    if any(re.search(r'\b' + w + r'\b', user_message_lower) for w in ["thank you", "salamat", "ok", "okay", "sige", "noted", "ty", "thanks"]):
        current_topic = "general"
        return "Walang anuman! May iba ka pa bang gustong itanong?" if detected_lang == "tagalog" else "You're welcome! Do you have another question?"

    # STEP 2: TOPIC SWITCHING
    COURSE_TOPICS = ["bscs", "computer science", "bsba", "business administration", "bsed", "beed", "crim", "criminology"]

    if re.search(r'\b(bscs|bsba|crim|bsed|beed)\b', user_message_lower):
        for topic in COURSE_TOPICS:
            if topic in user_message_lower:
                current_topic = topic
                if topic in knowledge_base:
                    lang = "tagalog" if detected_lang == "tagalog" else "english"
                    return knowledge_base[topic][lang]
                


    # STEP 3: CONTEXTUAL SEARCH 
    if current_topic != "general" and len(user_message_lower.split()) < 6:
        contextual_query = f"{current_topic} {user_message_lower}"
        best_match_keyword, score = find_best_match(contextual_query, knowledge_base.keys(), threshold=85)
        if best_match_keyword and score >= 85:
            response_text = knowledge_base.get(best_match_keyword, {}).get(detected_lang)
            if response_text:
                return response_text


    # STEP 3.5: QUESTION DETECTION 
    question_indicators = [
        "?", "paano", "kailan", "saan", "magkano", 
        "ano", "pwede", "can", "how", "when", 
        "where", "what", "could", "would", "may"
    ]

   # STEP 4: EXACT KEYWORD MATCHING
    # ================== STEP 4: EXACT KEYWORD MATCHING ==================
    single_letters = ["A", "B", "C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    default_fallback = "Paumanhin, hindi ko maintindihan ang iyong tanong. Maaari mo bang ulitin o linawin?"

    # Split user input by common separators
    multi_parts = re.split(r'\s*(?:and|,|\+|\/|&|at)\s*', user_message_lower)
    multi_parts = [p.strip() for p in multi_parts if p.strip()]

    # Sort keywords by length (longest first) to avoid partial matches
    sorted_keywords = sorted(knowledge_base.keys(), key=len, reverse=True)

    matched_keywords = set()
    found_responses = {}
    used_responses = set()  # tracks already added responses

    for part in multi_parts:
        # --- 1️⃣ Single letter default ---
        if len(part) == 1 and part.upper() in single_letters:
            if default_fallback.lower() not in used_responses:
                found_responses[part.upper()] = default_fallback
                used_responses.add(default_fallback.lower())
            continue  # skip normal keyword matching

     # --- 2️⃣ Exact keyword matching ---
    for keyword in sorted_keywords:
        # Handle keyword aliases separated by commas
        for kw in [k.strip().lower() for k in keyword.split(",")]:
            if re.search(rf'\b{re.escape(kw)}\b', part):
                response_text = knowledge_base.get(keyword, {}).get(detected_lang, "").strip()
                
                # Skip empty, default fallback placeholders, or already used
                if not response_text or response_text.lower() in ["walang sagot sa tagalog.", "no english response."]:
                    continue
                if response_text.lower() in used_responses:
                    continue

                # Add response and mark keyword/response as used
                matched_keywords.add(keyword)
                found_responses[keyword] = response_text
                used_responses.add(response_text.lower())

                # Stop after first match per part
                break

    # --- 3️⃣ Return combined responses if any ---
    if found_responses:
        current_topic = "general"
        return "\n\n".join(list(dict.fromkeys(found_responses.values()))).strip()




    # STEP 5: FUZZY SEARCH (improved threshold flexibility)
    best_match_keyword, score = find_best_match(user_message_lower, knowledge_base.keys(), threshold=85)  # lowered a bit
    if best_match_keyword and score >= 85:
        response_text = knowledge_base.get(best_match_keyword, {}).get(detected_lang)
        if response_text and response_text not in ["Walang sagot sa Tagalog.", "No English response."]:
            current_topic = "general"
            return response_text


    # STEP 6: NLP FALLBACK (no changes)
    try:
        best_keyword_nlp, similarity = get_best_nlp_match(user_message_lower)
        if similarity >= 0.90:
            current_topic = "general"
            return knowledge_base.get(best_keyword_nlp, {}).get(detected_lang)
    except Exception:
        pass


    # STEP 7: GENERIC FALLBACK (improved response wording)
    if not any(q in user_message_lower for q in question_indicators):
        log_unmatched_query(user_message_lower)
    
    current_topic = "general"

    return (
        "Paumanhin, hindi ko maintindihan ang iyong tanong. Maaari mo bang ulitin o linawin?"
        if detected_lang == "tagalog"
        else "Sorry, I couldn’t understand that. Could you please rephrase your question?"
    )


# ============================================================
# FLASK ROUTES
# ============================================================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    bot_response = get_bot_response(user_message)
    return jsonify({'response': bot_response})

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == '__main__':
    if USE_GOOGLE_SHEETS:
        if not load_knowledge_base_from_google_sheets():
            print("Initial knowledge base load failed.")
    else:
        if not load_knowledge_base_from_excel(EXCEL_FILE_PATH):
            print("Initial knowledge base load failed.")

    print("--------------------------------------------------")
    print("      REGI-BOT CONSOLE TEST (2-TIER SYSTEM)      ")
    print("--------------------------------------------------")

    test_queries = [
        "Hello",
        "Ano ang BSCS?",
        "I want to know about the tuition fee",
        "Magkano ang bayarin?",
        "Where is the registrar?",
        "How about BSBA?",
        "Salamat sa tulong",
        "requirements",
        "Gusto ko mag-shift ng course",
        "shift course"
    ]

    for query in test_queries:
        response = get_bot_response(query)
        print(f"User: '{query}'")
        print(f"Bot: '{response}'")
        print(f"Current Topic: {current_topic}")
        print("-" * 20)

    current_topic = None

    print("--------------------------------------------------")
    print("            Starting Flask Web Server            ")
    print("--------------------------------------------------")

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
