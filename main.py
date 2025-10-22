# app.py
import os
import io
import time
import uuid
import logging
import numpy as np
import pandas as pd
import google.generativeai as genai

from typing import List, Dict, Any, Tuple
from fastapi import Body, HTTPException, FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from motor.motor_asyncio import AsyncIOMotorClient
from sentence_transformers import SentenceTransformer

from fastapi.middleware.cors import CORSMiddleware
import re
from bson import ObjectId

# --- Load env ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING")

DB_NAME = "test_case_db"
COLLECTION_NAME = "multilevel_test_cases_mongo"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_INDEX_NAME = "vector_index"

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Globals ---
embedding_model: SentenceTransformer | None = None
mongo_client: AsyncIOMotorClient | None = None
db_collection: Any | None = None  # Motor collection handle

# --- Cache ---
SEARCH_CACHE: Dict[str, Tuple[float, Any]] = {}
CACHE_TTL_SECONDS = 60 * 5  # 5 minutes

# --- Config ---
CANDIDATES_TO_RETRIEVE = 15
FINAL_RESULTS = 3
GEMINI_RERANK_ENABLED = True
QUERY_EXPANSION_ENABLED = True
DIVERSITY_ENFORCE = True
DIVERSITY_PER_FEATURE = True
GEMINI_RATE_LIMIT_SLEEP = 0.5
GEMINI_RETRIES = 2  # number of retries for enrichment/rerank

# --- Helpers ---
def numpy_to_list(v) -> List[float]:
    if v is None:
        return []
    if isinstance(v, list):
        return [float(x) for x in v]
    try:
        arr = np.asarray(v, dtype=float)
        return [float(x) for x in arr.tolist()]
    except Exception:
        # final fallback: try iterating
        return [float(x) for x in list(v)]

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0

def cache_get(query: str):
    entry = SEARCH_CACHE.get(query)
    if not entry:
        return None
    ts, value = entry
    if time.time() - ts > CACHE_TTL_SECONDS:
        del SEARCH_CACHE[query]
        return None
    return value

def cache_set(query: str, value: Any):
    SEARCH_CACHE[query] = (time.time(), value)

def safe_parse_lines(text: str) -> List[str]:
    return [l.strip() for l in text.splitlines() if l.strip()]

# --- Keyword extractor fallback ---
_STOPWORDS = set(
    """a about above after again against all am an and any are aren't as at be because been before being below between both but by
    can cannot could couldn't did didn't do does doesn't doing don't down during each few for from further had hadn't has hasn't have
    haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is
    isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves
    out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves
    then there there's these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll
    we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you
    you'd you'll you're you've your yours yourself yourselves""".split()
)

def extract_keywords(text: str, max_keywords: int = 15) -> List[str]:
    if not text:
        return []
    text = text.lower()
    words = re.findall(r"\b[a-zA-Z0-9\-']+\b", text)
    words_filtered = [w for w in words if w not in _STOPWORDS and len(w) > 2]
    from collections import Counter
    uni_counts = Counter(words_filtered)
    bigrams = [" ".join(pair) for pair in zip(words_filtered, words_filtered[1:])]
    big_counts = Counter(bigrams)
    candidates = {}
    for w, c in uni_counts.items():
        candidates[w] = candidates.get(w, 0) + c
    for b, c in big_counts.items():
        candidates[b] = candidates.get(b, 0) + c * 1.4
    sorted_items = sorted(candidates.items(), key=lambda x: (-x[1], x[0]))
    keywords = [k for k, _ in sorted_items][:max_keywords]
    if not keywords:
        keywords = [w for w in words if w not in _STOPWORDS][:max_keywords]
    seen = set()
    final = []
    for k in keywords:
        if k not in seen:
            final.append(k)
            seen.add(k)
    return final

def build_fallback_summary(description: str, steps: str, max_sentences: int = 2) -> str:
    text = (description or "").strip()
    if steps:
        text = (description or "") + "\n\n" + steps
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return (text[:500] + "...") if text else "Summary not available."
    summary = " ".join(sentences[:max_sentences])
    if len(summary) < 40 and len(sentences) > max_sentences:
        summary = " ".join(sentences[: max_sentences + 1])
    return (summary[:800] + "...") if len(summary) > 800 else summary

# --- Gemini helpers (expansion, rerank, enrichment) ---
def configure_gemini():
    try:
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
            logger.info("Gemini configured.")
        else:
            logger.warning("GOOGLE_API_KEY not set; Gemini disabled.")
    except Exception as e:
        logger.error(f"Failed to configure Gemini: {e}")

def expand_query_with_gemini(query: str) -> List[str]:
    if not QUERY_EXPANSION_ENABLED or not GOOGLE_API_KEY:
        return [query]
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""
You are an assistant that expands short search queries into useful paraphrases and synonyms for software test-case search.
Return only a comma-separated single line of 5 short paraphrases/keywords (no numbering).
Query: "{query}"
"""
        response = model.generate_content(prompt)
        text = response.text.strip()
        parts = [p.strip() for p in text.replace("\n", ",").split(",") if p.strip()]
        expansions = [query] + parts
        time.sleep(GEMINI_RATE_LIMIT_SLEEP)
        return expansions[:6]
    except Exception as e:
        logger.warning(f"expand_query_with_gemini failed: {e}")
        return [query]

def _parse_gemini_enrichment_text(text: str) -> Tuple[str, List[str]]:
    summary = ""
    keywords = []
    for line in text.splitlines():
        if line.lower().startswith("summary:"):
            summary = line.split(":", 1)[1].strip()
        elif line.lower().startswith("keywords:"):
            raw_kw = line.split(":", 1)[1]
            keywords = [k.strip() for k in raw_kw.split(",") if k.strip()]
    if not summary:
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        if parts:
            summary = parts[0][:800]
    if not keywords:
        keywords = extract_keywords(text, max_keywords=15)
    return summary, keywords

def get_gemini_enrichment(test_case_description: str, feature: str, steps: str = "") -> dict:
    description_text = (test_case_description or "").strip()
    steps_text = (steps or "").strip()
    fallback_summary = build_fallback_summary(description_text, steps_text, max_sentences=2)
    fallback_keywords = extract_keywords((description_text + " " + steps_text).strip(), max_keywords=15)

    if not GOOGLE_API_KEY:
        return {"summary": fallback_summary, "keywords": fallback_keywords}

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""
Analyze the following software test case and generate enriched metadata.
Feature: "{feature}"
Test Case Description: "{test_case_description}"
Steps: "{steps}"
Output format (exactly):
Summary: <3-5 lines clearly explaining purpose and process>
Keywords: <15-20+ diverse keywords/phrases, comma-separated>
"""
        for attempt in range(max(1, GEMINI_RETRIES)):
            try:
                response = model.generate_content(prompt)
                text = response.text.strip()
                summary, keywords = _parse_gemini_enrichment_text(text)
                if summary and len(summary) > 30 and len(keywords) >= 3:
                    return {"summary": summary, "keywords": keywords}
                time.sleep(GEMINI_RATE_LIMIT_SLEEP)
            except Exception as e:
                logger.warning(f"Gemini enrichment attempt {attempt+1} failed: {e}")
                time.sleep(GEMINI_RATE_LIMIT_SLEEP)

        # last attempt: try once more and merge partials with fallback
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            summary, keywords = _parse_gemini_enrichment_text(text)
            summary = summary or fallback_summary
            if not keywords or len(keywords) < 3:
                keywords = list(dict.fromkeys((keywords or []) + fallback_keywords))[:15]
            return {"summary": summary, "keywords": keywords}
        except Exception:
            return {"summary": fallback_summary, "keywords": fallback_keywords}
    except Exception as e:
        logger.error(f"Error invoking Gemini enrichment: {e}", exc_info=True)
        return {"summary": fallback_summary, "keywords": fallback_keywords}

def rerank_with_gemini(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not GEMINI_RERANK_ENABLED or not GOOGLE_API_KEY:
        return candidates
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt_lines = [
            "You are a helpful assistant. Re-rank the following test cases by relevance to the query.",
            f'Query: "{query}"',
            "Return only a newline-separated list of candidate IDs (the '_id' field) in best-to-worst order. Do not add extra commentary.",
            "\nCandidates:"
        ]
        for c in candidates:
            brief = (c.get("description") or c.get("summary") or "")[:220].replace("\n"," ")
            prompt_lines.append(f"{c['_id']} | Feature: {c.get('feature','N/A')} | Desc: {brief}")
        prompt = "\n".join(prompt_lines)
        response = model.generate_content(prompt)
        text = response.text.strip()
        lines = safe_parse_lines(text)
        ordered_ids = [l.split()[0].strip().strip(".").strip("-") for l in lines]
        id_to_c = {str(c["_id"]): c for c in candidates}
        ordered = [id_to_c[i] for i in ordered_ids if i in id_to_c]
        seen = {c["_id"] for c in ordered}
        ordered += [c for c in candidates if c["_id"] not in seen]
        time.sleep(GEMINI_RATE_LIMIT_SLEEP)
        return ordered
    except Exception as e:
        logger.warning(f"Gemini rerank failed: {e}")
        return candidates

# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model, mongo_client, db_collection

    logger.info("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("✅ Embedding model loaded.")

    logger.info("Connecting to MongoDB Atlas (async)...")
    try:
        mongo_client = AsyncIOMotorClient(
            MONGO_CONNECTION_STRING,
            serverSelectionTimeoutMS=5000,
            tls=True,
        )
        await mongo_client.admin.command("ping")
        db_collection = mongo_client[DB_NAME][COLLECTION_NAME]
        logger.info(f"✅ Connected to MongoDB | Collection='{COLLECTION_NAME}'")
        logger.warning("⚠️ Ensure a Vector Search Index is created in MongoDB Atlas.")
    except Exception as e:
        logger.error(f"❌ Failed to connect to MongoDB: {e}", exc_info=True)
        mongo_client, db_collection = None, None

    logger.info("Configuring Gemini (if key present)...")
    configure_gemini()

    yield

    if mongo_client:
        mongo_client.close()
        logger.info("🔒 MongoDB connection closed.")
    logger.info("👋 Lifespan shutdown complete.")

# --- App init ---
app = FastAPI(
    title="Intelligent Test Case Search API (MongoDB Edition)",
    description="Upload, enrich, and search test cases with MongoDB Atlas Vector Search.",
    version="1.2.3",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET","POST","PUT","DELETE","OPTIONS"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=".")

# --- Pydantic ---
class UpdateTestCaseRequest(BaseModel):
    feature: str | None = None
    summary: str | None = None
    description: str | None = None
    prerequisites: str | None = None
    steps: str | None = None
    keywords: List[str] | None = None

# --- Routes --- (upload/get-all/delete-all/delete single/update/search)
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/upload")
async def upload_and_process_file(file: UploadFile = File(...)):
    if not file.filename.endswith((".csv", ".xlsx")):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV or XLSX file.")
    if db_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB connection not available.")
    try:
        contents = await file.read()
        buffer = io.BytesIO(contents)
        if file.filename.endswith(".csv"):
            df = pd.read_csv(buffer, encoding="utf-8")
        else:
            df = pd.read_excel(buffer)

        df = df.astype(str).replace(["nan", "NaN", pd.NA], "")
        if "Test Case ID" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV/XLSX must contain 'Test Case ID' column.")
        df["Test Case ID"] = (df["Test Case ID"].replace("", pd.NA).fillna(method="ffill"))
        df.dropna(subset=["Test Case ID"], inplace=True)
        df = df[
            df["Test Case ID"].str.strip().str.upper().ne("NA")
            & df["Test Case ID"].str.strip().ne("")
        ]
    except Exception as e:
        logger.error(f"Error reading file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")

    documents_to_insert = []
    grouped = df.groupby("Test Case ID")
    logger.info(f"Processing {len(grouped)} unique test cases...")

    for test_case_id, group in grouped:
        feature = str(group.get("Feature", "").iloc[0]) if "Feature" in group else ""
        description = str(group.get("Test Case Description", "").iloc[0]) if "Test Case Description" in group else ""
        prerequisites = str(group.get("Pre-requisites", "").iloc[0]) if "Pre-requisites" in group else ""

        steps_list = []
        for _, row in group.iterrows():
            step_no = str(row.get("Step No.", "")).strip()
            test_step = str(row.get("Test Step", "")).strip()
            expected_result = str(row.get("Expected Result", "")).strip()
            if test_step:
                formatted_step = (f"Step {step_no}: {test_step}" if step_no else test_step)
                if expected_result:
                    formatted_step += f" → Expected: {expected_result}"
                steps_list.append(formatted_step)
        steps_combined = "\n\n".join(steps_list)

        enrichment = get_gemini_enrichment(description, feature, steps_combined)
        summary = enrichment.get("summary", "")
        keywords = enrichment.get("keywords", []) or []

        desc_emb = numpy_to_list(embedding_model.encode(description)) if description else []
        steps_emb = numpy_to_list(embedding_model.encode(steps_combined)) if steps_combined else []
        summary_emb = numpy_to_list(embedding_model.encode(summary)) if summary else []

        valid_embeddings = [np.asarray(x, dtype=float) for x in [desc_emb, steps_emb, summary_emb] if x]
        if valid_embeddings:
            main_vector_np = np.mean(valid_embeddings, axis=0)
            main_vector = numpy_to_list(main_vector_np)
        else:
            main_vector = numpy_to_list(embedding_model.encode(""))

        document = {
            "_id": str(uuid.uuid4()),
            "Test Case ID": test_case_id,
            "Feature": feature,
            "Test Case Description": description,
            "Pre-requisites": prerequisites,
            "Steps": steps_combined,
            "TestCaseSummary": summary,
            "TestCaseKeywords": keywords,
            "desc_embedding": desc_emb,
            "steps_embedding": steps_emb,
            "summary_embedding": summary_emb,
            "main_vector": main_vector,
        }
        documents_to_insert.append(document)

    if not documents_to_insert:
        return {"message": "No valid test cases found to process in the file."}

    try:
        result = await db_collection.insert_many(documents_to_insert)
        logger.info(f"Inserted {len(result.inserted_ids)} documents.")
        return {"message": f"Successfully processed and stored {len(result.inserted_ids)} test cases."}
    except Exception as e:
        logger.error(f"Error inserting into MongoDB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error storing data: {e}")

@app.get("/api/get-all")
async def get_all_test_cases(skip: int = 0, limit: int = 50, sort_by: str = "Test Case ID", order: int = 1):
    if db_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB connection not available.")
    try:
        projection = {"_id": 1, "main_vector": 0, "desc_embedding": 0, "steps_embedding": 0, "summary_embedding": 0}
        sort_order = -1 if order < 0 else 1
        cursor = db_collection.find({}, projection).sort(sort_by, sort_order).skip(skip).limit(limit)
        test_cases = []
        async for doc in cursor:
            doc["id"] = str(doc["_id"])
            del doc["_id"]
            test_cases.append(doc)
        return {"success": True, "count": len(test_cases), "skip": skip, "limit": limit, "test_cases": test_cases}
    except Exception as e:
        logger.error(f"Error retrieving test cases from MongoDB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while retrieving data.")

@app.post("/api/delete-all")
async def delete_all_data(confirm: bool = False):
    if db_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB connection not available.")
    if not confirm:
        raise HTTPException(status_code=400, detail="Confirmation required. Pass ?confirm=true to delete all data.")
    try:
        await db_collection.drop()
        logger.warning(f"⚠️ Dropped collection '{COLLECTION_NAME}'; all data cleared.")
        return {"success": True, "collection": COLLECTION_NAME, "message": f"All test case data in '{COLLECTION_NAME}' has been successfully deleted."}
    except Exception as e:
        logger.error(f"❌ Failed to delete all data from '{COLLECTION_NAME}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while deleting data.")

@app.delete("/api/testcase/{doc_id}")
async def delete_test_case(doc_id: str):
    if db_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB connection not available.")
    try:
        result = await db_collection.delete_one({"_id": doc_id})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Test case not found")
        logger.info(f"Deleted test case {doc_id}")
        return {"success": True, "message": f"Test case {doc_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting test case {doc_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete test case")

@app.put("/api/update/{doc_id}")
async def update_test_case(doc_id: str, update_data: UpdateTestCaseRequest = Body(...)):
    if db_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB connection not available.")
    try:
        existing_doc = await db_collection.find_one({"_id": doc_id})
        if existing_doc is None:
            raise HTTPException(status_code=404, detail="Test case not found")

        updated_doc = existing_doc.copy()
        should_reprocess = False

        if update_data.feature is not None:
            updated_doc["Feature"] = update_data.feature
            should_reprocess = True
        if update_data.description is not None:
            updated_doc["Test Case Description"] = update_data.description
            should_reprocess = True
        if update_data.prerequisites is not None:
            updated_doc["Pre-requisites"] = update_data.prerequisites
        if update_data.steps is not None:
            updated_doc["Steps"] = update_data.steps
            should_reprocess = True
        if update_data.summary is not None:
            updated_doc["TestCaseSummary"] = update_data.summary
        if update_data.keywords is not None:
            updated_doc["TestCaseKeywords"] = update_data.keywords

        if should_reprocess or not updated_doc.get("TestCaseSummary") or not updated_doc.get("TestCaseKeywords"):
            enrichment = get_gemini_enrichment(
                updated_doc.get("Test Case Description", ""),
                updated_doc.get("Feature", ""),
                updated_doc.get("Steps", ""),
            )
            if not update_data.summary:
                updated_doc["TestCaseSummary"] = enrichment.get("summary", "") or updated_doc.get("TestCaseSummary", "")
            if not update_data.keywords:
                updated_doc["TestCaseKeywords"] = enrichment.get("keywords", []) or updated_doc.get("TestCaseKeywords", [])

            desc_emb = numpy_to_list(embedding_model.encode(updated_doc.get("Test Case Description", "")))
            steps_emb = numpy_to_list(embedding_model.encode(updated_doc.get("Steps", "")))
            summary_emb = numpy_to_list(embedding_model.encode(updated_doc.get("TestCaseSummary", "")))

            valid_embeddings = [np.asarray(x, dtype=float) for x in [desc_emb, steps_emb, summary_emb] if x]
            if valid_embeddings:
                main_vector_np = np.mean(valid_embeddings, axis=0)
                main_vector = numpy_to_list(main_vector_np)
            else:
                main_vector = numpy_to_list(embedding_model.encode(""))

            updated_doc.update({
                "desc_embedding": desc_emb,
                "steps_embedding": steps_emb,
                "summary_embedding": summary_emb,
                "main_vector": main_vector,
            })

        await db_collection.replace_one({"_id": doc_id}, updated_doc)
        logger.info(f"✅ Updated test case {doc_id}")

        response_doc = updated_doc.copy()
        for field in ["desc_embedding", "steps_embedding", "summary_embedding", "main_vector"]:
            response_doc.pop(field, None)

        return {"success": True, "message": f"Test case {doc_id} updated successfully", "updated_test_case": response_doc}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error updating test case {doc_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while updating the test case.")

@app.post("/api/search")
async def search_test_cases(request: Request):
    if db_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB connection not available.")

    data = await request.json()
    raw_query = (data.get("query") or "").strip()
    filter_feature = (data.get("feature") or "").strip() or None

    if not raw_query:
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    cache_key = f"{raw_query}::feature={filter_feature}"
    if cached := cache_get(cache_key):
        logger.info(f"⚡ Returning cached results for '{raw_query}' (feature={filter_feature})")
        return {**cached, "from_cache": True}

    logger.info(f"🔍 Search query='{raw_query}' | feature filter='{filter_feature}'")

    expansions = expand_query_with_gemini(raw_query) if QUERY_EXPANSION_ENABLED else [raw_query]
    all_expansions = [raw_query] + (expansions or [])
    combined_query = " ".join(all_expansions)

    try:
        query_vector_raw = embedding_model.encode(combined_query)
        query_vector = numpy_to_list(query_vector_raw)
    except Exception as e:
        logger.exception("Failed to compute query embedding")
        raise HTTPException(status_code=500, detail="Failed to compute query embedding.")

    query_vec_np = np.asarray(query_vector, dtype=float)

    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "main_vector",
                "queryVector": query_vector,
                "numCandidates": 150,
                "limit": CANDIDATES_TO_RETRIEVE,
            }
        },
        {"$project": {"score": {"$meta": "vectorSearchScore"}, "document": "$$ROOT"}},
    ]
    if filter_feature:
        pipeline[0]["$vectorSearch"]["filter"] = {"Feature": {"$eq": filter_feature}}

    try:
        search_results = await db_collection.aggregate(pipeline).to_list(length=CANDIDATES_TO_RETRIEVE)
    except Exception as e:
        logger.error(f"❌ MongoDB vector search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Search failed due to database error.")

    def cosine_sim(a, b):
        try:
            a_arr = np.asarray(a, dtype=float)
            b_arr = np.asarray(b, dtype=float)
            if a_arr.size == 0 or b_arr.size == 0:
                return 0.0
            denom = (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
            return float(np.dot(a_arr, b_arr) / denom) if denom > 0 else 0.0
        except Exception:
            return 0.0

    def tokenize_for_boost(text):
        if not text:
            return set()
        toks = re.findall(r"\b[\w\-']+\b", text.lower())
        return set(toks)

    expansion_tokens = set()
    for ex in all_expansions:
        expansion_tokens.update(tokenize_for_boost(ex))

    candidates = []
    for res in search_results:
        payload = res.get("document", {}) or {}
        try:
            base_score = float(res.get("score", 0.0) or 0.0)
        except Exception:
            base_score = 0.0

        desc_emb = payload.get("desc_embedding") or []
        steps_emb = payload.get("steps_embedding") or []
        summary_emb = payload.get("summary_embedding") or []

        sim_desc = cosine_sim(query_vec_np, desc_emb)
        sim_steps = cosine_sim(query_vec_np, steps_emb)
        sim_summary = cosine_sim(query_vec_np, summary_emb)

        keywords = [str(k).lower() for k in payload.get("TestCaseKeywords", []) or []]
        text_fields = f'{payload.get("Feature","")} {payload.get("Test Case Description","")} {payload.get("Steps","")}'.lower()
        text_tokens = tokenize_for_boost(text_fields)

        token_boost = 0.0
        for tok in expansion_tokens:
            if tok in text_tokens:
                token_boost += 0.10
            if tok in keywords:
                token_boost += 0.15

        local_score = (0.6 * base_score + 0.25 * max(sim_desc, sim_steps, sim_summary) + token_boost)

        candidates.append({
            "_id": payload.get("_id") or payload.get("id"),
            "raw_score": base_score,
            "local_score": float(local_score),
            "feature": payload.get("Feature", "N/A"),
            "test_case_id": payload.get("Test Case ID", "NA"),
            "description": payload.get("Test Case Description", "") or payload.get("description", ""),
            "summary": payload.get("TestCaseSummary", "") or payload.get("summary", ""),
            "keywords": payload.get("TestCaseKeywords", []) or payload.get("keywords", []),
            "payload": payload,
        })

    if not candidates:
        result = {"query": raw_query, "feature_filter": filter_feature, "results_count": 0, "results": [], "from_cache": False}
        cache_set(cache_key, result)
        return result

    local_scores = [c["local_score"] for c in candidates]
    min_s = float(np.min(local_scores))
    max_s = float(np.max(local_scores))
    if max_s - min_s > 1e-12:
        for c in candidates:
            c["local_score_norm"] = (c["local_score"] - min_s) / (max_s - min_s)
    else:
        for c in candidates:
            c["local_score_norm"] = 1.0

    candidates.sort(key=lambda x: x["local_score"], reverse=True)
    top_candidates = candidates[:CANDIDATES_TO_RETRIEVE]

    try:
        reranked = rerank_with_gemini(raw_query, top_candidates)
    except Exception as e:
        logger.warning(f"Rerank with Gemini failed: {e}; falling back to local ranking.")
        reranked = top_candidates

    final_list = []
    seen_features = set()
    for cand in reranked:
        c_id = cand.get("_id") if isinstance(cand, dict) else getattr(cand, "_id", None)
        c_key = str(c_id)
        original = next((x for x in candidates if (str(x["_id"]) == c_key) or (str(x.get("test_case_id","")) == str(cand.get("test_case_id","")))), cand)
        if len(final_list) >= FINAL_RESULTS:
            break
        if DIVERSITY_ENFORCE and DIVERSITY_PER_FEATURE:
            feat = (original.get("feature") or "N/A")
            if feat in seen_features:
                continue
            final_list.append(original)
            seen_features.add(feat)
        else:
            final_list.append(original)

    if len(final_list) < FINAL_RESULTS:
        seen_ids = {str(c["_id"]) for c in final_list}
        for cand in candidates:
            if len(final_list) >= FINAL_RESULTS:
                break
            if str(cand["_id"]) not in seen_ids:
                final_list.append(cand)

    response_items = []
    for c in final_list:
        norm = float(c.get("local_score_norm", 0.0))
        score_pct = round(max(0.0, min(1.0, norm)) * 100, 2)
        payload = c.get("payload", {}) or {}
        response_items.append({
            "id": str(c.get("_id") or payload.get("_id") or payload.get("id")),
            "probability": score_pct,
            "test_case_id": payload.get("Test Case ID", "NA"),
            "feature": payload.get("Feature", "N/A"),
            "description": payload.get("Test Case Description", "") or payload.get("description",""),
            "prerequisites": payload.get("Pre-requisites", ""),
            "steps": payload.get("Steps", ""),
            "summary": payload.get("TestCaseSummary", "") or payload.get("summary",""),
            "keywords": payload.get("TestCaseKeywords", []) or payload.get("keywords", []),
        })

    result = {"query": raw_query, "feature_filter": filter_feature, "results_count": len(response_items), "results": response_items, "from_cache": False}
    cache_set(cache_key, result)
    logger.info(f"✅ Returning {len(response_items)} results for query '{raw_query}'")
    return result

# --- End ---
