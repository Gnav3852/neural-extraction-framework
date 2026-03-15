#!/usr/bin/env python3
# nef_cli.py
# Usage examples are at the bottom of this file (or run with -h)

import os, sys, json, re, time, argparse, textwrap
from typing import List, Tuple, Sequence, Dict, Any, Optional, Set
from collections import OrderedDict

import numpy as np
from urllib.parse import quote
from getpass import getpass

# =============== Gemini client bootstrap ===============
try:
    from google import genai
    from google.genai import types
except Exception as _e:
    sys.stderr.write("ERROR: google-genai is required. Try: pip install google-genai\n")
    raise

def _bootstrap_gemini_client(api_key: Optional[str]) -> "genai.Client":
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key and sys.stdin.isatty():
        key = getpass("Enter your Google (Gemini) API key: ").strip()
    if not key:
        raise RuntimeError("No Gemini API key found. Pass --api-key or set GEMINI_API_KEY.")
    return genai.Client(api_key=key)

# =============== Utils ===============

_YEAR = re.compile(r"^\d{4}$")

def _year_uri(y: str) -> str:
    return f"http://dbpedia.org/resource/{y}"

def _normalize(vec: Sequence[float]) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v)) or 1e-12
    return v / n

def _json_from_model(text: str) -> Any:
    t = (text or "").strip()
    # strip markdown fences if any
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.IGNORECASE | re.MULTILINE).strip()
    m = re.search(r"\{.*\}|\[.*\]", t, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object/array found in model output.")
    return json.loads(m.group(0))

def _safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except Exception:
        sys.stdout.write((" ".join(map(str, args)) + "\n").encode("utf-8", "ignore").decode("utf-8"))

# =============== Redis Entity Linking ===============

class RedisEntityLinking:
    """Redis-based entity linking; hard-requires Redis (no surface fallbacks here)."""
    def __init__(
        self,
        host: str,
        port: int,
        password: Optional[str],
        connect_timeout: float = 2.0,
        verbose: bool = True,
        cache_size: int = 10000,  # Max cache entries
    ):
        self.available = False
        self.redis_forms = None
        self.redis_redir = None
        self.verbose = verbose
        self.cache_size = cache_size
        # In-memory cache: {cache_key: List[Tuple[str, float]]}
        # Using OrderedDict for LRU eviction
        self._lookup_cache: OrderedDict[str, List[Tuple[str, float]]] = OrderedDict()
        try:
            import redis  # local import so script still loads without it
            # Set timeouts to prevent hanging
            # socket_connect_timeout: time to establish connection
            # socket_timeout: time for each operation (read/write) - critical for preventing hangs
            common = dict(
                host=host, 
                port=port, 
                password=password,
                socket_connect_timeout=connect_timeout, 
                socket_timeout=3.0,  # 3 second timeout per operation (prevents 220s hangs)
                socket_keepalive=True,  # Keep connection alive
                decode_responses=True,
                retry_on_timeout=True,  # Retry on timeout
                health_check_interval=30,  # Health check every 30s
            )
            self.redis_forms = redis.Redis(db=0, **common)
            self.redis_redir  = redis.Redis(db=1, **common)
            
            # Test connection with timeout protection
            try:
                self.redis_forms.ping()
                self.redis_redir.ping()
                self.available = True
            except Exception:
                self.available = False
            
            if self.verbose:
                _safe_print("✓ Connected to Redis" if self.available else "✗ Redis ping failed")
        except Exception as e:
            self.available = False
            _safe_print(f"✗ Redis connection error (pipeline will drop ungrounded triples): {e}")

    def _redirect(self, uri: str, max_hops: int = 10) -> str:
        """Follow redirects in Redis db1 with timeout protection."""
        if not self.available:
            return uri
        seen = set()
        cur = uri
        for _ in range(max_hops):
            if cur in seen:
                break
            seen.add(cur)
            try:
                nxt = self.redis_redir.get(cur)
                if not nxt:
                    return cur
                cur = nxt
            except Exception:
                # If redirect lookup fails, return current URI
                return cur
        return cur

    def lookup(self, surface_form: str, top_k: int = 5, thr: float = 0.01) -> List[Tuple[str, float]]:
        """
        Strict Redis grounding (no synonyms). Tries simple, non-semantic variants:
        exact, lower, Title Case, capitalize, underscores, etc.
        Aggregates counts across variants and follows redirects in db1.
        Optimized with early exit, timeout protection, and in-memory caching.
        """
        if not self.available or not surface_form.strip():
            return []

        # Check cache first (fast path)
        cache_key = f"{surface_form}|{top_k}|{thr}"
        if cache_key in self._lookup_cache:
            # Move to end (most recently used) for LRU
            result = self._lookup_cache.pop(cache_key)
            self._lookup_cache[cache_key] = result
            if self.verbose:
                _safe_print(f"[CACHE HIT] '{surface_form}'")
            return result

        # Cache miss - proceed with Redis lookup
        if self.verbose:
            _safe_print(f"[CACHE MISS] '{surface_form}' - querying Redis...")

        # Remove leading articles (the, a, an) for better matching
        def remove_articles(text: str) -> str:
            text = text.strip()
            articles = ["the ", "a ", "an "]
            for article in articles:
                if text.lower().startswith(article):
                    text = text[len(article):].strip()
                    break
            return text

        base_form = remove_articles(surface_form)
        
        # Generate more comprehensive case variants (with and without articles)
        variants = [
            surface_form,                                    # Original (with article)
            base_form,                                       # Without article
            surface_form.lower(),                            # lowercase (with article)
            base_form.lower(),                               # lowercase (without article)
            surface_form.title(),                            # Title Case (with article)
            base_form.title(),                               # Title Case (without article)
            surface_form.capitalize(),                       # First letter capitalized (with article)
            base_form.capitalize(),                         # First letter capitalized (without article)
            surface_form.upper(),                           # UPPERCASE (for acronyms, with article)
            base_form.upper(),                              # UPPERCASE (for acronyms, without article)
            surface_form.replace(" ", "_"),                 # spaces to underscores (with article)
            base_form.replace(" ", "_"),                    # spaces to underscores (without article)
            surface_form.title().replace(" ", "_"),         # Title Case with underscores (with article)
            base_form.title().replace(" ", "_"),           # Title Case with underscores (without article)
            base_form.capitalize().replace(" ", "_"),      # Capitalize with underscores (without article)
        ]
        
        # Remove duplicates while preserving order
        seen_variants = set()
        unique_variants = []
        for v in variants:
            if v not in seen_variants:
                seen_variants.add(v)
                unique_variants.append(v)

        counts: Dict[str, int] = {}
        seen_keys = set()
        found_any = False
        
        for key in unique_variants:
            if key in seen_keys:
                continue
            seen_keys.add(key)
            
            try:
                # Try Redis lookup with timeout protection (socket_timeout should handle this)
                raw = self.redis_forms.hgetall(key)  # {uri: count}
                
                if raw:
                    found_any = True
                    for uri, v in raw.items():
                        try:
                            canon = self._redirect(uri)  # db1 redirect if any
                            counts[canon] = counts.get(canon, 0) + int(v)
                        except Exception:
                            # If redirect fails, use original URI
                            counts[uri] = counts.get(uri, 0) + int(v)
                    
                    # Early exit: if we found results and have enough support, stop trying variants
                    if counts:
                        max_count = max(counts.values())
                        if max_count >= 10:  # High confidence threshold for early exit
                            break
            except Exception as e:
                # If Redis operation fails, continue to next variant
                # Don't log every failure to avoid spam
                continue

        if not counts:
            result = []
        else:
            max_support = max(counts.values()) or 1
            items = [(uri, c / max_support) for uri, c in counts.items() if (c / max_support) >= thr]
            items.sort(key=lambda x: x[1], reverse=True)
            result = items[:top_k]

        # Store in cache (with LRU eviction if needed)
        if len(self._lookup_cache) >= self.cache_size:
            # Remove least recently used (first item)
            self._lookup_cache.popitem(last=False)
        self._lookup_cache[cache_key] = result

        return result

# =============== Predicate Retriever (precomputed) ===============

class PredicateEmbeddingRetriever:
    """
    Loads embeddings.npy (N, D) and predicates.csv (URIs or CSV with 'predicate' column),
    retrieves top-K predicates via cosine similarity. NO synonym expansion.
    """
    def __init__(
        self,
        client: "genai.Client",
        embeddings_path: Optional[str] = None,
        predicates_path: Optional[str] = None,
        embed_model: str = "gemini-embedding-001",
        verbose: bool = True,
    ):
        self.client = client
        self.embed_model = embed_model
        self.verbose = verbose

        emb_path, pred_path = self._find_files(embeddings_path, predicates_path)
        self.E: np.ndarray = np.load(emb_path)  # (N, D)
        self.predicates: List[str] = self._load_predicates(pred_path)
        if self.E.shape[0] != len(self.predicates):
            raise ValueError(f"Row count mismatch: embeddings ({self.E.shape[0]}) vs predicates ({len(self.predicates)})")

        self.D = int(self.E.shape[1])
        self.E_norm = self.E / (np.linalg.norm(self.E, axis=1, keepdims=True) + 1e-12)

        if self.verbose:
            _safe_print(f"✓ Loaded embeddings: {emb_path} shape={self.E.shape}")
            _safe_print(f"✓ Loaded predicates: {pred_path} count={len(self.predicates)}")

    def _find_files(self, emb_path: Optional[str], pred_path: Optional[str]) -> Tuple[str, str]:
        if emb_path and pred_path and os.path.exists(emb_path) and os.path.exists(pred_path):
            return emb_path, pred_path
        cand_emb = ["./embeddings.npy", "../embeddings.npy", "embeddings.npy"]
        cand_pred = ["./predicates.csv", "../predicates.csv", "predicates.csv"]
        e = next((p for p in cand_emb if os.path.exists(p)), None)
        p = next((q for q in cand_pred if os.path.exists(q)), None)
        if not (e and p):
            raise FileNotFoundError(f"Could not find embeddings.npy and predicates.csv in {os.getcwd()} or ../")
        if self.verbose:
            _safe_print(f"✓ Found files: {e}, {p}")
        return e, p

    def _load_predicates(self, path: str) -> List[str]:
        preds: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            head = f.readline()
            if "predicate" in head.lower():
                for line in f:
                    parts = line.rstrip("\n").split(",")
                    if parts:
                        preds.append(parts[0] if head.lower().startswith("predicate") else parts[-1])
            else:
                preds.append(head.strip())
                for line in f:
                    preds.append(line.strip())
        preds = [p for p in preds if p]
        return preds

    def _embed_text(self, text: str) -> np.ndarray:
        cfg = types.EmbedContentConfig(output_dimensionality=int(self.D))
        resp = self.client.models.embed_content(model=self.embed_model, contents=text, config=cfg)
        v = getattr(resp, "embeddings", None)
        v = (v[0].values if v else resp.embedding.values)
        return _normalize(v)

    def get_top_k_predicates(self, relation_text: str, top_k: int = 10, allowed_predicates: Optional[Set[str]] = None) -> List[Tuple[str, float]]:
        """
        Retrieve top-K predicates via cosine similarity.
        
        Args:
            relation_text: Natural language text describing the relation
            top_k: Number of top predicates to return
            allowed_predicates: Optional set of predicate URIs to filter to (pre-filters embeddings)
        
        Returns:
            List of (predicate_uri, similarity_score) tuples
        """
        q = self._embed_text(relation_text)      # (D,)
        
        # Pre-filter embeddings if allowed_predicates is provided
        if allowed_predicates is not None:
            # Find indices of predicates that are in the allowed set
            allowed_indices = [i for i, pred in enumerate(self.predicates) if pred in allowed_predicates]
            
            if not allowed_indices:
                # No allowed predicates found in embeddings
                if self.verbose:
                    _safe_print(f"   ⚠️  No allowed predicates found in embeddings (searched {len(allowed_predicates)} predicates)")
                return []
            
            # Filter embeddings and similarities to only allowed predicates
            E_filtered = self.E_norm[allowed_indices]  # (M, D) where M = len(allowed_indices)
            sims_filtered = E_filtered @ q              # (M,)
            predicates_filtered = [self.predicates[i] for i in allowed_indices]
            
            # Get top-k from filtered set
            order = sims_filtered.argsort()[-top_k:][::-1]
            return [(predicates_filtered[i], float(sims_filtered[i])) for i in order]
        else:
            # Original behavior: search all predicates
            sims = self.E_norm @ q                   # (N,)
            order = sims.argsort()[-top_k:][::-1]
            return [(self.predicates[i], float(sims[i])) for i in order]

# =============== LLM Disambiguator ===============

class LLMDisambiguator:
    def __init__(
        self,
        client,
        model_name: str = "gemini-2.5-flash",
        predicate_threshold: float = 0.5,
        new_predicate_namespace: str | None = None,
        verbose: bool = True,
        **kwargs,  # absorb any unexpected args from the pipeline
    ):
        self.client = client
        self.model_name = model_name
        self.thr = float(predicate_threshold)
        self.new_predicate_namespace = new_predicate_namespace
        self.verbose = verbose
        # Optionally stash the rest if you ever want to inspect them:
        self._extra_kwargs = kwargs

    # ------------------------ helpers ------------------------

    @staticmethod
    def _as_pairs(lst: Any) -> List[Tuple[str, float]]:
        """Normalize a list of candidates to [(uri:str, score:float), ...]."""
        out: List[Tuple[str, float]] = []
        if not lst:
            return out
        for item in lst:
            uri, score = "", 1.0
            if isinstance(item, (list, tuple)):
                if len(item) >= 1:
                    uri = str(item[0])
                if len(item) >= 2:
                    try:
                        score = float(item[1])
                    except Exception:
                        score = 1.0
            elif isinstance(item, dict):
                uri = str(item.get("uri") or item.get("id") or item.get("value") or "")
                try:
                    score = float(item.get("score", 1.0))
                except Exception:
                    score = 1.0
            else:
                uri = str(item)
                score = 1.0
            if uri:
                out.append((uri, score))
        return out

    @staticmethod
    def _fmt_indexed(lst: List[Tuple[str, float]]) -> str:
        """Pretty print [(uri, score)] as indexed lines; tolerate empty."""
        if not lst:
            return "(empty)"
        try:
            return "\n".join([f'{i}. "{u}" (sim={s:.3f})' for i, (u, s) in enumerate(lst)]) or "(empty)"
        except Exception:
            # ultra-safe fallback
            return "\n".join([f'{i}. "{str(item)}"' for i, item in enumerate(lst)]) or "(empty)"

    @staticmethod
    def _safe_idx(lst: List[Any], value: Any) -> Optional[int]:
        """Clamp int(value) to valid range; return 0 on parse error; None for empty lst."""
        if not lst:
            return None
        try:
            i = int(value)
        except Exception:
            i = 0
        return max(0, min(i, len(lst) - 1))

    @staticmethod
    def _get_json(text: Optional[str]) -> Dict[str, Any]:
        try:
            return json.loads((text or "").strip() or "{}")
        except Exception:
            return {}

    # --------------------- main entrypoint ---------------------

    def disambiguate_triple(
        self,
        context: str,
        subject_candidates: List[Tuple[str, float]],
        predicate_candidates: List[Tuple[str, float]],
        object_candidates: List[Tuple[str, float]],
        predicate_metadata: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        """
        Returns: (subject_uri, predicate_uri, object_uri, meta)
        - subject_uri, object_uri are ALWAYS from the provided candidate lists.
        - predicate_uri is chosen from candidate predicates >= threshold.
        """

        # Normalize inputs so mixed shapes don't crash anything
        subject_candidates   = self._as_pairs(subject_candidates)
        predicate_candidates = self._as_pairs(predicate_candidates)
        object_candidates    = self._as_pairs(object_candidates)

        total_k = len(predicate_candidates)
        sim_map: Dict[str, Tuple[Optional[float], int]] = {
            u: (s, i) for i, (u, s) in enumerate(predicate_candidates)
        }

        # Filter predicates by threshold
        above = [(u, s) for (u, s) in predicate_candidates if (s is not None and s >= self.thr)]

        # If no predicate passes threshold, fall back to top-1 S/O and return None predicate
        if not above:
            s_uri = subject_candidates[0][0] if subject_candidates else ""
            o_uri = object_candidates[0][0] if object_candidates else ""
            meta = {
                "label": "no_predicate_above_threshold",
                "topk": total_k,
                "threshold": self.thr,
            }
            return (s_uri, None, o_uri, meta)

        # Prepare prompt pieces
        allowed = [u for (u, _s) in above]
        
        # Build predicate list with metadata if available
        if predicate_metadata:
            pred_lines = []
            for uri in allowed:
                meta = predicate_metadata.get(uri, {})
                label = meta.get("label", uri.split("/")[-1] if "/" in uri else uri)
                domain = meta.get("domain", "")
                range_type = meta.get("range", "")
                
                # Build description line
                desc_parts = [f'label: {label}']
                if domain:
                    desc_parts.append(f'domain: {domain}')
                if range_type:
                    desc_parts.append(f'range: {range_type}')
                
                desc = f" ({', '.join(desc_parts)})" if desc_parts else ""
                pred_lines.append(f'- "{uri}"{desc}')
            pred_list_text = "\n".join(pred_lines)
        else:
            pred_list_text = "\n".join([f'- "{u}"' for u in allowed])
        
        subj_list_text = self._fmt_indexed(subject_candidates)
        obj_list_text  = self._fmt_indexed(object_candidates)

        prompt = f"""Given the context and the candidate options below, pick the one subject (by index), one predicate (by URI), and one object (by index) that best fit the meaning of the context.

Allowed predicate URIs (with semantic information):
{pred_list_text}

Subject candidates (choose by INDEX):
{subj_list_text}

Object candidates (choose by INDEX):
{obj_list_text}

Context:
{context}

Reply with JSON on one line only:
{{"subject_index": 0, "predicate": "URI", "object_index": 0}}
"""

        # Call the model (Gemini client style)
        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )

        data = self._get_json(getattr(resp, "text", None))

        # Validate predicate
        pred_uri = data.get("predicate", "")
        if pred_uri not in allowed:
            pred_uri = allowed[0]

        # Clamp indices and map to URIs
        si = self._safe_idx(subject_candidates, data.get("subject_index", 0))
        oi = self._safe_idx(object_candidates,  data.get("object_index", 0))

        s_uri = subject_candidates[si][0] if (subject_candidates and si is not None) else ""
        o_uri = object_candidates[oi][0] if (object_candidates and oi is not None) else ""

        # Paranoia: role-swap & collision guards
        subj_pool = {u for (u, _s) in subject_candidates}
        obj_pool  = {u for (u, _s) in object_candidates}

        # If s_uri is actually from object pool and o_uri from subject pool, swap them back
        if s_uri in obj_pool and o_uri in subj_pool and s_uri not in subj_pool:
            s_uri, o_uri = o_uri, s_uri

        # Avoid identical subject/object when pools differ (rare but can happen with weird lists)
        if subj_pool and obj_pool and s_uri == o_uri and len(obj_pool) > 1:
            # push object to top-1 object if collision looks suspicious
            o_uri = object_candidates[0][0]

        # Build meta (compatible with your previous code)
        chosen_sim, rank0 = sim_map.get(pred_uri, (None, None))
        meta = {
            "label": "candidate",
            "chosen_similarity": float(chosen_sim) if chosen_sim is not None else None,
            "rank_in_topk": (rank0 + 1) if rank0 is not None else None,
            "topk": total_k,
            "threshold": self.thr,
        }

        return (s_uri, pred_uri, o_uri, meta)

# =============== Orchestrator ===============

class EnhancedNEFPipeline:
    """
    End-to-end pipeline:
      1) Extract triples with Gemini
      2) Redis entity linking (subject/object)  [REQUIRED]
      3) Predicate retrieval via precomputed embeddings (no synonyms)
      4) LLM disambiguation
    
    Optional: allowed_predicates can be used to filter predicates to a specific ontology.
    This improves ontology conformance by only considering predicates that match the ontology.
    """
    def __init__(
        self,
        client: "genai.Client",
        embeddings_path: Optional[str] = None,
        predicates_path: Optional[str] = None,
        llm_model: str = "gemini-2.5-flash",
        predicate_threshold: float = 0.5,
        new_predicate_namespace: str = "http://nef.local/rel/",
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        redis_password: Optional[str] = None,
        allowed_predicates: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        self.redis_host = redis_host or os.getenv("NEF_REDIS_HOST")
        self.redis_port = (
            redis_port
            if redis_port is not None
            else int(os.getenv("NEF_REDIS_PORT"))
        )
        # No hardcoded password; stays None unless provided
        self.redis_password = (
            redis_password
            if redis_password is not None
            else os.getenv("NEF_REDIS_PASSWORD")
        )
        self.verbose = verbose
        self.client = client  # for extractor too
        self.require_redis_grounding = True  # strict
        
        # Store allowed predicates (set of URIs for fast lookup)
        # If provided, only predicates matching these URIs will be considered
        self.allowed_predicates: Optional[set] = None
        if allowed_predicates:
            self.allowed_predicates = set(allowed_predicates)
            if self.verbose:
                _safe_print(f"✓ Ontology filtering enabled: {len(self.allowed_predicates)} allowed predicates")
        
        # Store predicate metadata (label, domain, range) for better disambiguation
        self.predicate_metadata: Optional[Dict[str, Dict[str, str]]] = None
        
        # Store ontology context for extraction phase
        self.ontology_context: Optional[str] = None
        
        self.redis_el = RedisEntityLinking(
            host=redis_host, port=int(redis_port), password=redis_password, verbose=verbose
        )
        self.pred = PredicateEmbeddingRetriever(
            client=self.client,
            embeddings_path=embeddings_path,
            predicates_path=predicates_path,
            embed_model="gemini-embedding-001",
            verbose=verbose,
        )
        self.llm = LLMDisambiguator(
            client=self.client,
            model_name=llm_model,
            predicate_threshold=predicate_threshold,
            new_predicate_namespace=new_predicate_namespace,
            verbose=verbose,
        )
        if self.verbose:
            _safe_print("✓ Enhanced NEF Pipeline initialized")

    # helpers: lowercase + spacing only
    def _lc_space(self, s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip().lower())

    def _valid_predicate(self, p: str) -> bool:
        w = re.sub(r"\s+", " ", (p or "").strip()).split()
        return 1 <= len(w) <= 3

    def _resolve_entities(self, mention: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Strict grounding via Redis, except:
          - If mention is a 4-digit year, mint a DBpedia year URI immediately.
        """
        m = (mention or "").strip()
        if _YEAR.fullmatch(m):
            return [(_year_uri(m), 1.0)]  # treat year as an entity, no Redis hit

        cands = self.redis_el.lookup(m, top_k=k)
        fixed: List[Tuple[str, float]] = []
        for uri, score in cands:
            if not (uri.startswith("http://") or uri.startswith("https://")):
                uri = f"http://dbpedia.org/resource/{uri}"
            fixed.append((uri, score))
        return fixed

    def _is_likely_literal(self, text: str) -> bool:
        """
        Check if text looks like a number or date literal.
        Returns True if it should skip Redis grounding.
        """
        if not text:
            return False
        
        text = text.strip().lower()  # Normalize to lowercase for date detection
        
        # Remove all spaces for number checking
        text_no_spaces = text.replace(' ', '')
        
        # Check for numbers (integers, decimals, with commas)
        # Remove commas and check if it's a number
        text_no_commas = text_no_spaces.replace(',', '')
        # Check if it's all digits (possibly with one decimal point)
        if text_no_commas.replace('.', '', 1).isdigit():
            return True
        
        # Check for structured date patterns (YYYY-MM-DD, YYYY/MM/DD, etc.)
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',      # 1920-08-16
            r'^\d{4}/\d{2}/\d{2}$',      # 1920/08/16
            r'^\d{4}\.\d{2}\.\d{2}$',    # 1920.08.16
            r'^\d{2}-\d{2}-\d{4}$',      # 08-16-1920
            r'^\d{2}/\d{2}/\d{4}$',      # 08/16/1920
        ]
        for pattern in date_patterns:
            if re.match(pattern, text):
                return True
        
        # Check for natural language dates (e.g., "august 16th, 1920", "january 1, 2000")
        # Look for month names followed by optional day and year
        month_names = [
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december',
            'jan', 'feb', 'mar', 'apr', 'may', 'jun',
            'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec'
        ]
        
        # Check if text starts with a month name
        for month in month_names:
            if text.startswith(month):
                # Check if it contains a 4-digit year
                if re.search(r'\b\d{4}\b', text):
                    return True
        
        # Check for patterns like "YYYY-MM-DD" with spaces: "1920 - 08 - 16"
        if re.match(r'^\d{4}\s*[-/\.]\s*\d{1,2}\s*[-/\.]\s*\d{1,2}$', text):
            return True
        
        # Check if text is just a 4-digit year
        if re.match(r'^\d{4}$', text):
            return True
        
        return False

    def _extract_triples(self, text: str, ontology_context: Optional[str] = None, debug: bool = False) -> list[dict]:
        """
        Strict extractor:
        - preserves subject/object casing as in the text (for entity linking)
        - enforces 1–3 word predicates
        - confidence ≥ 0.5
        - REQUIRES Redis grounding for subject
        - Object can be entity (Redis grounding) or literal (numbers/dates)
        - Optionally uses ontology context to guide extraction
        """
        # Build prompt with optional ontology context
        ontology_section = ""
        if ontology_context:
            ontology_section = f"\n{ontology_context}\n"
        
        prompt = f"""
SYSTEM: Return ONLY a valid JSON array (no prose, no markdown fences).

Task: Read the text and extract up to 5 RDF triples with confidence.{ontology_section}
You MUST:
- Write subject, predicate, and object exactly as they appear in the text (preserve capitalization; do not lowercase).
- Use the most complete, consistent entity names.
- Resolve clear pronouns (he, she, it, they, this/that, here/there) to the correct entity; if unclear, do not guess.
- Keep predicates extremely concise: 1–3 words max (e.g., "founded", "born in", "wrote").
- Include only items with confidence ≥ 0.5.
- If ontology context is provided, extract triples according to the ontology relations and concepts.
- For DATE objects: Normalize to YYYY-MM-DD format (e.g., "Aug. 16, 1920" → "1920-08-16", "January 1, 2000" → "2000-01-01").
- For NUMBER objects: Remove commas and use digits only (e.g., "15,100,000,000" → "15100000000", "5,594" → "5594").
- Optionally include "object_type" field: "entity", "literal", "number", or "date" to help processing.

Output schema:
[
  {{"subject":"...", "predicate":"...", "object":"...", "object_type":"entity|literal|number|date", "confidence":0.0}},
  ...
]

Text:
{text}
""".strip()

        try:
            # DIAGNOSTIC: Track timing at each step
            if self.verbose:
                _safe_print("[DIAG] Step 1: About to call generate_content...")
                _safe_print(f"[DIAG] Prompt length: {len(prompt)} characters")
            
            step1_start = time.time()
            resp = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            step1_time = time.time() - step1_start
            
            if self.verbose:
                _safe_print(f"[DIAG] Step 1: generate_content returned in {step1_time:.2f}s")
            
            # DIAGNOSTIC: Check resp.text access
            if self.verbose:
                _safe_print("[DIAG] Step 2: About to access resp.text...")
            
            step2_start = time.time()
            response_text = resp.text or "[]"
            step2_time = time.time() - step2_start
            
            if self.verbose:
                _safe_print(f"[DIAG] Step 2: resp.text accessed in {step2_time:.2f}s")
                _safe_print(f"[DIAG] Response length: {len(response_text)} characters")
                if len(response_text) > 1000:
                    _safe_print(f"[DIAG] Response preview (first 200 chars): {response_text[:200]}...")
                else:
                    _safe_print(f"[DIAG] Full response: {response_text}")

            if debug:
                _safe_print("\n[DEBUG] Raw model response text:")
                _safe_print(response_text.strip() or "<EMPTY>")

            # DIAGNOSTIC: Check JSON parsing
            if self.verbose:
                _safe_print("[DIAG] Step 3: About to parse JSON...")
            
            step3_start = time.time()
            try:
                items = _json_from_model(response_text)
                step3_time = time.time() - step3_start
                if self.verbose:
                    _safe_print(f"[DIAG] Step 3: JSON parsed in {step3_time:.2f}s")
                    _safe_print(f"[DIAG] Found {len(items)} items in JSON array")
            except Exception as e:
                step3_time = time.time() - step3_start
                if self.verbose:
                    _safe_print(f"[DIAG] Step 3: JSON parse FAILED after {step3_time:.2f}s")
                    _safe_print(f"[DIAG] Error: {e}")
                if debug:
                    _safe_print(f"[DEBUG] JSON parse error: {e}")
                return []

            if debug:
                _safe_print("[DEBUG] Parsed JSON items:",
                            json.dumps(items, indent=2) if isinstance(items, list) else items)

            if not isinstance(items, list):
                if debug:
                    _safe_print("[DEBUG] Parsed payload is not a list → aborting.")
                return []

            out, seen = [], set()
            if self.verbose:
                _safe_print(f"[DIAG] Step 4: Processing {len(items)} extracted items...")
            
            for idx, it in enumerate(items, 1):
                if self.verbose:
                    _safe_print(f"[DIAG] Processing item {idx}/{len(items)}...")
                
                s_raw = (it.get("subject") or "").strip()
                p_raw = (it.get("predicate") or "").strip()
                o_raw = (it.get("object") or "").strip()
                obj_type = (it.get("object_type") or "").strip().lower()  # Get object_type from LLM
                conf  = it.get("confidence", None)

                try:
                    conf_f = float(conf) if conf is not None else 1.0
                except Exception:
                    conf_f = 0.0

                reasons = []
                if not s_raw or not p_raw or not o_raw:
                    reasons.append("missing field(s)")

                s = self._lc_space(s_raw)
                p = self._lc_space(p_raw)
                o = self._lc_space(o_raw)

                if not self._valid_predicate(p):
                    reasons.append(f"predicate length invalid: '{p}'")

                if conf_f < 0.5:
                    reasons.append(f"confidence < 0.5 (got {conf_f:.3f})")

                # DIAGNOSTIC: Check Redis lookups
                if self.verbose:
                    _safe_print(f"[DIAG] Step 4.{idx}.1: About to resolve entities for subject: '{s_raw}'")
                step4_sub_start = time.time()
                try:
                    sub_cands = self._resolve_entities(s_raw, k=5)
                except Exception as e:
                    if self.verbose:
                        _safe_print(f"[DIAG] Step 4.{idx}.1: Subject resolution ERROR: {e}")
                    sub_cands = []
                step4_sub_time = time.time() - step4_sub_start
                if self.verbose:
                    _safe_print(f"[DIAG] Step 4.{idx}.1: Subject resolution took {step4_sub_time:.2f}s, found {len(sub_cands)} candidates")
                    if step4_sub_time > 5.0:
                        _safe_print(f"[DIAG] ⚠️ WARNING: Subject resolution took {step4_sub_time:.2f}s (>5s threshold)")
                
                # Check if object is a literal (use LLM's object_type if available, otherwise use function)
                is_literal = False
                if obj_type in ["literal", "number", "date"]:
                    is_literal = True
                elif not obj_type:  # If LLM didn't provide object_type, use function detection
                    is_literal = self._is_likely_literal(o_raw)
                
                if is_literal:
                    # Skip Redis grounding for literals
                    obj_cands = []
                    if self.verbose:
                        _safe_print(f"[DIAG] Step 4.{idx}.2: Object '{o_raw}' detected as literal (type: {obj_type or 'auto-detected'}), skipping Redis")
                else:
                    # Normal Redis grounding for entities
                    if self.verbose:
                        _safe_print(f"[DIAG] Step 4.{idx}.2: About to resolve entities for object: '{o_raw}'")
                    step4_obj_start = time.time()
                    try:
                        obj_cands = self._resolve_entities(o_raw, k=5)
                    except Exception as e:
                        if self.verbose:
                            _safe_print(f"[DIAG] Step 4.{idx}.2: Object resolution ERROR: {e}")
                        obj_cands = []
                    step4_obj_time = time.time() - step4_obj_start
                    if self.verbose:
                        _safe_print(f"[DIAG] Step 4.{idx}.2: Object resolution took {step4_obj_time:.2f}s, found {len(obj_cands)} candidates")
                        if step4_obj_time > 5.0:
                            _safe_print(f"[DIAG] ⚠️ WARNING: Object resolution took {step4_obj_time:.2f}s (>5s threshold)")
                
                # Only require Redis grounding for subject and non-literal objects
                if not sub_cands:
                    reasons.append("no Redis grounding for subject (required)")
                if not is_literal and not obj_cands:
                    reasons.append("no Redis grounding for object (entity required)")

                if (s, p, o) in seen:
                    reasons.append("duplicate triple")

                if reasons:
                    if debug:
                        _safe_print(f"[DEBUG] Item #{idx} REJECTED:",
                                    {"subject": s_raw, "predicate": p_raw, "object": o_raw, "confidence": conf},
                                    "→ reasons:", "; ".join(reasons))
                    continue

                seen.add((s, p, o))
                kept = {
                    "subject": s, 
                    "predicate": p, 
                    "object": o,
                    "_sub_cands": sub_cands, 
                    "_obj_cands": obj_cands,
                    "_is_literal": is_literal,  # Store literal flag for run_pipeline
                    "_object_type": obj_type  # Store object_type from LLM
                }
                out.append(kept)

                if debug:
                    _safe_print(f"[DEBUG] Item #{idx} KEPT:",
                                {"subject": s, "predicate": p, "object": o,
                                 "sub_cands": sub_cands[:2], "obj_cands": obj_cands[:2], "confidence": conf_f})

                if len(out) >= 5:
                    break

            if debug and not out:
                _safe_print("[DEBUG] Result: 0 triples kept after filtering.")
            
            if self.verbose:
                total_time = time.time() - step1_start
                _safe_print(f"[DIAG] Total extraction time: {total_time:.2f}s")
                _safe_print(f"[DIAG] Breakdown: API={step1_time:.2f}s, text_access={step2_time:.2f}s, parse={step3_time:.2f}s, processing={total_time-step1_time-step2_time-step3_time:.2f}s")
                _safe_print(f"[DIAG] Final result: {len(out)} triples kept")

            return out

        except Exception as e:
            if getattr(self, "verbose", True) or debug:
                _safe_print(f"✗ Triple extraction error: {e}")
                import traceback
                if self.verbose:
                    _safe_print("[DIAG] Full traceback:")
                    traceback.print_exc()
            return []

    def run_pipeline(self, sentence: str, debug: bool = False) -> list[tuple[str, str, str, Dict[str, Any]]]:
        """
        End-to-end for one sentence.
        - Subject: Redis grounding (required)
        - Object: Redis grounding for entities, literal detection for numbers/dates
        - Predicate: Embedding-based retrieval with LLM disambiguation
        Returns list of (subjectURI, predicateURI, objectURI_or_literal, meta)
        """
        if self.verbose:
            _safe_print(f"\n📝 {sentence!r}")

        raw_triples = self._extract_triples(sentence, ontology_context=self.ontology_context, debug=debug)
        if not raw_triples:
            if self.verbose:
                _safe_print("   ⚠ No triples extracted.")
            return []

        results: list[tuple[str, str, str, Dict[str, Any]]] = []

        for t in raw_triples:
            s_text = t.get("subject", "")
            p_text = t.get("predicate", "")
            o_text = t.get("object", "")
            if not (s_text and p_text and o_text):
                continue

            if self.verbose:
                _safe_print(f"\n🔍 Triple: {s_text} — {p_text} — {o_text}")
                _safe_print("   📍 Using entity candidates collected during extraction...")

            subject_candidates = t.get("_sub_cands") or self._resolve_entities(s_text, k=5)
            
            # Check if object is a literal (use stored flag from extraction, or detect)
            is_literal = t.get("_is_literal", False)
            if not is_literal:
                # Fallback: check if LLM marked it or use function detection
                obj_type = t.get("_object_type", "").lower()
                if obj_type in ["literal", "number", "date"]:
                    is_literal = True
                else:
                    is_literal = self._is_likely_literal(o_text)
            
            if is_literal:
                # Skip Redis grounding for literals - use literal value directly
                object_candidates = [(o_text, 1.0)]
                if self.verbose:
                    obj_type_str = t.get("_object_type", "auto-detected")
                    _safe_print(f"   [Object:literal] {o_text} (type: {obj_type_str})")
            else:
                # Normal Redis grounding for entities
                object_candidates = t.get("_obj_cands") or self._resolve_entities(o_text, k=5)

            if self.verbose:
                _safe_print("   [Redis:subject]", subject_candidates[:5] if subject_candidates else "NO CANDIDATES")
                if not is_literal:
                    _safe_print("   [Redis:object]",  object_candidates[:5]  if object_candidates  else "NO CANDIDATES")

            # Only check subject grounding - object can be literal or entity
            if not subject_candidates:
                if self.verbose:
                    _safe_print("   ⚠ Abandoning triple (no Redis candidates for subject).")
                continue

            # Object can be empty only if it's not a literal
            if not object_candidates and not is_literal:
                if self.verbose:
                    _safe_print("   ⚠ Abandoning triple (no Redis candidates for object).")
                continue

            # Pre-filter embeddings to only search within allowed predicates
            predicate_candidates = self.pred.get_top_k_predicates(
                p_text, 
                top_k=10, 
                allowed_predicates=self.allowed_predicates
            )
            
            if self.verbose:
                if self.allowed_predicates:
                    _safe_print(f"   [Pre-filtered search: {len(self.allowed_predicates)} allowed predicates]")
                if predicate_candidates:
                    _safe_print(f"   [Found: {len(predicate_candidates)} predicate candidates]")
                else:
                    _safe_print(f"   ⚠️  No predicate candidates found (may need to increase top_k or check ontology mapping)")
            
            if self.verbose:
                _safe_print("   [Predicates:top5]", predicate_candidates[:5])

            s_final, p_final, o_final, meta = self.llm.disambiguate_triple(
                sentence, subject_candidates, predicate_candidates, object_candidates,
                predicate_metadata=self.predicate_metadata
            )
            results.append((s_final, p_final, o_final, meta or {}))

            label = (meta or {}).get("label", "candidate")
            tag_str = "[GENERATED]" if label == "generated" else "[CANDIDATE]"
            sim = (meta or {}).get("chosen_similarity")
            rank = (meta or {}).get("rank_in_topk")
            topk = (meta or {}).get("topk")
            thr = (meta or {}).get("threshold")
            sim_str = f" (sim={sim:.3f})" if isinstance(sim, (int, float)) else ""
            rank_str = f" rank={rank}/{topk}" if (isinstance(rank, int) and isinstance(topk, int)) else ""
            thr_str = f" | thr={thr:.2f}" if isinstance(thr, (int, float)) else ""
            if self.verbose:
                _safe_print("   ✅ Final", f"{tag_str}{sim_str}{rank_str}{thr_str}:",
                            s_final, p_final, o_final, sep="\n            ")

        return results

# =============== CLI ===============

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="nef_cli.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Enhanced NEF (strict) — CLI for triple extraction → Redis grounding → predicate disambiguation."
    )
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("-s", "--sentence", nargs="+", help="One or more sentences to process.")
    inp.add_argument("-f", "--file", type=str, help="Path to a UTF-8 text file (one sentence per line).")
    inp.add_argument("--stdin", action="store_true", help="Read sentences from STDIN (one per line).")

    # Output
    p.add_argument("-o", "--output", choices=["json", "jsonl", "tsv", "nt"], default="json",
                   help=textwrap.dedent("""\
                   Output format:
                     json  = single JSON object with 'results' array
                     jsonl = one JSON object per line
                     tsv   = subject\\tpredicate\\tobject
                     nt    = N-Triples (URIs only)
                   """))
    p.add_argument("--no-verbose", dest="verbose", action="store_false", help="Silence progress logs.")
    p.add_argument("--debug", action="store_true", help="Show extractor debug dumps.")
    p.set_defaults(verbose=True)

    # Gemini / models
    p.add_argument("--api-key", type=str, default=None, help="Gemini API key (or set GEMINI_API_KEY).")
    p.add_argument("--llm-model", type=str, default="gemini-2.5-flash", help="LLM for disambiguation/generation.")
    p.add_argument("--embed-model", type=str, default="gemini-embedding-001", help="Embedding model name.")
    p.add_argument("--predicate-threshold", type=float, default=0.5, help="Similarity threshold to accept a predicate.")
    p.add_argument("--new-predicate-namespace", type=str, default="http://nef.local/rel/",
                   help="Namespace for generated predicates.")

    # Predicate embeddings
    p.add_argument("--embeddings", type=str, default=None, help="Path to embeddings.npy")
    p.add_argument("--predicates", type=str, default=None, help="Path to predicates.csv")

    # Redis
    p.add_argument("--redis-host", type=str, default=os.getenv("NEF_REDIS_HOST", ""))
    p.add_argument("--redis-port", type=int, default=int(os.getenv("NEF_REDIS_PORT", "")))
    p.add_argument("--redis-password", type=str, default=os.getenv("NEF_REDIS_PASSWORD", ""))

    return p.parse_args(argv)

def _collect_sentences(args: argparse.Namespace) -> List[str]:
    if args.sentence:
        return [" ".join(args.sentence)] if len(args.sentence) > 1 else [args.sentence[0]]
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    if args.stdin:
        return [line.strip() for line in sys.stdin if line.strip()]
    return []

def _emit(results: Dict[str, Any], fmt: str, verbose: bool):
    """results = {'items': [ {'input': str, 'triples':[{'s':..., 'p':..., 'o':..., 'meta':{...}}, ...]}, ... ]}"""
    items = results.get("items", [])

    if fmt == "json":
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    if fmt == "jsonl":
        for rec in items:
            print(json.dumps(rec, ensure_ascii=False))
        return

    if fmt == "tsv":
        for rec in items:
            for t in rec.get("triples", []):
                print(f"{t['s']}\t{t['p']}\t{t['o']}")
        return

    if fmt == "nt":
        # Only URIs are printed; assume s/p/o are URIs
        for rec in items:
            for t in rec.get("triples", []):
                s, p, o = t["s"], t["p"], t["o"]
                # N-Triples line
                print(f"<{s}> <{p}> <{o}> .")
        return

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    try:
        client = _bootstrap_gemini_client(args.api_key)
    except Exception as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return 2

    # Build pipeline
    try:
        pipe = EnhancedNEFPipeline(
            client=client,
            embeddings_path=args.embeddings,
            predicates_path=args.predicates,
            llm_model=args.llm_model,
            predicate_threshold=args.predicate_threshold,
            new_predicate_namespace=args.new_predicate_namespace,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            redis_password=args.redis_password,
            verbose=args.verbose,
        )
        # ensure retriever uses requested embed model
        pipe.pred.embed_model = args.embed_model
    except Exception as e:
        sys.stderr.write(f"ERROR initializing pipeline: {e}\n")
        return 3

    sentences = _collect_sentences(args)
    if not sentences:
        sys.stderr.write("No input sentences.\n")
        return 1

    out: Dict[str, Any] = {"items": []}
    had_any = False

    for s in sentences:
        triples = pipe.run_pipeline(s, debug=args.debug)
        norm = [{"s": t[0], "p": t[1], "o": t[2], "meta": t[3]} for t in triples]
        out["items"].append({"input": s, "triples": norm})
        if norm:
            had_any = True

    _emit(out, args.output, args.verbose)
    return 0 if had_any else 4

if __name__ == "__main__":
    # Quick usage help when executed without args in TTY
    if len(sys.argv) == 1 and sys.stdin.isatty():
        _safe_print(textwrap.dedent("""\
            Enhanced NEF CLI

            Examples:
              # Single sentence → JSON
              python nef_cli.py -s "Steve Jobs founded Apple" --embeddings embeddings.npy --predicates predicates.csv

              # Read from file, emit N-Triples
              python nef_cli.py -f sentences.txt --output nt --embeddings embeddings.npy --predicates predicates.csv

              # Read from STDIN (one per line), quiet logs, JSONL
              cat sentences.txt | python nef_cli.py --stdin --no-verbose --output jsonl --embeddings embeddings.npy --predicates predicates.csv

              # Custom Redis and model/threshold
              python nef_cli.py -s "Marie Curie discovered radium" \\
                --redis-host 127.0.0.1 --redis-port 6379 --redis-password secret \\
                --llm-model gemini-2.5-flash --predicate-threshold 0.6 \\
                --embeddings embeddings.npy --predicates predicates.csv
        """))
    sys.exit(main())
