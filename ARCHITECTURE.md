# System Architecture: AI-Driven Energy Supply Chain Resilience

## 1. Architectural Overview

The AI-Driven Energy Supply Chain Resilience system is engineered as a hybrid distributed decision intelligence platform. It ingests high-frequency unstructured geopolitical news feeds, passes them through a four-stage neural natural language processing (NLP) pipeline, updates a relational state machine of global energy suppliers and maritime infrastructure, and computes an optimal oil procurement strategy using a multi-criteria mathematical optimization engine.

The platform separates compute-heavy deep learning inference from local deterministic business logic through an asynchronous hybrid cloud topology.

```mermaid
flowchart TB
    subgraph Data_Ingestion ["1. Data Ingestion Layer"]
        RSS["Google News RSS Feeds\n(8 Targeted Topic Feeds)"]
        NF["news_fetcher.py\n(HTML Parsing, Cleaning, Keyword Filtering)"]
        RAW["raw_news.csv\n(Initial Article Pool)"]
        RSS --> NF --> RAW
    end

    subgraph Neural_NLP_Pipeline ["2. Neural NLP & Extraction Pipeline (Colab T4 + Cloudflare)"]
        DEDUP["deduplicate.py\n(Cosine Similarity >= 0.90)"]
        EMB_SRV["Embedding Server\n(sentence-transformers/all-MiniLM-L6-v2)"]
        UNIQ["unique_news.csv"]

        CLASS["classify_news.py\n(ThreadPoolExecutor 10 Workers)"]
        CLS_SRV["Zero-Shot Classifier Server\n(DeBERTa-v3-base-mnli-fever-anli)"]
        CLS_CSV["classified_news.csv\n(Confidence >= 0.45)"]

        SUMM["summarize_news.py\n(ThreadPoolExecutor 6 Workers)"]
        SUM_SRV["Summarizer Server\n(distilbart-cnn-12-6)"]
        SUM_CSV["summarized_news.csv"]

        EXTR["extract_events.py\n(Prompt Injection & JSON Parsing)"]
        LLM_SRV["LLM Brain Server\n(Qwen/Qwen2.5-7B-Instruct)"]
        GEO_EVT["csv/geopolitical_events.csv"]

        RAW --> DEDUP
        DEDUP <-->|REST API| EMB_SRV
        DEDUP --> UNIQ --> CLASS
        CLASS <-->|REST API| CLS_SRV
        CLASS --> CLS_CSV --> SUMM
        SUMM <-->|REST API| SUM_SRV
        SUMM --> SUM_CSV --> EXTR
        EXTR <-->|REST API| LLM_SRV
        EXTR --> GEO_EVT
    end

    subgraph Intelligence_State_Machine ["3. Dynamic Intelligence Database Layer"]
        UPD["update_all_csv.py\n(State Machine & Relational Evaluator)"]
        
        subgraph Permanent_Store ["10 Relational Knowledge Tables (csv/)"]
            T_SUP["supplier_preference.csv"]
            T_SAN["sanctions.csv"]
            T_PRT["port_security.csv"]
            T_CNF["conflict_status.csv"]
            T_DIP["diplomatic_risk.csv"]
            T_CHK["chokepoint_dependency.csv"]
            T_REL["geopolitical_relation.csv"]
            T_EXP["exporters.csv"]
            T_DEP["supplier_dependency.csv"]
        end

        GEO_EVT --> UPD
        UPD <--> Permanent_Store
    end

    subgraph Decision_Optimization_Engine ["4. Multi-Criteria Procurement Decision Engine"]
        RD["route_decider.py"]
        API_AV["Alpha Vantage API\n(Live Brent Crude Price)"]
        
        SE_OIL["Oil Price Engine\n(Differential Calculations)"]
        SE_SHP["Shipping Route Engine\n(Freight Cost & Safety Index)"]
        SE_INS["Insurance Engine\n(Risk Premium Aggregation)"]
        SE_MCD["Multi-Criteria Decision Engine\n(8-Factor Weighted Scoring)"]
        
        OUT_CSV["output/final_supplier_ranking.csv"]
        TERM["Terminal Executive Briefing\n& Volumetric Financial Sizing"]

        Permanent_Store --> RD
        API_AV --> RD
        RD --> SE_OIL --> SE_SHP --> SE_INS --> SE_MCD
        SE_MCD --> OUT_CSV
        SE_MCD --> TERM
    end
```

---

## 2. Distributed Hybrid Client-Server Infrastructure

Due to the compute requirements of hosting multiple transformer models (ranging from lightweight embedding models to 7-billion parameter generative LLMs), the architecture adopts a zero-marginal-cost decoupled design. The heavy inference tasks run on cloud GPU worker instances (such as Google Colab T4 runtimes), exposed securely via Cloudflare Tunnels (`trycloudflare.com`). The local orchestrator executes lightweight data manipulation, concurrent request pooling, state updates, and deterministic optimization.

```mermaid
sequenceDiagram
    autonumber
    participant Local as Local Pipeline Orchestrator (main.py)
    participant Cloudflare as Cloudflare Tunnel Ingress
    participant Colab as GPU Inference Server (FastAPI / T4 GPU)
    participant LocalDB as Local CSV Intelligence Store

    Note over Local,Colab: Step 1: Semantic Deduplication
    Local->>Cloudflare: POST /chat (Batch text payload)
    Cloudflare->>Colab: Ingress to all-MiniLM-L6-v2 (Port 8000)
    Colab-->>Cloudflare: 384-d dense vector response
    Cloudflare-->>Local: JSON array of embeddings
    Local->>Local: Compute Cosine Matrix & Prune >= 0.90

    Note over Local,Colab: Step 2: Concurrent Zero-Shot Classification
    Local->>Cloudflare: Concurrent POST /chat (15 candidate labels)
    Cloudflare->>Colab: Ingress to DeBERTa-v3 (Port 8000)
    Colab-->>Cloudflare: Classification label + probability score
    Cloudflare-->>Local: Filter records where confidence < 0.45

    Note over Local,Colab: Step 3: Neural Summarization
    Local->>Cloudflare: Concurrent POST /chat (Filtered article bodies)
    Cloudflare->>Colab: Ingress to distilbart-cnn-12-6 (Port 8000)
    Colab-->>Cloudflare: 1-2 sentence concise factual summaries
    Cloudflare-->>Local: Aggregated summary responses

    Note over Local,Colab: Step 4: Structured Information Extraction
    Local->>Cloudflare: POST /chat (Strict JSON Extraction Prompt)
    Cloudflare->>Colab: Ingress to Qwen2.5-7B-Instruct (Port 8000)
    Colab-->>Cloudflare: Raw JSON output
    Cloudflare-->>Local: Parsed and validated event parameters

    Note over Local,LocalDB: Step 5: State Machine Synchronization
    Local->>LocalDB: Update conflict, sanctions, port, chokepoint, and preference tables
    Local->>LocalDB: Run multi-factor preference scoring formula

    Note over Local,LocalDB: Step 6: Procurement Optimization
    Local->>Local: Execute Oil, Route, Insurance, and Decision sub-engines
    Local->>LocalDB: Persist final rankings to output/final_supplier_ranking.csv
```

---

## 3. NLP Pipeline & Data Flow Architecture

The data transformation pipeline converts noisy external RSS articles into clean, structured geopolitical facts with high precision and low token overhead.

```mermaid
flowchart LR
    subgraph S1 [Phase 1: Ingestion]
        A[RSS Feed URLs] -->|feedparser| B[HTML Stripping\nBeautifulSoup]
        B -->|Regex Normalization| C[Keyword Filter\n27 Energy Terms]
        C -->|raw_news.csv| D[Raw Articles]
    end

    subgraph S2 [Phase 2: Deduplication]
        D -->|Vector Generation| E[all-MiniLM-L6-v2]
        E -->|Cosine Similarity Matrix| F{Score >= 0.90?}
        F -->|Yes| G[Keep Longest Article]
        F -->|No| H[Retain Both]
        G & H -->|unique_news.csv| I[Deduplicated Stream]
    end

    subgraph S3 [Phase 3: Classification]
        I -->|Parallel Workers: 10| J[DeBERTa-v3 Classifier]
        J -->|15 Geopolitical Classes| K{Confidence >= 0.45?}
        K -->|No| L[Discard Noise]
        K -->|Yes| M[classified_news.csv]
    end

    subgraph S4 [Phase 4: Summarization]
        M -->|Parallel Workers: 6| N[distilbart-cnn-12-6]
        N -->|Factual Compression| O[summarized_news.csv]
    end

    subgraph S5 [Phase 5: Extraction]
        O -->|System Prompt Template| P[Qwen2.5-7B-Instruct]
        P -->|Regex Sanitization| Q[JSON Validation Engine]
        Q -->|geopolitical_events.csv| R[Structured Event Stream]
    end

    S1 --> S2 --> S3 --> S4 --> S5
```

### Event Extraction JSON Contract

The interface contract between the unstructured news summarizer and `Qwen 2.5-7B` requires the model to output a schema-compliant JSON document:

```json
{
  "country": "Iran",
  "actor": "IRGC Navy",
  "event_type": "Shipping Attack",
  "severity": 4,
  "political_risk": 4,
  "export_risk": 5,
  "india_affected": true,
  "affected_exporters": ["Iraq", "Kuwait", "Saudi Arabia", "UAE", "Qatar"],
  "affected_ports": ["Kharg Island", "Ras Tanura", "Al Basrah"],
  "affected_chokepoints": ["Strait of Hormuz"],
  "sanction": false,
  "reason": "IRGC seized foreign commercial tanker transiting the Strait of Hormuz claiming maritime violations."
}
```

---

## 4. Geopolitical State Machine & Knowledge Graph Architecture

When new events are extracted, `update_all_csv.py` acts as a state machine that propagates risk adjustments across the permanent database tables.

```mermaid
flowchart TD
    EVT[geopolitical_events.csv] --> DISPATCH{Event Dispatcher}

    subgraph State_Updaters [State Update Modules]
        DISPATCH -->|War / Terror Keywords| U_CNF[update_conflict\nModifies War, Civil War, Terror Risk, Internal Stability]
        DISPATCH -->|Sanction Keywords| U_SNC[update_sanctions\nModifies Sanctioned, Severity, Active]
        DISPATCH -->|Port / Terminal Keywords| U_PRT[update_ports\nModifies Risk, Military Threat, Blocked, Sanctions]
        DISPATCH -->|Chokepoint Keywords| U_CHK[update_chokepoints\nModifies Hormuz, Red Sea, Suez, Bab, Cape Flags]
        DISPATCH -->|Political / Strike Keywords| U_DIP[update_diplomatic\nModifies Political, Export, Government Stability]
        DISPATCH -->|Dynamic Risk Propagation| U_SUP[update_supplier_preference\nUpdates 6 Numerical Risk Dimensions]
    end

    U_CNF --> DB_CNF[(conflict_status.csv)]
    U_SNC --> DB_SNC[(sanctions.csv)]
    U_PRT --> DB_PRT[(port_security.csv)]
    U_CHK --> DB_CHK[(chokepoint_dependency.csv)]
    U_DIP --> DB_DIP[(diplomatic_risk.csv)]
    U_SUP --> DB_SUP[(supplier_preference.csv)]

    subgraph Score_Synthesizer [Preference Synthesizer: calculate_preference_score]
        DB_CNF & DB_SNC & DB_PRT & DB_CHK & DB_DIP & DB_SUP --> SYNTH[Multi-Factor Mathematical Formulator]
        DB_REL[(geopolitical_relation.csv)] --> SYNTH
        DB_EXP[(exporters.csv)] --> SYNTH
        DB_DEP[(supplier_dependency.csv)] --> SYNTH
        SYNTH --> RES_SUP[(Updated supplier_preference.csv\nPreference Score: 0-100)]
    end
```

### State Machine Transition Rules

1. **Conflict Escalation**:
   - `War` token detected in event text $\rightarrow$ `War = Yes`.
   - `Civil` token detected $\rightarrow$ `Civil War = Yes`.
   - Severity $\ge 8 \rightarrow \text{Internal Stability} = \text{Very Low}$, $\text{Terror Risk} = \text{Very High}$.
   - Severity $\ge 6 \rightarrow \text{Internal Stability} = \text{Low}$, $\text{Terror Risk} = \text{High}$.
   - Severity $\ge 4 \rightarrow \text{Internal Stability} = \text{Medium}$, $\text{Terror Risk} = \text{Medium}$.

2. **Sanction Tracking**:
   - `Sanction` token detected $\rightarrow$ `Sanctioned = Yes`, `Active = Yes`.
   - Severity mapped to `Low`, `Medium`, `High`, or `Very High`.

3. **Port & Terminal Security**:
   - Port/Terminal tokens with attack/missile/drone/naval keywords trigger immediate elevation of `Risk` and `Military Threat`.
   - `Blocked` or `Closure` tokens $\rightarrow$ `Blocked = Yes`.

4. **Chokepoint Dependency Mapping**:
   - Detected incidents in `Hormuz`, `Red Sea`, `Suez`, `Bab-el-Mandeb`, or `Cape Route` set corresponding maritime dependency flags to `Yes`.

---

## 5. Decision Engine Mathematical Formulation

The Procurement Decision Engine in `route_decider.py` implements an 8-criteria multi-attribute decision model (MADM) integrated with live commodity spot prices and maritime freight routing equations.

```mermaid
flowchart TD
    subgraph Data_Inputs [Input Vectors]
        SP[(supplier_preference.csv)]
        PR[(port_security.csv)]
        CD[(chokepoint_dependency.csv)]
        CS[(conflict_status.csv)]
        SN[(sanctions.csv)]
        DP[(diplomatic_risk.csv)]
        EV[(geopolitical_events.csv)]
        AP[Alpha Vantage API: Live Brent Price]
    end

    subgraph Sub_Engine_1 [1. Oil Price Engine]
        AP --> BRENT[Live Brent Price]
        BRENT --> DIFF[Price Differential per Exporter]
        DIFF --> BOP["Base Oil Price = Brent + Diff"]
        BOP --> N_BOP["Price Score = Normalize_Inv(Base Oil Price)"]
    end

    subgraph Sub_Engine_2 [2. Shipping Route Engine]
        PR & CD --> ROUTE[Route Builder & Port Assigner]
        ROUTE --> FREIGHT[Freight Cost Table lookup]
        FREIGHT --> N_FRT["Shipping Score = Normalize_Inv(Shipping Cost)"]
        CD --> R_SAFE["Route Safety Score = 100 - Sum(Chokepoint Penalties) + Cape Bonus"]
    end

    subgraph Sub_Engine_3 [3. Insurance Engine]
        CS & SN & PR & CD --> INS_CALC["Insurance Premium = Base ($0.80) + War + Civil War + Terror + Sanctions + Port Blockade + Chokepoint Risk"]
        INS_CALC --> N_INS["Insurance Score = Normalize_Inv(Insurance Premium)"]
        BOP & FREIGHT & INS_CALC --> DELIV["Delivered Price = Base Oil Price + Shipping Cost + Insurance Premium"]
        DELIV --> N_DEL["Delivered Price Score = Normalize_Inv(Delivered Price)"]
    end

    subgraph Sub_Engine_4 [4. Weighted Decision Model]
        DP --> STAB["Stability Scores (Political, Export, Government x 20)"]
        EV --> EV_PEN["Event Severity Penalty Deduction (0 to -20 pts)"]
        
        SP & N_DEL & R_SAFE & N_INS & N_FRT & STAB & EV_PEN --> WEIGHT_CALC["Final Score = 0.40(Pref) + 0.20(Price) + 0.15(Route) + 0.10(Ins) + 0.05(Ship) + 0.04(Pol) + 0.03(Exp) + 0.03(Gov) - Event Penalty"]
        WEIGHT_CALC --> RANK[Rank Suppliers Descending]
        RANK --> OUT[output/final_supplier_ranking.csv]
        RANK --> TOP[Top Supplier Recommendation]
    end
```

### Mathematical Equations

#### 1. Route Safety Formulation
$$\text{Route Safety} = \text{Clamp}_{0}^{100}\left(100 - 12(H) - 10(RS) - 8(BM) - 6(SZ) + 3(CR)\right)$$
Where:
- $H$: Hormuz dependency ($1.0$ for Yes, $0.5$ for Partial, $0.0$ for No)
- $RS$: Red Sea dependency ($1.0$ for Yes, $0.5$ for Partial, $0.0$ for No)
- $BM$: Bab-el-Mandeb dependency ($1.0$ for Yes, $0.5$ for Partial, $0.0$ for No)
- $SZ$: Suez Canal dependency ($1.0$ for Yes, $0.5$ for Partial, $0.0$ for No)
- $CR$: Cape of Good Hope open ocean transit ($1.0$ for Yes, $0.5$ for Partial, $0.0$ for No)

#### 2. Insurance Premium Formulation
$$\text{Insurance Premium (\$/bbl)} = P_{\text{base}} + C_{\text{war}} + C_{\text{civil}} + C_{\text{terror}} + C_{\text{sanction}} + C_{\text{port}} + C_{\text{choke}}$$
Where:
- $P_{\text{base}} = \$0.80$
- $C_{\text{war}} = \$0.60$ if active war
- $C_{\text{civil}} = \$0.50$ if active civil war
- $C_{\text{terror}} \in \{\$0.40, \$0.30, \$0.15, \$0.05, \$0.00\}$ based on threat level
- $C_{\text{sanction}} = \$0.40$ if active sanctions
- $C_{\text{port}} = \$0.40$ if port blocked, $\$0.20$ if partially blocked
- $C_{\text{choke}} = 0.30(H) + 0.25(RS) + 0.25(BM) + 0.20(SZ)$

#### 3. Total Delivered Price
$$\text{Delivered Price (\$/bbl)} = \text{Base Oil Price} + \text{Shipping Freight Cost} + \text{Insurance Premium}$$

#### 4. Multi-Criteria Final Scoring
$$\text{Final Score} = \sum_{i=1}^{8} \left(w_i \times S_i\right) - \Delta_{\text{event}}$$
Weights ($w_i$):
- $\text{Preference Score}: 40\%$
- $\text{Delivered Price Score}: 20\%$
- $\text{Route Safety Score}: 15\%$
- $\text{Insurance Score}: 10\%$
- $\text{Shipping Score}: 5\%$
- $\text{Political Stability Score}: 4\%$
- $\text{Export Stability Score}: 3\%$
- $\text{Government Stability Score}: 3\%$

Event Deduction ($\Delta_{\text{event}}$):
- Maximum Event Severity $\ge 5 \implies -20\text{ points}$
- Maximum Event Severity $= 4 \implies -15\text{ points}$
- Maximum Event Severity $= 3 \implies -8\text{ points}$
- Maximum Event Severity $= 2 \implies -4\text{ points}$
- Maximum Event Severity $< 2 \implies 0\text{ points}$

---

## 6. Maritime Logistics & Chokepoint Topology

The maritime model evaluates global shipping lanes connecting export terminals across Africa, the Middle East, the Americas, Europe, and Asia-Pacific to the Port of Mumbai (Nhava Sheva / Jawahar Dweep terminal).

```mermaid
flowchart TD
    subgraph Middle_East_Corridors ["Middle East Export Terminals"]
        RT["Ras Tanura (Saudi Arabia)"]
        AB["Al Basrah (Iraq)"]
        MA["Mina Al Ahmadi (Kuwait)"]
        RL["Ras Laffan (Qatar)"]
        KI["Kharg Island (Iran)"]
        FUJ["Fujairah (UAE - Hormuz Bypass)"]
        MAF["Mina Al Fahal (Oman)"]
    end

    subgraph Chokepoint_Hormuz ["Critical Chokepoint: Strait of Hormuz"]
        HORMUZ{"Strait of Hormuz\n(High Risk: -35 pts, +$0.30/bbl)"}
    end

    subgraph Atlantic_African_Corridors ["West Africa & Americas (Cape Route)"]
        BON["Bonny Terminal (Nigeria)"]
        LUA["Luanda (Angola)"]
        SAN["Santos (Brazil)"]
        HOU["Houston (USA)"]
        CAPE{"Cape of Good Hope\n(Open Ocean: +3 Route Safety pts)"}
    end

    subgraph European_Mediterranean_Corridors ["Europe & Black Sea (Suez Corridor)"]
        NOV["Novorossiysk / CPC (Russia/Kazakhstan)"]
        MON["Mongstad / Hound Point (Norway/UK)"]
        SUEZ{"Suez Canal & Red Sea\n(Vulnerable: -20 pts, +$0.25/bbl)"}
    end

    subgraph Asia_Pacific_Corridors ["Asia-Pacific (Malacca Corridor)"]
        KER["Kertih (Malaysia)"]
        DAM["Dampier (Australia)"]
        MALACCA{"Strait of Malacca\n(Moderate Risk)"}
    end

    subgraph Destination_Terminal ["Import Destination"]
        MUMBAI["Port of Mumbai (India)"]
    end

    RT & AB & MA & RL & KI --> HORMUZ --> ARB["Arabian Sea"] --> MUMBAI
    FUJ & MAF --> ARB --> MUMBAI
    BON & LUA & SAN & HOU --> CAPE --> IND_OCN["Indian Ocean"] --> MUMBAI
    NOV & MON --> SUEZ --> ARB --> MUMBAI
    KER & DAM --> MALACCA --> IND_OCN --> MUMBAI
```

---

## 7. Failure Handling & Resilience Mechanisms

| Layer | Potential Failure | Handling & Mitigation Strategy |
| :--- | :--- | :--- |
| **Ingestion** | RSS endpoint unavailable or connection timeout | `feedparser` ignores unreachable endpoints; pipeline proceeds with available feeds. |
| **Deduplication** | Embedding server timeout or Cloudflare error | Falls back to preserving original record order without crashing downstream tasks. |
| **Classification** | Zero-shot model outputs invalid JSON or low score | Output is sanitized using regex and filtered out if score is below the 0.45 threshold. |
| **Summarization** | DistilBART server rate limit or worker error | Unsummarized article text is passed through fallback truncation to prevent data loss. |
| **LLM Brain** | Generative model outputs non-JSON markdown wrapper | `re.search(r"\{.*\}", re.DOTALL)` extracts embedded JSON, stripping markdown code fences. |
| **Oil Pricing** | Alpha Vantage API quota exhausted or network error | Automatically falls back to calibrated default spot benchmark ($78.40/barrel). |
| **Decision Engine** | Missing exporter record in diplomatic or port tables | Default baseline safety parameters are assigned to prevent NaN propagation. |
