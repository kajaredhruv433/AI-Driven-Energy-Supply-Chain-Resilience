# System Architecture: AI-Driven Energy Supply Chain Resilience

## 1. Architectural Overview and Hybrid Cloud Topology

The AI-Driven Energy Supply Chain Resilience system is architected as an asynchronous, hybrid-cloud decision intelligence platform. It bridges continuous, high-frequency unstructured geopolitical news streams with a deterministic multi-criteria mathematical decision engine to optimize crude oil procurement routes for import-dependent nations.

To achieve enterprise-grade neural processing while maintaining a zero-marginal-cost operating model, the architecture strictly decouples heavy deep learning inference from local deterministic business logic:
- Cloud GPU Inference Tier: Free Google Colab T4 GPU runtime workers host four distinct transformer models exposed via FastAPI web services and secured behind Cloudflare Tunnels.
- Local Orchestration Tier: A lightweight local Python runtime executes RSS feed ingestion, concurrent network polling, dense matrix similarity pruning, relational database state machine transitions, and multi-criteria decision optimization.

```mermaid
flowchart TB
    subgraph Data_Ingestion ["1. Data Ingestion Layer"]
        RSS["Google News RSS Feeds
(8 Targeted Topic Feeds)"]
        NF["news_fetcher.py
(HTML Parsing, Cleaning, Keyword Filtering)"]
        RAW["raw_news.csv
(Initial Article Pool)"]
        RSS --> NF --> RAW
    end

    subgraph Neural_NLP_Pipeline ["2. Neural NLP and Extraction Pipeline (Colab T4 + Cloudflare)"]
        DEDUP["deduplicate.py
(Cosine Similarity >= 0.90)"]
        EMB_SRV["Embedding Server
(sentence-transformers/all-MiniLM-L6-v2)"]
        UNIQ["unique_news.csv"]

        CLASS["classify_news.py
(ThreadPoolExecutor 10 Workers)"]
        CLS_SRV["Zero-Shot Classifier Server
(DeBERTa-v3-base-mnli-fever-anli)"]
        CLS_CSV["classified_news.csv
(Confidence >= 0.45)"]

        SUMM["summarize_news.py
(ThreadPoolExecutor 6 Workers)"]
        SUM_SRV["Summarizer Server
(distilbart-cnn-12-6)"]
        SUM_CSV["summarized_news.csv"]

        EXTR["extract_events.py
(Prompt Injection and JSON Parsing)"]
        LLM_SRV["LLM Brain Server
(Qwen/Qwen2.5-7B-Instruct)"]
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
        UPD["update_all_csv.py
(State Machine and Relational Evaluator)"]
        
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
        API_AV["Alpha Vantage API
(Live Brent Crude Price)"]
        
        SE_OIL["Oil Price Engine
(Differential Calculations)"]
        SE_SHP["Shipping Route Engine
(Freight Cost and Safety Index)"]
        SE_INS["Insurance Engine
(Risk Premium Aggregation)"]
        SE_MCD["Multi-Criteria Decision Engine
(8-Factor Weighted Scoring)"]
        
        OUT_CSV["output/final_supplier_ranking.csv"]
        TERM["Terminal Executive Briefing
and Volumetric Financial Sizing"]

        Permanent_Store --> RD
        API_AV --> RD
        RD --> SE_OIL --> SE_SHP --> SE_INS --> SE_MCD
        SE_MCD --> OUT_CSV
        SE_MCD --> TERM
    end
```

---

## 2. Distributed Client-Server Infrastructure and Sequence Architecture

Network communication between the local orchestrator and the remote GPU workers relies on Cloudflare Tunnel ingress tunnels. This eliminates the need for public IP addresses, port forwarding, or fee-based hosting subscriptions.

```mermaid
sequenceDiagram
    autonumber
    participant Local as Local Pipeline Orchestrator (main.py)
    participant Cloudflare as Cloudflare Tunnel Ingress
    participant Colab as GPU Inference Server (FastAPI / T4 GPU)
    participant LocalDB as Local CSV Intelligence Store

    Note over Local,Colab: Stage 1: Semantic Deduplication
    Local->>Cloudflare: POST /chat (Batch text payload)
    Cloudflare->>Colab: Ingress to all-MiniLM-L6-v2 (Port 8000)
    Colab-->>Cloudflare: 384-dimensional dense vector response
    Cloudflare-->>Local: JSON array of embeddings
    Local->>Local: Compute Cosine Matrix and Prune >= 0.90

    Note over Local,Colab: Stage 2: Concurrent Zero-Shot Classification
    Local->>Cloudflare: Concurrent POST /chat (15 candidate labels)
    Cloudflare->>Colab: Ingress to DeBERTa-v3 (Port 8000)
    Colab-->>Cloudflare: Classification label and probability score
    Cloudflare-->>Local: Filter records where confidence < 0.45

    Note over Local,Colab: Stage 3: Neural Summarization
    Local->>Cloudflare: Concurrent POST /chat (Filtered article bodies)
    Cloudflare->>Colab: Ingress to distilbart-cnn-12-6 (Port 8000)
    Colab-->>Cloudflare: 1-2 sentence concise factual summaries
    Cloudflare-->>Local: Aggregated summary responses

    Note over Local,Colab: Stage 4: Structured Information Extraction
    Local->>Cloudflare: POST /chat (Strict JSON Extraction Prompt)
    Cloudflare->>Colab: Ingress to Qwen2.5-7B-Instruct (Port 8000)
    Colab-->>Cloudflare: Schema-compliant JSON output
    Cloudflare-->>Local: Parsed and validated event parameters

    Note over Local,LocalDB: Stage 5: State Machine Synchronization
    Local->>LocalDB: Update conflict, sanctions, port, chokepoint, and preference tables
    Local->>LocalDB: Run multi-factor preference scoring formula

    Note over Local,LocalDB: Stage 6: Procurement Optimization
    Local->>Local: Execute Oil, Route, Insurance, and Decision sub-engines
    Local->>LocalDB: Persist final rankings to output/final_supplier_ranking.csv
```

---

## 3. NLP Pipeline and Data Flow Architecture

The natural language processing pipeline systematically refines noisy open-source RSS web documents into high-value structured geopolitical crisis vectors.

```mermaid
flowchart LR
    subgraph S1 [Phase 1: Ingestion]
        A[RSS Feed URLs] -->|feedparser| B[HTML Stripping
BeautifulSoup]
        B -->|Regex Normalization| C[Keyword Filter
27 Energy Terms]
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

### Event Extraction JSON Data Contract

The interface between the unstructured news summarizer and `Qwen 2.5-7B-Instruct` enforces a strict JSON schema:

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
  "reason": "IRGC intercepted commercial tanker transiting the Strait of Hormuz citing maritime safety inspections."
}
```

---

## 4. Geopolitical State Machine and Knowledge Graph Architecture

When structured events are output by `extract_events.py`, the state machine engine in `update_all_csv.py` propagates risk adjustments across the permanent database layer.

```mermaid
flowchart TD
    EVT[geopolitical_events.csv] --> DISPATCH{Event Dispatcher}

    subgraph State_Updaters [State Update Modules]
        DISPATCH -->|War and Terror Keywords| U_CNF[update_conflict
Modifies War, Civil War, Terror Risk, Internal Stability]
        DISPATCH -->|Sanction Keywords| U_SNC[update_sanctions
Modifies Sanctioned, Severity, Active]
        DISPATCH -->|Port and Terminal Keywords| U_PRT[update_ports
Modifies Risk, Military Threat, Blocked, Sanctions]
        DISPATCH -->|Chokepoint Keywords| U_CHK[update_chokepoints
Modifies Hormuz, Red Sea, Suez, Bab, Cape Flags]
        DISPATCH -->|Political and Strike Keywords| U_DIP[update_diplomatic
Modifies Political, Export, Government Stability]
        DISPATCH -->|Dynamic Risk Propagation| U_SUP[update_supplier_preference
Updates 6 Numerical Risk Dimensions]
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
        SYNTH --> RES_SUP[(Updated supplier_preference.csv
Preference Score: 0-100)]
    end
```

### State Machine Transition Rules

1. Conflict State Transitions:
- War token detected in event text sets War = Yes.
- Civil token detected sets Civil War = Yes.
- Severity >= 8 sets Internal Stability = Very Low and Terror Risk = Very High.
- Severity >= 6 sets Internal Stability = Low and Terror Risk = High.
- Severity >= 4 sets Internal Stability = Medium and Terror Risk = Medium.

2. Sanction State Transitions:
- Sanction token detected sets Sanctioned = Yes and Active = Yes.
- Severity mapped to Low, Medium, High, or Very High.

3. Port and Terminal Security Transitions:
- Attack, missile, drone, or naval tokens referencing recognized ports elevate Risk and Military Threat levels.
- Blocked or closure tokens set Blocked = Yes.

4. Chokepoint Dependency Mapping:
- Incidents detected in Hormuz, Red Sea, Suez, Bab-el-Mandeb, or Cape Route set corresponding waterway risk flags to Yes.

5. Dynamic Preference Score Synthesis Formula:
The composite preference score for each supplier is computed using the multi-factor formulation:

```
Preference Score = clamp(
    (Base x 0.7)
    + Relation Bonus
    + Spare Capacity Bonus
    + Import Share Bonus
    + Core Exporter Bonus
    + Diversification Bonus
    + Resilience Bonus
    + Special Geopolitical Adjustment
    - (Political Risk x 0.5)
    - (Sanction Risk x 1.5)
    - (Conflict Risk x 0.8)
    - (Port Risk x 0.1)
    - (Chokepoint Risk x 0.1)
    - Chokepoint Penalty
    - Event Penalty,
    0,
    100
)
```

Where:
- Base = Baseline Export Stability from supplier_preference.csv
- Relation Bonus = (Relation with India - 75) x 0.5
- Spare Capacity Bonus = Available spare capacity in mbpd x 6
- Import Share Bonus = India import percentage x 0.2
- Core Exporter Bonus = 15 points for High volume suppliers, 10 points for Medium volume, 3 points for Low/Spot
- Resilience Bonus = +10 points if the exporter can partially bypass Hormuz (such as UAE via Fujairah)
- Diversification Bonus = +10 points if the route transits zero Middle Eastern chokepoints (such as Nigeria, Brazil)
- Special Geopolitical Adjustment = +8 points for UAE (Fujairah strategic bypass), +6 points for Nigeria, -15 points for Iraq (spot cancellation penalties)
- Chokepoint Penalty = -35 points for Hormuz transit, -5 points for Partial Hormuz, -20 points for Red Sea or Bab el-Mandeb transit, -10 points for Partial
- Event Penalty = Sum of severity penalties for active event keywords (such as cargo cancellations, military strikes, refinery damage)

---

## 5. Procurement Decision Engine Mathematical Formulation

The Procurement Decision Engine in `route_decider.py` implements an 8-criteria multi-attribute decision model (MADM) combining live spot crude benchmarks with maritime logistics and insurance pricing.

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
        PR & CD --> ROUTE[Route Builder and Port Assigner]
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

### Mathematical Specifications

#### 1. Route Safety Index
```
Route Safety = clamp(100 - (12 x H) - (10 x RS) - (8 x BM) - (6 x SZ) + (3 x CR), 0, 100)
```
Where:
- H: Hormuz transit (1.0 for Yes, 0.5 for Partial, 0.0 for No)
- RS: Red Sea transit (1.0 for Yes, 0.5 for Partial, 0.0 for No)
- BM: Bab-el-Mandeb transit (1.0 for Yes, 0.5 for Partial, 0.0 for No)
- SZ: Suez Canal transit (1.0 for Yes, 0.5 for Partial, 0.0 for No)
- CR: Cape of Good Hope open ocean transit (1.0 for Yes, 0.0 for No)

#### 2. Risk-Adjusted Insurance Premium
```
Insurance Premium ($/bbl) = Base Insurance (0.80) + C_war + C_civil + C_terror + C_sanction + C_port + C_choke
```
Where:
- Base Insurance = 0.80 dollars per barrel
- C_war = 0.60 dollars if active international war
- C_civil = 0.50 dollars if active civil war
- C_terror evaluates to 0.40 (Very High), 0.30 (High), 0.15 (Medium), 0.05 (Low), or 0.00 dollars
- C_sanction = 0.40 dollars if active sanctions exist
- C_port = 0.40 dollars for blocked port, 0.20 dollars for partially blocked port
- C_choke = (0.30 x H) + (0.25 x RS) + (0.25 x BM) + (0.20 x SZ)

#### 3. Total Delivered Price
```
Delivered Price ($/bbl) = Base Oil Price + Shipping Freight Cost + Insurance Premium
```

#### 4. Multi-Criteria Final Scoring Model
```
Final Score = (0.40 x Preference) + (0.20 x Delivered Price) + (0.15 x Route Safety) + (0.10 x Insurance) + (0.05 x Shipping) + (0.04 x Political) + (0.03 x Export) + (0.03 x Government) - Event Penalty
```

Immediate Event Severity Penalty Schedule:
- Maximum Event Severity >= 5: -20 points deduction
- Maximum Event Severity == 4: -15 points deduction
- Maximum Event Severity == 3: -8 points deduction
- Maximum Event Severity == 2: -4 points deduction
- Maximum Event Severity < 2: 0 points deduction

---

## 6. Maritime Logistics and Global Chokepoint Topology

The maritime logistics framework models global crude carrier transit lanes connecting export terminals across Africa, the Middle East, the Americas, Europe, and Asia-Pacific to the Port of Mumbai (Nhava Sheva / Jawahar Dweep terminal).

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
        HORMUZ{"Strait of Hormuz
(High Risk: -35 pts, +$0.30/bbl)"}
    end

    subgraph Atlantic_African_Corridors ["West Africa and Americas (Cape Route)"]
        BON["Bonny Terminal (Nigeria)"]
        LUA["Luanda (Angola)"]
        SAN["Santos (Brazil)"]
        HOU["Houston (USA)"]
        CAPE{"Cape of Good Hope
(Open Ocean: +3 Route Safety pts)"}
    end

    subgraph European_Mediterranean_Corridors ["Europe and Black Sea (Suez Corridor)"]
        NOV["Novorossiysk / CPC (Russia/Kazakhstan)"]
        MON["Mongstad / Hound Point (Norway/UK)"]
        SUEZ{"Suez Canal and Red Sea
(Vulnerable: -20 pts, +$0.25/bbl)"}
    end

    subgraph Asia_Pacific_Corridors ["Asia-Pacific (Malacca Corridor)"]
        KER["Kertih (Malaysia)"]
        DAM["Dampier (Australia)"]
        MALACCA{"Strait of Malacca
(Moderate Risk)"}
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

### Comprehensive Route and Terminal Mapping Catalog

| Country | Primary Export Terminal | Maritime Route Description | Freight Cost ($/bbl) | Chokepoints Encountered |
| :--- | :--- | :--- | :--- | :--- |
| UAE | Fujairah | Fujairah -> Arabian Sea -> Mumbai | $1.90 | None (Bypasses Hormuz) |
| Oman | Mina Al Fahal | Mina Al Fahal -> Arabian Sea -> Mumbai | $2.00 | None (Open Arabian Sea) |
| Kuwait | Mina Al Ahmadi | Mina Al Ahmadi -> Strait of Hormuz -> Arabian Sea -> Mumbai | $2.00 | Strait of Hormuz |
| Saudi Arabia | Ras Tanura | Ras Tanura -> Strait of Hormuz -> Arabian Sea -> Mumbai | $2.10 | Strait of Hormuz |
| Qatar | Ras Laffan | Ras Laffan -> Strait of Hormuz -> Arabian Sea -> Mumbai | $2.20 | Strait of Hormuz |
| Iraq | Al Basrah | Al Basrah -> Strait of Hormuz -> Arabian Sea -> Mumbai | $2.40 | Strait of Hormuz |
| Malaysia | Kertih | Kertih -> Strait of Malacca -> Mumbai | $3.50 | Strait of Malacca |
| Russia | Novorossiysk | Novorossiysk -> Bosphorus -> Suez -> Mumbai | $4.50 | Bosphorus, Suez Canal, Red Sea |
| Nigeria | Bonny Terminal | Bonny Terminal -> Cape of Good Hope -> Mumbai | $4.80 | Cape of Good Hope |
| Australia | Dampier | Dampier -> Indian Ocean -> Mumbai | $4.90 | Open Indian Ocean |
| Angola | Luanda | Luanda -> Cape of Good Hope -> Mumbai | $5.10 | Cape of Good Hope |
| Brazil | Santos | Santos -> Cape of Good Hope -> Mumbai | $6.00 | Cape of Good Hope |
| Norway | Mongstad | North Sea -> Gibraltar -> Suez -> Mumbai | $6.40 | Gibraltar, Suez, Bab el-Mandeb |
| Mexico | Cayo Arcas | Cayo Arcas -> Gulf of Mexico -> Cape of Good Hope -> Mumbai | $6.50 | Cape of Good Hope |
| Canada | Vancouver | Vancouver -> Pacific Ocean -> Mumbai | $6.80 | Open Pacific Ocean |
| United States | Houston | Houston -> Atlantic -> Cape of Good Hope -> Mumbai | $7.20 | Cape of Good Hope |

---

## 7. Relational Database Schema and Stateful Data Dictionary

The persistence tier stores dynamic and reference data across 10 tabular CSV files:

```
[geopolitical_events.csv] (Event Audit Log)
       |
       +---> [conflict_status.csv] (War, Civil War, Terror)
       +---> [sanctions.csv] (Trade Sanctions and Severity)
       +---> [port_security.csv] (Terminal Military Threats, Blockades)
       +---> [chokepoint_dependency.csv] (Maritime Chokepoint Transit Flags)
       +---> [diplomatic_risk.csv] (Governance and Political Stability)
       |
       v
[supplier_preference.csv] <--- [geopolitical_relation.csv]
       |                  <--- [exporters.csv]
       |                  <--- [supplier_dependency.csv]
       v
[route_decider.py]
       v
[output/final_supplier_ranking.csv]
```

### Table Definitions and Schema Constraints

1. `csv/geopolitical_events.csv`:
- Record Type: Append-only audit log
- Fields: country (String), actor (String), event_type (String), severity (Integer 0-5), political_risk (Integer 0-5), export_risk (Integer 0-5), india_affected (Boolean), affected_exporters (String), affected_ports (String), affected_chokepoints (String), sanction (Boolean), reason (String), title (String), published (String), link (String), summary (String)

2. `csv/supplier_preference.csv`:
- Record Type: Stateful supplier risk index
- Fields: Country (String Primary Key), Preference Score (Integer 0-100), Political Risk (Integer 0-100), Sanction Risk (Integer 0-100), Conflict Risk (Integer 0-100), Export Stability (Integer 0-100), Port Risk (Integer 0-100), Chokepoint Risk (Integer 0-100), Confidence (Float 0.0-1.0)

3. `csv/sanctions.csv`:
- Record Type: Sovereign trade restrictions
- Fields: Country (String Primary Key), Sanctioned (Yes/No), By Whom (String), Severity (Low/Medium/High/Very High), Active (Yes/No)

4. `csv/port_security.csv`:
- Record Type: Marine terminal security index
- Fields: Port (String Primary Key), Country (String), Risk (Low/Medium/High/Very High), Military Threat (Low/Medium/High/Very High), Blocked (Yes/Partial/No), Sanctions (Yes/No)

5. `csv/conflict_status.csv`:
- Record Type: Sovereign armed conflict index
- Fields: Country (String Primary Key), War (Yes/No), Civil War (Yes/No), Internal Stability (Very Low/Low/Medium/High), Terror Risk (Low/Medium/High/Very High)

6. `csv/diplomatic_risk.csv`:
- Record Type: Institutional stability metrics
- Fields: Country (String Primary Key), Political Stability (Very Low to High), Export Stability (Very Low to High), Government Stability (Very Low to High)

7. `csv/chokepoint_dependency.csv`:
- Record Type: Maritime transit topology
- Fields: Country (String Primary Key), Hormuz (Yes/Partial/No), Red Sea (Yes/Partial/No), Suez (Yes/Partial/No), Bab-el-Mandeb (Yes/Partial/No), Cape Route (Yes/No)

8. `csv/geopolitical_relation.csv`:
- Record Type: Diplomatic tie indices
- Fields: Country (String Primary Key), Relation with India (Integer 0-100), Relation with USA (Integer 0-100), Relation with Iran (Integer 0-100), Relation Score (Integer 0-100)

9. `csv/exporters.csv`:
- Record Type: Exporter baseline registry
- Fields: Country (String Primary Key), Export Capacity (Integer bpd), Production (Integer bpd), OPEC (Yes/No), OPEC Plus (Yes/No), Crude Grade (String), Current Export to India (High/Medium/Low/Spot), Strategic Notes (String)

10. `csv/supplier_dependency.csv`:
- Record Type: Import reliance and expansion potential
- Fields: Country (String Primary Key), India Import Percentage (Float), Can Increase Export (Yes/No), Spare Capacity (String mbpd)

---

## 8. Failure Handling and Resilience Mechanisms

| Subsystem | Failure Scenario | Automated Fallback and Recovery Mechanism |
| :--- | :--- | :--- |
| News Fetcher | RSS endpoint failure, timeout, or DNS resolution drop | Ingestion engine catches HTTP exceptions per feed and continues with remaining active endpoints |
| Vector Deduplication | Embedding server timeout or Cloudflare 502/504 gateway error | Fallback preserves original article order, skipping vector pruning to prevent pipeline stoppage |
| Classifier Server | Candidate label parsing error or low confidence scores | Articles failing the 0.45 threshold or generating non-standard labels are safely discarded as noise |
| Neural Summarizer | Summarizer worker exception or rate limiting | Full article body is truncated to the first 250 characters as a fallback brief |
| LLM Event Extraction | Generative LLM produces conversational text without strict JSON | Regular expression parser extracts content matching the JSON curly bracket pattern, stripping markdown fences |
| Oil Price Sub-Engine | Alpha Vantage API quota reached or connection failure | System automatically activates calibrated default Brent spot crude benchmark of 78.40 dollars per barrel |
| Route Decider | Unlisted country in manual port registry | Fallback assigns Direct Ocean Route, default shipping rate of 5.50 dollars, and baseline route safety of 70 |

---

## 9. System Performance, Concurrency, and Scalability

1. Multi-Threaded Model Querying:
- Zero-Shot Classifier: Uses ThreadPoolExecutor with 10 concurrent worker threads to dispatch batches to DeBERTa-v3, completing classification of 40 articles in under 35 seconds.
- Neural Summarizer: Employs ThreadPoolExecutor with 6 worker threads to feed DistilBART, balancing throughput against Colab T4 GPU VRAM saturation.

2. Context Window and Token Optimization:
- Deduplication prunes approximately 25% to 35% of redundant news feeds prior to GPU ingestion.
- Summarization reduces raw news article bodies (averaging 350 words) into compact 30-word summaries, reducing token usage by 91% for downstream Qwen 2.5-7B parsing.

3. Low Latency Local Deterministic Optimization:
- All relational state updates and multi-criteria mathematical evaluations execute locally within 1.2 seconds, ensuring instant decision output once neural extraction finishes.

---

## 10. Authors and Acknowledgments

Lead Developer:
- Dhruv Kajare

Team Members:
- Juhi Nale
- Neha Tarmale

Project Context:
Engineered for the INNOV8 3.0 Hackathon AI Geopolitical Decision Intelligence Initiative exploring autonomous neural intelligence pipelines, distributed zero-cost GPU inference, and multi-criteria mathematical optimization for national energy supply security.
