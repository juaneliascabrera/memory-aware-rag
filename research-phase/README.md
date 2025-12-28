This is a personal research project I've started thanks to Y-Hat event: Investigathon. There, my team and I were challenged to develop, in two weeks, a RAG system for on-devices LLMs. It was such an exciting challenge that I decided to build my own solution based on the same idea. 

## About this project
I'm a Junior Software Developer and I'm taking the first steps in professional topics. This research was started with no knowledge at all, and I've evolved through the weeks its quality, according to the main standards nowadays. 
## What is a RAG?
A **RAG** (Retrieval Augmented Generation) is a technique to extend a LLM knowledge base with one it was not trained with. It may include private data such as company policies, secrets, it could contain real-time data updated frequently, **conversation history** and so forth. 

**How it works?**
Regardless of the technical details, it has two main phases: 
1. **Retrieval**: Given a query $Q$ and a external knowledge base $E$, we search in $E$ the most relevant information to answer $Q$. This stage has many possibly implementations and it's one of the most relevant. 
2. **Augmented Generation**: After we've retrieved the relevant information, we need to **add it** to the query $Q$. Usually, an **augmented prompt** is generated, with a format similar to the following: 
> You're a helpful assistant. Answer the query: **{query}** based on the evidence: **{retrieved_evidence}**

The principal one is to **extend** a LLM capacity to correctly answer queries with data it was not trained with. I selected the benchmark *LongMemEval* which main goal is to evaluate the ability to answer to questions based on a previous conversation history. It has 500 questions with their responses and, for each one, a session history. 

----
## Research format
I've decided to divide the project in two main phases: ***Research*** & **Implementation***
## Research phase
I've decided to start investigating about different **isolated** techniques, and, complementing my Investigathon project, I decided to start with **BM25**. 
#### BM25
BM25 (Best Match 25) is a probabilistic retrieval technique based on **keywords**, term frequency and inverse document frequency, it's the evolved version of different original methods such as [**TF-IDF**](https://en.wikipedia.org/wiki/Tf%E2%80%93idf).
Its main goal is to analyze a **list** of documents and rank them based on a criteria, broadly speaking, it takes into account factors such as how often a word appears, the length of the document, saturation, and similar considerations.

I will be using standard metrics nowadays such as [**Recall@K** & **Precision@K**.](https://www.evidentlyai.com/ranking-metrics/precision-recall-at-k). 

### First iteration
These were the first results, taking **k = 5**. 

| Question Type             | Recall@5 | Precision@5 | Total Count |
| ------------------------- | -------- | ----------- | ----------- |
| Abstention                | 84.44%   | 32.00%      | 30          |
| Knowledge-Update          | 98.61%   | 39.44%      | 72          |
| Multi-session             | 84.85%   | 42.81%      | 121         |
| Single-session-assistant  | 98.21%   | 19.64%      | 56          |
| Single-session-preference | 80.00%   | 16.00%      | 30          |
| Single-session-user       | 100%     | 20.00%      | 64          |
| Temporal reasoning        | 88.40%   | 37.80%      | 127         |
| **GLOBAL RESULTS**        | 90.85%   | 33.28%      | 500         |
Let's analyze these first results: The BM25 system is extraordinarily useful for the 'single-session-user/assistant' questions because those depends only on one session, and it's easy to find the most relevant session if it has a directly lexical relationship with the query. 
Obviously, taking and arbitrary fixed $k = 5$, in these kind of questions the **Precision@5** will be 20%. 

Single-session-preference questions are the ones with less **Recall@5** score, and it could be explained by understanding how these questions works: Normally, these questions requires to find a **semantic relationship** between the query and the sessions, to answer the user preferences, and BM25 it's not quite good with this kind of relationships because it's focused on **keywords**. 

### Second iteration
Exactly the same system, k = 4.

| Question Type             | Recall@4 | Precision@4 | Total Count |
| ------------------------- | -------- | ----------- | ----------- |
| Abstention                | 82.78%   | 39.17%      | 30          |
| Knowledge-Update          | 97.22%   | 48.61%      | 72          |
| Multi-session             | 81.18%   | 50.41%      | 121         |
| Single-session-assistant  | 98.21%   | 24.55%      | 56          |
| Single-session-preference | 76.67%   | 19.17%      | 30          |
| Single-session-user       | 100%     | 25.00%      | 64          |
| Temporal reasoning        | 86.33%   | 45.28%      | 127         |
| **GLOBAL RESULTS**        | 88.94%   | 40.15%      | 500         |
The most relevant, and expected, conclusion is the **Precision@4** improve. Of course, taking fewer sessions reduces the noise, trading off the **Recall@4** which had a 1.91% loss. 

Something important I noticed here: The single-session-user/assistant questions got the exactly same value for **Recall@4 and 5**, meaning that the most relevant document is never, in these kind of questions, in the last place. I investigated about some metric to measure this and I found [MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank). 

The same experiment with **k = 3** showed an even stronger loss in **Recall@K** (it went to 85.82%), however, the **Precision@K** raised to 51.00%. 

## Third iteration
After these first results and some extra research, I've decided to use a **Re-Ranking** technique: this works by given a query and and a list of relevant documents to the query, re-ranks its by comparing the relevance of each document to the query. 

Considering that this project is aimed to find a lightweight implementation, I've decided to use one of the cross encoder with best size/precision ratio: [ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2).

These were the results:

| K         | Recall@K | Precision@K |
| --------- | -------- | ----------- |
| K = 3     | 91.88%   | 55.53%      |
| **K = 4** | 93.88%   | 43.35%      |
| **K = 5** | 94.78%   | 35.44%      |

We can observe an improvement in every k, just by adding the cross-encoder phase. (For 80-90 mb in space and a 6-layer system, is an extremely positive trade-off). 

Simultaneously, I was finding the floor 'optimal k', by analyzing in the benchmark how many questions required k = 1,2,3,4,5. Only a 4.95% of the questions requires 5 sessions so I've decided to use k = 4, which encompasses the 95.05% of the questions. 

Also, I was researching about a *dynamic threshold*. It 'cuts off' the relevant document which less than a given ranking value. After testing all the threshold between 1.0 and 10.0, to k = 5 and k = 4, $
t = 6.0$ is the optimal, and for k = 3, $t = 6.5$. 

## Fourth iteration
I ran an experiment with BM25 + Cross-Encoder + dynamic threshold. These were the results: 
K = 4, Recall@4 = 89.34%, Precision@K = 85.93%, MRR = 92.99%

Those were nice results! We achieved a 40% improvement in Precision@K by using a threshold trading off a 4% in Recall@K. 

The next and final step, embeddings. 

## Fifth and last iteration.
I decided to use embeddings because the single-session-preferences question were always the ones with less Recall@K. This is, as a I already explained, because BM25 can not find documents by semantic relationship, and these kind of questions need them more than any other. 

So, I used a lightweight embedding model: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

At a high level, an embedding maps a 'string' to a vector (384 dimensions in this model) and then we can search in this vector database to relevant semantic data by looking to vectorial similarity such as cosine similarity or dot product. 

I've decided to embed the messages individually. This means: The embedding will be of the message of each session, then, we find in this database the relevant messages and finally we get the sessions of that messages. 

Of course, this 'overrides' BM25, but after so much research, I didn't want to do this, so I study about 'Hybrid-Search'. It's amazing.

Given a query, you launch two simultaneously searchs: BM25 and Embeddings. Then, you get, for example, 20 sessions from each ranking (of course, there will be repeated sessions between both rankings) and then you can use a [Reciprocal Rank Fusion](https://medium.com/@devalshah1619/mathematical-intuition-behind-reciprocal-rank-fusion-rrf-explained-in-2-mins-002df0cc5e2a) technique to merge these rankings.

I've decided to use this kind of embed: 
> "TimeStamp Role: Content", an example would be:
> 21st June, 2025 User: 'I would love to explore new cities'

This finally incorporates the date, addressing a recurrent issue in the previous system where temporal data was completely ignored. I decided this format because it'll help the LLM to know **who**, **when** and **what**.

These were the final results previous to the implementation stage: 
Hybrid Search (BM25 + Embeddings) + ReRanking (Top 50) + Threshold 6.0 and K = 4.

| Question Type             | Recall@4 | Precision@4 | MRR    | Total Count |
| ------------------------- | -------- | ----------- | ------ | ----------- |
| Abstention                | 88.06%   | 79.44%      | N/A    | 30          |
| Knowledge-Update          | 95.14%   | 96.76%      | N/A    | 72          |
| Multi-session             | 85.11%   | 89.46%      | N/A    | 121         |
| Single-session-assistant  | 100%     | 94.64%      | 100%   | 56          |
| Single-session-preference | 90.00%   | 36.94%      | 73.89% | 30          |
| Single-session-user       | 100%     | 95.70%      | 99.22% | 64          |
| Temporal reasoning        | 86.36%   | 81.76%      | 91.25% | 127         |
| **GLOBAL RESULTS**        | 90.92%   | 86.18%      | 93.89% | 500         |
I was very proud at this point of all the process, and I decided, with this results, to start the implementation stage. 

Thanks to all the websites referenced, documents, information, Y-Hat, and the life. 