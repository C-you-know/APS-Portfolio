import React from 'react';
import { createRoot } from 'react-dom/client';

interface Feature {
  id: string;
  title: string;
  description: string;
}

const llmFeatures: Feature[] = [
  {
    id: 'storage',
    title: 'Efficient Storage for Previous Responses',
    description: 'Scalable and fast-access storage solutions are crucial for caching LLM responses, enabling quicker retrieval for repeated queries and fine-tuning processes. Algorithmic approaches can optimize storage structures and retrieval patterns.',
  },
  {
    id: 'distributed',
    title: 'Distributed Inferencing',
    description: 'Large Language Models often require significant computational power. Distributing inference tasks across multiple nodes or devices enhances throughput and reduces latency. This involves complex load distribution and synchronization algorithms.',
  },
  {
    id: 'latency',
    title: 'Ultra-Low Latency',
    description: 'For real-time applications, minimizing inference latency is paramount. This can be achieved through model optimization, hardware acceleration, and efficient request queuing and processing algorithms.',
  },
  {
    id: 'loadbalancing',
    title: 'Intelligent Load Balancing',
    description: 'Effectively distributing incoming inference requests across available resources prevents bottlenecks and ensures optimal utilization. Algorithmic load balancers consider factors like server load, data locality, and request priority.',
  },
  {
    id: 'multimodal',
    title: 'Multi-Modality Support',
    description: 'Modern LLMs increasingly handle diverse data types (text, images, audio). Infrastructure must support efficient processing and fusion of multiple modalities, requiring sophisticated data handling and routing algorithms.',
  },
];

// --- Data for Algorithmic Deep Dive ---
interface AlgorithmSnippet {
  name: string;
  purpose: string;
  snippet?: string;
  snippetLang?: string;
}

interface DeepDiveCategory {
  id: string;
  title: string;
  summary: string;
  techniques: AlgorithmSnippet[];
}

const algorithmicDeepDiveData: DeepDiveCategory[] = [
  {
    id: 'multiUserIsolation',
    title: '1. Multi-User State Isolation & Privacy',
    summary: 'Ensuring user A never sees user B’s data, even in distributed compute, long-lived sessions, and logs.',
    techniques: [
      { name: 'Merkle Trees (for Session Isolation)', purpose: 'Verifiable state integrity. Each user session or data segment can have its state değişiklikleri cryptographically hashed and combined into a tree, allowing efficient verification of data integrity without exposing all data.', snippet: `class MerkleNode {
  constructor(data) {
    this.data = data;
    this.hash = calculateHash(data.toString()); // Simplified
    this.left = null;
    this.right = null;
  }
}
// const root = buildMerkleTree(userSessionData);`, snippetLang: 'javascript' },
      { name: 'Hash-based Sharding (for Storage Mapping)', purpose: 'Fast lookup & scaling by distributing user data across N storage shards. `shard_id = hash(user_id) % N_shards;`', snippet: `const N_SHARDS = 16;
function getShardId(userId) {
  // A simple numeric hash for illustration
  let hash = 0;
  for (let i = 0; i < userId.length; i++) {
    hash = (hash << 5) - hash + userId.charCodeAt(i);
    hash |= 0; // Convert to 32bit integer
  }
  return Math.abs(hash) % N_SHARDS;
}
// const shard = getShardId('user123');`, snippetLang: 'javascript' },
      { name: 'Bloom Filters + Tombstones (for Deletion Requests)', purpose: 'Fast deletion checks in logs. Bloom filter quickly checks if an item *might* be deleted; tombstone confirms actual deletion.', snippet: `// Conceptual
bloomFilter.add(deletedItemId);
function isLikelyDeleted(itemId) {
  return bloomFilter.mightContain(itemId);
}
// If true, check tombstone_store[itemId]`, snippetLang: 'pseudocode' },
    ],
  },
  {
    id: 'networkRouting',
    title: '2. Efficient Network Routing & Request Dispatch',
    summary: 'Directing billions of user requests to the appropriate model, data center, or GPU.',
    techniques: [
      { name: 'Consistent Hashing (for Load Balancing)', purpose: 'Sticky sessions without major rebalancing when nodes are added/removed. Maps requests to nodes on a virtual ring.', snippet: `// Conceptual: Assign request to a node on a hash ring
function getNodeForRequest(requestId, nodes) {
  const requestHash = hash(requestId);
  // Find first node on ring with hash >= requestHash
  return nodes.sort((a,b) => a.hash - b.hash)
              .find(node => node.hash >= requestHash) || nodes[0];
}`, snippetLang: 'javascript' },
      { name: 'Token Bucket (for Routing Overload Protection)', purpose: 'Prevents GPU queue explosions by rate-limiting requests.', snippet: `let tokens = MAX_TOKENS;
let lastRefillTime = Date.now();

function allowRequest() {
  const now = Date.now();
  const elapsed = now - lastRefillTime;
  tokens += elapsed * REFILL_RATE;
  tokens = Math.min(tokens, MAX_TOKENS);
  lastRefillTime = now;

  if (tokens >= 1) {
    tokens -= 1;
    return true; // Allow
  }
  return false; // Deny or queue
}`, snippetLang: 'javascript' },
    ],
  },
  {
    id: 'promptStorageRetrieval',
    title: '3. Storing & Retrieving Prompts/Responses (History)',
    summary: 'Managing chat history, resumable sessions, and real-time collaboration efficiently and securely.',
    techniques: [
      { name: 'Inverted Index + LSH (for Search Past Prompts)', purpose: 'Enables fast substring or semantic search in prompt history. LSH helps find similar prompts.', snippet: `// Inverted Index Structure
const invertedIndex = {
  "hello": ["doc1_id", "doc3_id"],
  "world": ["doc2_id", "doc3_id"]
};
// LSH: findSimilar(hash(prompt), k)`, snippetLang: 'json' },
      { name: 'SimHash / MinHash + LSH (for Deduplication)', purpose: 'Prevents storing identical or very similar prompts multiple times, saving space.', snippet: `// Conceptual SimHash
function simhash(text) {
  // 1. Vectorize text (e.g., TF-IDF)
  // 2. Hash vector components
  // 3. Combine hashes to form fingerprint
  return fingerprint; // e.g., a 64-bit number
}
// If hammingDistance(simhash1, simhash2) < threshold, then similar.`, snippetLang: 'pseudocode' },
      { name: 'Persistent Trie (for Continuation)', purpose: 'Allows branching and resuming conversations at any token point efficiently.', snippet: `class TrieNode {
  constructor(char) {
    this.char = char;
    this.children = {};
    this.isEndOfWord = false;
  }
}
// Each path from root represents a sequence/prompt.`, snippetLang: 'javascript' },
    ],
  },
  {
    id: 'sharingChats',
    title: '4. Sharing Chats, Continuation, Branching',
    summary: 'Supporting shared chats, forking conversations (like Git), co-editing, and public/private toggles.',
    techniques: [
      { name: 'DAG of Messages (for Shared Chats)', purpose: 'Models forks and merges in conversations, similar to Git commit history.', snippet: `const message = {
  id: "msg_c",
  text: "Great idea!",
  user: "user2",
  timestamp: 1678886400,
  previous: ["msg_a", "msg_b"] // Merged from two branches
};`, snippetLang: 'json' },
      { name: 'Persistent Data Structures (for Snapshots/Versions)', purpose: 'Efficiently create snapshots for undo/redo and version control by sharing unchanged parts.', snippet: `// Conceptual with immutable.js or similar
// let chatStateV1 = Immutable.Map({ messages: [...] });
// let chatStateV2 = chatStateV1.setIn(['messages', 0, 'text'], 'new text');
// V1 and V2 share most underlying data.`, snippetLang: 'pseudocode' },
    ],
  },
  {
    id: 'fileUploadsSandbox',
    title: '5. Upload Files + Run in Shared Environment',
    summary: 'Allowing users to upload code or files and run them in a sandboxed environment (like ChatGPT Code Interpreter).',
    techniques: [
      { name: 'Content-Addressable Storage (CAS)', purpose: 'Deduplicates files and provides an audit trail. File ID is its hash.', snippet: `async function storeFile(fileContent) {
  const hashBuffer = await crypto.subtle.digest('SHA-256', fileContent);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  const fileId = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  // saveToStorage(fileId, fileContent);
  return fileId;
}`, snippetLang: 'javascript' },
      { name: 'Deterministic State Machines / Snapshot Isolation', purpose: 'Ensures sandboxed code execution is replayable and secure.', snippet: `// Conceptual
// function executeInSandbox(code, initialState) {
//   let vmState = clone(initialState);
//   // ... execute code, only affecting vmState ...
//   return getDiff(initialState, vmState); // Or final state
// }`, snippetLang: 'pseudocode' },
    ],
  },
  {
    id: 'ragRetrieval',
    title: '6. RAG / Deep Retrieval',
    summary: 'Retrieving relevant documents from a knowledge base and combining them with the user prompt.',
    techniques: [
      { name: 'Approximate Nearest Neighbors (ANN) (e.g., FAISS, HNSW)', purpose: 'Fast lookup of similar vectors (embeddings) in huge knowledge bases.', snippet: `// Conceptual - library like FAISS would be used
// annIndex.add(document_embeddings);
// const results = annIndex.search(query_embedding, k_neighbors);
function findNearest(queryVector, vectorList, k) {
  // Simplified brute-force for illustration
  return vectorList
    .map(v => ({ vector: v, dist: cosineSimilarity(queryVector, v.embedding) }))
    .sort((a,b) => b.dist - a.dist) // higher similarity is better
    .slice(0, k);
}`, snippetLang: 'javascript' },
      { name: 'Vector Caching (LRU or 2Q)', purpose: 'Speeds up retrieval of frequently accessed or recently used document embeddings.', snippet: `class LRUCache {
  constructor(capacity) { /* ... */ }
  get(key) { /* ... */ }
  put(key, value) { /* ... */ }
}
// const embeddingCache = new LRUCache(1000);`, snippetLang: 'javascript' },
    ],
  },
  {
    id: 'highConcurrency',
    title: '7. High-Concurrency Message Handling',
    summary: 'Managing situations when many users are simultaneously using the same backend model or file store.',
    techniques: [
      { name: 'Lock-Free Queues (e.g., Michael-Scott)', purpose: 'Non-blocking enqueue/dequeue operations for high-throughput message passing between threads or services.', snippet: `// Highly conceptual, real implementation is complex
// atomicPush(queue, item) { /* uses CAS operations */ }
// atomicPop(queue) { /* uses CAS operations */ }`, snippetLang: 'pseudocode' },
      { name: 'CRDTs (Conflict-free Replicated Data Types)', purpose: 'Enable collaborative editing and state synchronization with automatic merging of offline edits.', snippet: `// Conceptual: G-Counter (Grow-Only Counter)
// let counterA = { nodeId: 'A', counts: {'A': 3} };
// let counterB = { nodeId: 'B', counts: {'B': 2} };
// merge(counterA, counterB) -> {'A':3, 'B':2} (value = 5)`, snippetLang: 'pseudocode' },
    ],
  },
  {
    id: 'analyticsMonitoring',
    title: '8. Usage Analytics & Monitoring',
    summary: 'Providing real-time dashboards, audit logs, and per-user heatmaps for system observability.',
    techniques: [
      { name: 'Count-Min Sketch', purpose: 'Space-efficient estimation of frequencies for large-scale event counting (e.g., prompt popularity).', snippet: `// Conceptual: d hash functions, w counters per row
// update(item): for i=0..d-1, sketch[i][hash_i(item)]++
// estimate(item): min(for i=0..d-1, sketch[i][hash_i(item)])`, snippetLang: 'pseudocode' },
      { name: 'HyperLogLog', purpose: 'Estimates cardinality (number of unique items like users or IPs) with high accuracy and low memory.', snippet: `// Conceptual
// hll.add(userId);
// const uniqueUserCount = hll.estimate();`, snippetLang: 'pseudocode' },
    ],
  },
];


// --- Data for Strategic LLM Optimizations ---
interface StrategicTechnique extends AlgorithmSnippet {}

interface StrategicArea {
  id: string;
  title: string;
  focusSummary: string;
  techniques: StrategicTechnique[];
  whyItMatters: string;
}

const strategicAreasData: StrategicArea[] = [
  {
    id: 'tokenStreamHandling',
    title: '1. Token Stream Handling / Partial Generation',
    focusSummary: 'Focus on maintaining summary statistics of generation using constant space (e.g., min/max, avg token rate), employing Trie-based prefix decoders for token validity pruning, and enabling preemption-resilient decoding to resume generation mid-sequence (persistent beam search tree).',
    techniques: [
      { name: 'Streaming Algorithms (Summary Stats)', purpose: 'Maintain real-time generation statistics (e.g., token rate) efficiently.', snippetLang: 'python', snippet: `class TokenStreamMonitor:
    def __init__(self):
        self.token_count = 0
        self.start_time = time.time()
        self.min_rate = float('inf')
        self.max_rate = 0.0

    def add_token(self):
        self.token_count += 1
        # Update rates, etc.

    def get_avg_rate(self):
        elapsed_time = time.time() - self.start_time
        return self.token_count / elapsed_time if elapsed_time > 0 else 0` },
      { name: 'Trie-based Prefix Decoders', purpose: 'Prune invalid token sequences early during generation.', snippetLang: 'cpp', snippet: `struct TrieNode {
    std::map<int, TrieNode*> children; // token_id -> Node
    bool isEndOfValidSequence;
    TrieNode() : isEndOfValidSequence(false) {}

    void insert(const std::vector<int>& token_ids) {
        TrieNode* curr = this;
        for (int token_id : token_ids) {
            if (curr->children.find(token_id) == curr->children.end()) {
                curr->children[token_id] = new TrieNode();
            }
            curr = curr->children[token_id];
        }
        curr->isEndOfValidSequence = true;
    }
};` },
    ],
    whyItMatters: 'Optimizing for fast, non-blocking generation under pressure while remaining accurate.'
  },
  {
    id: 'promptCompression',
    title: '2. Prompt Compression / Token Optimization',
    focusSummary: 'Utilize semantic-aware compression using tries and semantic hashing, apply dynamic programming for optimal token segmentation, and explore subword merge optimization strategies (like BPE or Unigram LM).',
    techniques: [
      { name: 'Optimal Token Segmentation (Dynamic Programming)', purpose: 'Find the most efficient way to segment text into known tokens.', snippetLang: 'python', snippet: `def optimal_segmentation(text, token_costs):
    n = len(text)
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    for i in range(1, n + 1):
        for j in range(i):
            segment = text[j:i]
            if segment in token_costs:
                if dp[j] + token_costs[segment] < dp[i]:
                    dp[i] = dp[j] + token_costs[segment]
    return dp[n] if dp[n] != float('inf') else -1 # Cost or -1 if not possible` },
      { name: 'Trie for Subword Merges', purpose: 'Efficiently manage and search for subword units during BPE-like processes.', snippetLang: 'cpp', snippet: `struct SubwordTrieNode {
    std::map<char, SubwordTrieNode*> children;
    int frequency;
    bool isEndOfWord;
    SubwordTrieNode() : frequency(0), isEndOfWord(false) {}
    // Methods to build/update from corpus, find best pair to merge
};` },
    ],
    whyItMatters: 'Token limits directly affect LLM quality, latency, and cost — this saves tokens algorithmically.'
  },
  {
    id: 'speculativeDecoding',
    title: '3. Speculative Decoding / Prefetching',
    focusSummary: 'Implement branch prediction-like speculative trees, manage rollbacks with tree pruning, and consider model distillation for generating fast speculative "drafts".',
    techniques: [
      { name: 'Speculative Tree with Rollbacks', purpose: 'Explore multiple token sequences in parallel, rolling back incorrect paths.', snippetLang: 'python', snippet: `class SpeculativeNode:
    def __init__(self, token_id, parent=None, score=0):
        self.token_id = token_id
        self.parent = parent
        self.children = []
        self.score = score # Likelihood from draft model
        self.is_verified = False # Verified by main model

    def add_child(self, token_id, score):
        child = SpeculativeNode(token_id, self, score)
        self.children.append(child)
        return child

# During generation:
# draft_model generates k candidates (children)
# main_model verifies one, prunes others (rollback)` },
    ],
    whyItMatters: 'Big performance booster — can show using search tree algorithms with backtracking and checkpointing.'
  },
  {
    id: 'attentionWindow',
    title: '4. Attention Window Management',
    focusSummary: 'Employ sliding windows with Segment Trees for summarizing distant history, use priority attention (Min-heap or Top-K) for salient tokens, and implement cache-aware KV attention using paging or token replay buffers.',
    techniques: [
      { name: 'Segment Tree for Summarizing Distant History', purpose: 'Efficiently query aggregated information (e.g., max attention score) over ranges of past tokens.', snippetLang: 'cpp', snippet: `// Segment Tree Node for e.g. max value in range
struct SegmentTreeNode {
    int max_val;
    // Can store other aggregates
    SegmentTreeNode() : max_val(0) {}
};
// Array-based implementation of segment tree
// void build(int arr[], SegmentTreeNode tree[], int node, int start, int end);
// void update(SegmentTreeNode tree[], int node, int start, int end, int idx, int val);
// int query(SegmentTreeNode tree[], int node, int start, int end, int l, int r);` },
      { name: 'Min-Heap for Priority Attention', purpose: 'Maintain the top-K most salient tokens to attend to.', snippetLang: 'python', snippet: `import heapq

class PriorityAttention:
    def __init__(self, k):
        self.k = k
        self.heap = [] # Stores (salience_score, token_id)

    def add_token(self, token_id, salience_score):
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, (salience_score, token_id))
        elif salience_score > self.heap[0][0]: # If more salient than smallest in heap
            heapq.heapreplace(self.heap, (salience_score, token_id))

    def get_top_k_tokens(self):
        return [item[1] for item in self.heap]` },
    ],
    whyItMatters: 'Very algorithmic — designing what to forget, and what to cache.'
  },
  {
    id: 'promptReplay',
    title: '5. Prompt Replay / Reuse / Caching',
    focusSummary: 'Use Dedup Trees (Tries) to store prior generations for partial reuse, employ delta encoding and hash trees for shared sub-prompts, and implement reinsertion caches (LRU/2Q) with token-level similarity.',
    techniques: [
      { name: 'Trie for Prompt/Generation Deduplication', purpose: 'Store and quickly find existing prompt prefixes or full generations.', snippetLang: 'cpp', snippet: `struct GenerationTrieNode {
    std::map<int, GenerationTrieNode*> children; // token_id -> Node
    bool isEndOfGeneration;
    std::string generation_result_id; // If end of generation
    GenerationTrieNode() : isEndOfGeneration(false) {}
};` },
      { name: 'LRU Cache for Token Sequences', purpose: 'Cache frequently accessed token sequences or their embeddings.', snippetLang: 'python', snippet: `from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)` },
    ],
    whyItMatters: 'Saves 10–30% inference time for repeated queries. Show how algorithms improve hit rates.'
  },
  {
    id: 'checkpointingFaultTolerance',
    title: '6. Checkpointing & Fault-Tolerant Inference',
    focusSummary: 'Utilize persistent stacks for beam decoding, implement versioned checkpoints (ZFS-style diffs), and ensure stateful resumption with monotonic queues.',
    techniques: [
      { name: 'Persistent Stack for Beam Decoding States', purpose: 'Allow efficient saving and restoring of beam search states for fault tolerance.', snippetLang: 'cpp', snippet: `// Conceptual Persistent Stack Node
struct PersistentStackNode {
    int value; // Or complex beam state
    std::shared_ptr<PersistentStackNode> previous;
    PersistentStackNode(int val, std::shared_ptr<PersistentStackNode> prev)
        : value(val), previous(prev) {}
};
// Each 'push' creates a new head pointing to the old one.
// std::shared_ptr<PersistentStackNode> latest_state = ...;`},
      { name: 'Monotonic Queue for Stateful Resumption', purpose: 'Ensure tasks are processed in order and can be resumed correctly after a failure.', snippetLang: 'python', snippet: `from collections import deque

class MonotonicQueue: # For Job IDs or sequence numbers
    def __init__(self):
        self.queue = deque()
        self.last_processed_id = -1

    def enqueue(self, job_id, job_data):
        if job_id > self.last_processed_id:
            self.queue.append((job_id, job_data))
            # Potentially sort or maintain order if jobs can arrive out of order
            # but are expected to be processed monotonically

    def dequeue(self):
        if self.queue:
            job_id, job_data = self.queue.popleft()
            self.last_processed_id = job_id
            return job_data
        return None`}
    ],
    whyItMatters: 'Super important at scale — everything from autosave to GPU crashes must be handled.'
  },
  {
    id: 'fineTunedTokenScheduling',
    title: '7. Fine-tuned Token Scheduling',
    focusSummary: 'Compare greedy batching vs. lookahead-based scheduling, apply token-level batching heuristics (dynamic bin packing), and use batch merging with Huffman coding for shared prefix reuse.',
    techniques: [
      { name: 'Dynamic Bin Packing for Token Batching', purpose: 'Optimize packing of sequences of different lengths into fixed-size batches.', snippetLang: 'python', snippet: `def first_fit_decreasing_batching(sequences, batch_capacity):
    # sequences is a list of (id, length) tuples
    sequences.sort(key=lambda x: x[1], reverse=True)
    batches = []
    for seq_id, seq_len in sequences:
        placed = False
        for batch in batches:
            if sum(s[1] for s in batch) + seq_len <= batch_capacity:
                batch.append((seq_id, seq_len))
                placed = True
                break
        if not placed:
            batches.append([(seq_id, seq_len)])
    return batches`},
      { name: 'Huffman Coding for Shared Prefix Reuse (Conceptual)', purpose: 'Identify common prefixes in a batch and encode them efficiently.', snippetLang: 'python', snippet: `# 1. Build a frequency map of all prefixes in the batch.
# 2. Use Huffman algorithm to create codes for frequent prefixes.
# 3. When batching, represent shared prefixes with their shorter codes.
# This is more about data representation within a batch processing system.`}
    ],
    whyItMatters: 'Purely algorithmic: how to optimally pack sequences of different lengths.'
  },
  {
    id: 'latencyAwareRouting',
    title: '8. Latency-Aware Routing / Queuing',
    focusSummary: 'Employ Priority Queues with Shortest Remaining Time First (SRTF)-like heuristics, use dynamic batching via token-count histograms, and explore reinforcement learning-based dispatch.',
    techniques: [
      { name: 'Priority Queue for SRTF-like Request Scheduling', purpose: 'Prioritize requests that are shorter or closer to completion to reduce average latency.', snippetLang: 'cpp', snippet: `struct Request {
    int id;
    int remaining_tokens;
    // Other metadata like arrival_time, priority_score
    bool operator>(const Request& other) const { // For min-priority queue
        return remaining_tokens > other.remaining_tokens;
    }
};
// std::priority_queue<Request, std::vector<Request>, std::greater<Request>> request_q;`},
      { name: 'Dynamic Batching with Token-Count Histograms', purpose: 'Group requests into batches based on similar token counts to optimize GPU utilization.', snippetLang: 'python', snippet: `def create_batches_from_histogram(request_queue, histogram_bins, max_batch_size):
    # histogram_bins = [50, 100, 200, 500] (max tokens for bin)
    # Group requests by which bin their token count falls into
    # Then form batches from these groups, respecting max_batch_size
    # ...
    pass`}
    ],
    whyItMatters: 'Key for low-latency and high-throughput systems.'
  },
  {
    id: 'crossShardConsistency',
    title: '9. Cross-Shard Consistency / Replication',
    focusSummary: 'Implement CRDTs (Conflict-Free Replicated Data Types), use Version Vectors or Vector Clocks for ordering, and apply optimistic replication with rollback trees.',
    techniques: [
      { name: 'Vector Clocks for Event Ordering', purpose: 'Determine causal relationships between events across distributed replicas.', snippetLang: 'cpp', snippet: `class VectorClock {
public:
    std::map<std::string, int> clock; // node_id -> count
    void increment(const std::string& node_id) {
        clock[node_id]++;
    }
    // Merge, compare methods...
    // static bool happensBefore(const VectorClock& vc1, const VectorClock& vc2);
};` },
      { name: 'CRDTs (Example: G-Counter)', purpose: 'Allow concurrent updates on different replicas that can be merged without conflicts.', snippetLang: 'python', snippet: `class GCounter: # Grow-Only Counter
    def __init__(self, replica_id):
        self.replica_id = replica_id
        self.payload = {} # replica_id -> count

    def increment(self, val=1):
        self.payload[self.replica_id] = self.payload.get(self.replica_id, 0) + val

    def value(self):
        return sum(self.payload.values())

    def merge(self, other_counter):
        for replica, count in other_counter.payload.items():
            self.payload[replica] = max(self.payload.get(replica, 0), count)`}
    ],
    whyItMatters: 'Showcase replicated state machines with conflict resolution — very algorithm-heavy.'
  },
  {
    id: 'securityAwareDataStructures',
    title: '10. Security-Aware Data Structures',
    focusSummary: 'Use DAGs of tokens for prompt taint tracking, employ token anomaly detection with Markov chains or FSMs, and maintain content-addressed audit trails (Merkle log of generations).',
    techniques: [
      { name: 'DAG for Prompt Taint Tracking', purpose: 'Track the origin and influence of tokens to detect potential injections.', snippetLang: 'cpp', snippet: `struct TokenNode {
    int token_id;
    std::string source_info; // "user_input", "retrieved_doc", "generated"
    std::vector<std::shared_ptr<TokenNode>> dependencies;
    // Each generated token depends on previous tokens.
};` },
      { name: 'Merkle Log for Audit Trails', purpose: 'Create a tamper-proof log of generations for security and auditing.', snippetLang: 'python', snippet: `import hashlib

class MerkleLog:
    def __init__(self):
        self.log = [] # List of (data_item, hash)
        self.root_hash = None

    def add_entry(self, data_item):
        data_str = str(data_item)
        item_hash = hashlib.sha256(data_str.encode()).hexdigest()
        self.log.append((data_item, item_hash))
        # Recalculate Merkle root (simplified - would build tree)
        # For a log, can hash (prev_root + new_item_hash)
        if self.root_hash is None:
            self.root_hash = item_hash
        else:
            self.root_hash = hashlib.sha256((self.root_hash + item_hash).encode()).hexdigest()

    def verify(self):
        # Verification involves recomputing root or checking proofs
        pass`}
    ],
    whyItMatters: 'Brings algorithmic integrity to AI, ensuring the system is not just fast but also secure.'
  }
];

interface PortfolioEnhancement {
  feature: string;
  whatToShow: string;
}

const portfolioEnhancementsData: PortfolioEnhancement[] = [
  { feature: "Visual interactive demo", whatToShow: "Trie-based generation replay or history branching" },
  { feature: "Simulations", whatToShow: "Token scheduler simulator, load balancer stress test" },
  { feature: "Visuals", whatToShow: "DAGs, segment trees, hash maps applied in real cases" },
  { feature: "Benchmarks", whatToShow: "MinHash vs SimHash for dedup speed" },
  { feature: "Git-based", whatToShow: "Shared chat DAG like Git commit tree with merge/fork" },
  { feature: "CLI tool", whatToShow: "Trie-based prompt dedup CLI or vector store with ANN search" }
];

const finalTipsData: string[] = [
  "Pick 4–6 of these areas to go deep.",
  "Connect each one to a real user/business goal like \"lower latency\" or \"enable multi-user collaboration.\"",
  "Show algorithms not in theory but in system design context — that’s what makes it exceptional."
];

// --- Data for Technical Implementation Guide ---
interface TechnicalGuideCodeBlock {
  language: string;
  code: string;
  title?: string;
}

interface TechnicalGuideItem {
  id: string;
  title: string;
  description?: string;
  codeBlocks?: TechnicalGuideCodeBlock[];
  subItems?: TechnicalGuideItem[];
}

interface TechnicalGuideMajorSection {
  id: string;
  title: string;
  items: TechnicalGuideItem[];
}

const technicalImplementationGuideData: TechnicalGuideMajorSection[] = [
  {
    id: 'core-architecture',
    title: '1. Core Architecture with Specific Technologies',
    items: [
      {
        id: 'request-processing-pipeline',
        title: '1.1 Request Processing Pipeline',
        codeBlocks: [{ language: 'text', code: 'Client → CloudFlare (DDoS) → AWS ALB → API Gateway (Kong/Envoy) → Service Mesh (Istio) → Inference Pods' }],
        description: `**Specific Implementation:**
- **API Gateway**: Kong with Redis backend for rate limiting
- **Load Balancer**: Envoy with Consistent Hash Ring for session affinity
- **Service Discovery**: Consul with health checking every 10s
- **Message Queue**: Apache Kafka with 3 partitions per topic`
      },
      {
        id: 'data-structures-specs',
        title: '1.2 Data Structures - Precise Specifications',
        subItems: [
          {
            id: 'session-management',
            title: 'Session Management',
            codeBlocks: [{
              language: 'python',
              title: 'Conversation Context Store',
              code: `class ConversationContext:
    session_id: UUID
    messages: CircularBuffer[Message]  # Max 50 messages, O(1) append/pop
    embeddings: CompressedEmbedding   # 384-dim → 128-dim via PCA
    metadata: Dict[str, Any]
    ttl: int  # 24 hours default`
            }],
            description: `**Storage**: Redis Cluster with consistent hashing
- **Key Pattern**: \`session:{user_id}:{session_id}\`
- **TTL**: 86400 seconds (24 hours)
- **Compression**: zlib for messages > 1KB`
          },
          {
            id: 'prompt-caching-system',
            title: 'Prompt Caching System',
            codeBlocks: [{
              language: 'python',
              code: `class PromptCache:
    # L1: In-memory LRU (per inference pod)
    l1_cache: LRUCache[str, CachedResponse]  # 1GB capacity
    
    # L2: Distributed cache (Redis)
    l2_cache: RedisCluster
    
    # Cache key generation
    def generate_key(self, prompt: str, model_params: Dict) -> str:
        return sha256(f"{prompt}:{json.dumps(model_params, sort_keys=True)}").hexdigest()[:16]`
            }],
            description: `**Cache Strategy**: 
- L1 hit rate target: 15-20%
- L2 hit rate target: 35-40%
- Eviction: LRU with TTL (6 hours for completions)`
          }
        ]
      },
      {
        id: 'vector-db-architecture',
        title: '1.3 Vector Database Architecture',
        description: `**Technology**: Qdrant with HNSW indexing`,
        codeBlocks: [{
          language: 'json', // Or python, it's a dict assignment
          title: 'Vector Index Configuration',
          code: `qdrant_config = {
    "vectors": {
        "size": 384,  # sentence-transformers/all-MiniLM-L6-v2
        "distance": "Cosine"
    },
    "hnsw_config": {
        "m": 16,           # Number of bi-directional links
        "ef_construct": 200, # Size of dynamic candidate list  
        "full_scan_threshold": 10000
    }
}`
        }],
        subItems: [ {
          id: 'vector-db-sharding',
          title: 'Sharding Strategy',
          description: `Hash-based on document_id
- 4 shards initially, horizontal scaling to 16
- Replication factor: 3
- Consistency level: Quorum (2/3)`
        }]
      }
    ]
  },
  {
    id: 'inference-engine',
    title: '2. Inference Engine - Detailed Implementation',
    items: [
      {
        id: 'model-serving-vllm',
        title: '2.1 Model Serving with vLLM',
        codeBlocks: [{
          language: 'python',
          title: 'vLLM Configuration',
          code: `engine_config = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "tensor_parallel_size": 2,      # 2x A100 40GB
    "pipeline_parallel_size": 1,
    "max_model_len": 8192,
    "block_size": 16,               # Token block size
    "max_num_seqs": 256,            # Batch size
    "enable_chunked_prefill": True,
    "max_num_batched_tokens": 4096
}`
        }]
      },
      {
        id: 'token-streaming',
        title: '2.2 Token Streaming Implementation',
        codeBlocks: [{
          language: 'python',
          title: 'Server-Sent Events with backpressure',
          code: `class StreamingResponse:
    def __init__(self, generator):
        self.buffer = asyncio.Queue(maxsize=32)  # Backpressure buffer
        self.generator = generator
    
    async def stream_tokens(self):
        async for token_chunk in self.generator:
            if self.buffer.full():
                await asyncio.sleep(0.001)  # Backpressure delay
            await self.buffer.put(f"data: {json.dumps(token_chunk)}\\n\\n")`
        }]
      },
      {
        id: 'dynamic-batching',
        title: '2.3 Dynamic Batching Algorithm',
        codeBlocks: [{
          language: 'python',
          code: `class DynamicBatcher:
    def __init__(self, max_batch_size=32, max_wait_ms=10):
        self.pending_requests = []
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
    
    async def add_request(self, request):
        self.pending_requests.append(request)
        
        # Trigger batch if conditions met
        if (len(self.pending_requests) >= self.max_batch_size or 
            self._oldest_request_age() > self.max_wait_ms):
            return await self._process_batch()`
        }]
      }
    ]
  },
  {
    id: 'advanced-ds-algo',
    title: '3. Advanced Data Structures & Algorithms',
    items: [
      {
        id: 'rate-limiting-token-bucket',
        title: '3.1 Rate Limiting - Token Bucket with Redis',
        codeBlocks: [{
          language: 'python',
          code: `class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity      # 1000 requests
        self.refill_rate = refill_rate # 10 requests/second
        
    async def consume(self, user_id: str, tokens: int = 1) -> bool:
        lua_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local tokens_requested = tonumber(ARGV[3])
        local now = tonumber(ARGV[4])
        
        local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1]) or capacity
        local last_refill = tonumber(bucket[2]) or now
        
        -- Calculate tokens to add
        local time_passed = now - last_refill
        local tokens_to_add = time_passed * refill_rate
        tokens = math.min(capacity, tokens + tokens_to_add)
        
        if tokens >= tokens_requested then
            tokens = tokens - tokens_requested
            redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
            redis.call('EXPIRE', key, 3600)
            return 1
        else
            return 0
        end
        """`
        }]
      },
      {
        id: 'content-moderation-pipeline',
        title: '3.2 Content Moderation Pipeline',
        codeBlocks: [{
          language: 'python',
          code: `class ModerationPipeline:
    def __init__(self):
        # Stage 1: Fast regex-based filtering
        self.profanity_trie = AhoCorasick()  # O(n + m + z) pattern matching
        
        # Stage 2: ML-based classification
        self.toxicity_model = "unitary/toxic-bert"
        
        # Stage 3: Custom business rules
        self.rule_engine = RuleEngine()
    
    async def moderate(self, text: str) -> ModerationResult:
        # Fast path: Trie-based detection (< 1ms)
        if self.profanity_trie.search(text.lower()):
            return ModerationResult(blocked=True, reason="profanity", confidence=1.0)
        
        # Slower path: ML inference (< 50ms)
        toxicity_score = await self.toxicity_model.predict(text)
        if toxicity_score > 0.8:
            return ModerationResult(blocked=True, reason="toxicity", confidence=toxicity_score)`
        }]
      },
      {
        id: 'session-affinity-consistent-hashing',
        title: '3.3 Session Affinity with Consistent Hashing',
        codeBlocks: [{
          language: 'python',
          code: `class ConsistentHashRing:
    def __init__(self, nodes: List[str], virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring = {}
        self.sorted_keys = []
        
        for node in nodes:
            self.add_node(node)
    
    def add_node(self, node: str):
        for i in range(self.virtual_nodes):
            key = self._hash(f"{node}:{i}")
            self.ring[key] = node
        self._update_sorted_keys()
    
    def get_node(self, session_id: str) -> str:
        if not self.ring:
            return None
        
        key = self._hash(session_id)
        idx = bisect.bisect_right(self.sorted_keys, key)
        if idx == len(self.sorted_keys):
            idx = 0
        return self.ring[self.sorted_keys[idx]]`
        }]
      }
    ]
  },
  {
    id: 'db-schema-storage',
    title: '4. Database Schema & Storage',
    items: [
      {
        id: 'postgres-schema-analytics',
        title: '4.1 PostgreSQL Schema for Analytics',
        codeBlocks: [{
          language: 'sql',
          code: `-- Conversation tracking
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    session_id UUID NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    model_version VARCHAR(50),
    total_tokens INTEGER,
    total_cost DECIMAL(10,6)
);

-- Partitioned by month for performance
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id),
    role VARCHAR(20) NOT NULL, -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    tokens INTEGER,
    latency_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
) PARTITION BY RANGE (created_at);

-- Indexes for common queries
CREATE INDEX CONCURRENTLY idx_conversations_user_created 
ON conversations (user_id, created_at DESC);

CREATE INDEX CONCURRENTLY idx_messages_conversation_created 
ON messages (conversation_id, created_at);`
        }]
      },
      {
        id: 'clickhouse-realtime-analytics',
        title: '4.2 ClickHouse for Real-time Analytics',
        codeBlocks: [{
          language: 'sql',
          code: `-- High-frequency metrics table
CREATE TABLE llm_metrics (
    timestamp DateTime64(3),
    user_id String,
    model String,
    endpoint String,
    latency_ms UInt32,
    tokens_input UInt32,
    tokens_output UInt32,
    cache_hit Bool,
    status_code UInt16
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, user_id);

-- Materialized view for real-time dashboards
CREATE MATERIALIZED VIEW llm_metrics_5min
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, model, endpoint)
AS SELECT
    toStartOfFiveMinutes(timestamp) as timestamp,
    model,
    endpoint,
    count() as requests,
    avg(latency_ms) as avg_latency,
    quantile(0.95)(latency_ms) as p95_latency,
    sum(tokens_input) as total_input_tokens,
    sum(tokens_output) as total_output_tokens
FROM llm_metrics
GROUP BY timestamp, model, endpoint;`
        }]
      }
    ]
  },
  {
    id: 'deployment-infrastructure',
    title: '5. Deployment & Infrastructure',
    items: [
      {
        id: 'kubernetes-manifests',
        title: '5.1 Kubernetes Manifests',
        codeBlocks: [{
          language: 'yaml',
          title: 'Inference pod with GPU scheduling',
          code: `apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-inference
  template:
    spec:
      containers:
      - name: vllm-server
        image: vllm/vllm-openai:latest
        resources:
          requests:
            memory: "32Gi"
            nvidia.com/gpu: 2
          limits:
            memory: "64Gi"
            nvidia.com/gpu: 2
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0,1"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10`
        }]
      },
      {
        id: 'hpa',
        title: '5.2 Horizontal Pod Autoscaler',
        codeBlocks: [{
          language: 'yaml',
          code: `apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-inference
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: gpu_utilization
      target:
        type: AverageValue
        averageValue: "80"`
        }]
      }
    ]
  },
  {
    id: 'monitoring-observability',
    title: '6. Monitoring & Observability',
    items: [
      {
        id: 'custom-metrics',
        title: '6.1 Custom Metrics',
        codeBlocks: [{
          language: 'python',
          title: 'Prometheus metrics',
          code: `from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter('llm_requests_total', 'Total requests', ['model', 'endpoint'])
REQUEST_LATENCY = Histogram('llm_request_duration_seconds', 'Request latency')
GPU_UTILIZATION = Gauge('gpu_utilization_percent', 'GPU utilization', ['gpu_id'])
CACHE_HIT_RATE = Gauge('cache_hit_rate', 'Cache hit rate', ['cache_type'])`
        }]
      },
      {
        id: 'distributed-tracing',
        title: '6.2 Distributed Tracing',
        codeBlocks: [{
          language: 'python',
          title: 'OpenTelemetry tracing',
          code: `from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

tracer = trace.get_tracer(__name__)

@tracer.start_as_current_span("llm_inference")
async def process_request(request):
    span = trace.get_current_span()
    span.set_attribute("model.name", request.model)
    span.set_attribute("input.token_count", request.token_count)
    
    with tracer.start_as_current_span("cache_lookup"):
        cached_response = await cache.get(request.cache_key)
    
    if not cached_response:
        with tracer.start_as_current_span("model_inference"):
            response = await model.generate(request.prompt)`
        }]
      }
    ]
  },
  {
    id: 'performance-slas',
    title: '7. Performance Targets & SLAs',
    items: [
      {
        id: 'latency-requirements',
        title: '7.1 Latency Requirements',
        description: `- **P50 Latency**: < 200ms (first token)
- **P95 Latency**: < 500ms (first token) 
- **P99 Latency**: < 1000ms (first token)
- **Streaming Latency**: < 50ms between tokens`
      },
      {
        id: 'throughput-targets',
        title: '7.2 Throughput Targets',
        description: `- **Requests/second**: 10,000 concurrent users
- **Tokens/second**: 50,000 output tokens globally
- **GPU Utilization**: > 80% average`
      },
      {
        id: 'availability',
        title: '7.3 Availability',
        description: `- **Uptime SLA**: 99.95% (21.9 minutes downtime/month)
- **Recovery Time**: < 5 minutes for regional failures
- **Data Durability**: 99.999999999% (11 9's)`
      }
    ]
  },
  {
    id: 'cost-optimization',
    title: '8. Cost Optimization Strategies',
    items: [
      {
        id: 'gpu-resource-management',
        title: '8.1 GPU Resource Management',
        codeBlocks: [{
          language: 'python',
          code: `class GPUScheduler:
    def __init__(self):
        self.gpu_pools = {
            "high_priority": GPUPool(gpu_type="A100", count=8),
            "standard": GPUPool(gpu_type="A40", count=16), 
            "spot": GPUPool(gpu_type="V100", count=32, spot=True)
        }
    
    def assign_gpu(self, request_priority: str, estimated_tokens: int):
        if request_priority == "premium" or estimated_tokens > 2000:
            return self.gpu_pools["high_priority"].acquire()
        elif estimated_tokens > 500:
            return self.gpu_pools["standard"].acquire()
        else:
            return self.gpu_pools["spot"].acquire()`
        }],
        description: `This enhanced design provides the technical depth and specific implementation details needed to build a production-grade LLM interface system. Each component includes concrete technology choices, data structures, algorithms, and performance targets.`
      }
    ]
  }
];


// --- Components ---

const Header: React.FC = () => (
  <header className="app-header" role="banner">
    <h1>LLM Inference Portfolio</h1>
    <p>Showcasing Algorithmic Problem Solving in AI Infrastructure</p>
  </header>
);

const HeroSection: React.FC = () => (
  <section className="hero-section section" aria-labelledby="hero-title">
    <h2 id="hero-title" className="section-title sr-only">Welcome</h2>
    <p className="centered-text-block">
      This portfolio explores the application of algorithmic problem-solving
      to enhance Large Language Model (LLM) inference infrastructure.
      Discover how robust algorithms contribute to efficient, scalable, and low-latency AI solutions.
    </p>
  </section>
);

const BusinessCasesSection: React.FC = () => (
  <section className="section" aria-labelledby="business-cases-title">
    <h2 id="business-cases-title" className="section-title">Business Cases & Applications</h2>
    <div className="business-cases-placeholder">
      <p>
        <em>
          This section will detail specific business cases where advanced LLM inference capabilities,
          driven by algorithmic excellence, provide significant value. Examples include enhanced customer service
          chatbots, real-time content generation, complex data analysis, and more.
          Stay tuned for detailed examples!
        </em>
      </p>
    </div>
  </section>
);

interface FeatureCardProps {
  feature: Feature;
}

const FeatureCard: React.FC<FeatureCardProps> = ({ feature }) => (
  <article className="feature-card" aria-labelledby={`feature-title-${feature.id}`}>
    <h3 id={`feature-title-${feature.id}`}>{feature.title}</h3>
    <p>{feature.description}</p>
  </article>
);

const FeaturesSection: React.FC = () => (
  <section className="section" aria-labelledby="features-title">
    <h2 id="features-title" className="section-title">Key LLM Infrastructure Features</h2>
    <div className="features-grid">
      {llmFeatures.map((feature) => (
        <FeatureCard key={feature.id} feature={feature} />
      ))}
    </div>
  </section>
);

const AlgorithmicDeepDiveSection: React.FC = () => (
  <section className="section algorithmic-deep-dive" aria-labelledby="deep-dive-title">
    <h2 id="deep-dive-title" className="section-title">Algorithmic Deep Dive: Powering LLM Infrastructure</h2>
    {algorithmicDeepDiveData.map(category => (
      <article key={category.id} className="deep-dive-category" aria-labelledby={`category-title-${category.id}`}>
        <h3 id={`category-title-${category.id}`}>{category.title}</h3>
        <p className="category-summary centered-text-block">{category.summary}</p>
        <div className="algorithms-list">
          <h4>Key Techniques & Algorithms:</h4>
          <ul>
            {category.techniques.map(tech => (
              <li key={tech.name}>
                <strong className="algorithm-name">{tech.name}:</strong> {tech.purpose}
                {tech.snippet && (
                  <div className="code-snippet-container">
                    <span className="snippet-label">Illustrative Snippet ({tech.snippetLang || 'pseudocode'}):</span>
                    <pre className="code-snippet"><code className={`language-${tech.snippetLang}`}>{tech.snippet.trim()}</code></pre>
                  </div>
                )}
              </li>
            ))}
          </ul>
        </div>
      </article>
    ))}
  </section>
);

const StrategicAreasSection: React.FC = () => (
  <section className="section strategic-areas" aria-labelledby="strategic-areas-title">
    <h2 id="strategic-areas-title" className="section-title">Strategic LLM Optimizations: Advanced Algorithmic Insights</h2>
     <p className="section-intro centered-text-block">
      Beyond foundational features, these strategic areas highlight where deep algorithmic thinking can provide
      significant advantages in building cutting-edge, efficient, and robust LLM inference providers.
    </p>
    {strategicAreasData.map(area => (
      <article key={area.id} className="strategic-area-category" aria-labelledby={`strategic-area-${area.id}`}>
        <h3 id={`strategic-area-${area.id}`}>{area.title}</h3>
        <div className="focus-areas-summary">
          <h4>Focus Areas & Rationale:</h4>
          <p className="centered-text-block">{area.focusSummary}</p>
          <p className="centered-text-block"><em>Why it matters: {area.whyItMatters}</em></p>
        </div>
        <div className="algorithms-list">
          <h4>Illustrative Techniques & Snippets:</h4>
          <ul>
            {area.techniques.map(tech => (
              <li key={tech.name}>
                <strong className="algorithm-name">{tech.name}:</strong> {tech.purpose}
                {tech.snippet && (
                  <div className="code-snippet-container">
                    <span className="snippet-label">Illustrative Snippet ({tech.snippetLang || 'pseudocode'}):</span>
                    <pre className="code-snippet"><code className={`language-${tech.snippetLang}`}>{tech.snippet.trim()}</code></pre>
                  </div>
                )}
              </li>
            ))}
          </ul>
        </div>
      </article>
    ))}

    <article className="portfolio-enhancements" aria-labelledby="portfolio-enhancements-title">
      <h3 id="portfolio-enhancements-title">Portfolio Enhancement Ideas</h3>
      <p className="centered-text-block">Consider these enhancements to further demonstrate your skills:</p>
      <table className="enhancements-table">
        <thead>
          <tr>
            <th>Feature</th>
            <th>What You Can Show</th>
          </tr>
        </thead>
        <tbody>
          {portfolioEnhancementsData.map(enhancement => (
            <tr key={enhancement.feature}>
              <td>{enhancement.feature}</td>
              <td>{enhancement.whatToShow}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </article>

    <article className="final-tips" aria-labelledby="final-tips-title">
      <h3 id="final-tips-title">Final Tips for Your Portfolio</h3>
      <ul className="tips-list">
        {finalTipsData.map((tip, index) => (
          <li key={index}>{tip}</li>
        ))}
      </ul>
    </article>
  </section>
);

const TechnicalImplementationGuideSection: React.FC = () => {
  const renderTechnicalItems = (items: TechnicalGuideItem[], level: number): JSX.Element[] | null => {
    if (!items || items.length === 0) return null;
    const HeadingTag = `h${Math.min(6, 5 + level)}` as keyof JSX.IntrinsicElements; // Start with h5 for sub-items

    return items.map(item => (
      <div key={item.id} className={`technical-item-level-${level}`}>
        <HeadingTag id={`tech-item-${item.id}`}>{item.title}</HeadingTag>
        {item.description && <p className="technical-item-description" style={{ whiteSpace: 'pre-wrap' }}>{item.description}</p>}
        {item.codeBlocks && item.codeBlocks.map((cb, index) => (
          <div key={index} className="code-snippet-container">
            {cb.title && <span className="snippet-label">{cb.title} ({cb.language}):</span>}
            {!cb.title && <span className="snippet-label">Snippet ({cb.language}):</span>}
            <pre className="code-snippet"><code className={`language-${cb.language}`}>{cb.code.trim()}</code></pre>
          </div>
        ))}
        {item.subItems && renderTechnicalItems(item.subItems, level + 1)}
      </div>
    ));
  };

  return (
    <section className="section technical-implementation-guide" aria-labelledby="tech-impl-guide-title">
      <h2 id="tech-impl-guide-title" className="section-title">Enhanced Distributed LLM Interface System - Technical Implementation Guide</h2>
      {technicalImplementationGuideData.map(majorSection => (
        <article key={majorSection.id} className="technical-major-section" aria-labelledby={`tech-major-section-${majorSection.id}`}>
          <h3 id={`tech-major-section-${majorSection.id}`}>{majorSection.title}</h3>
          {majorSection.items.map(item => ( // These are top-level items within a major section (e.g., 1.1, 1.2)
            <div key={item.id} className="technical-item-level-0">
              <h4 id={`tech-item-${item.id}`}>{item.title}</h4> {/* Main items are h4 */}
              {item.description && <p className="technical-item-description" style={{ whiteSpace: 'pre-wrap' }}>{item.description}</p>}
              {item.codeBlocks && item.codeBlocks.map((cb, index) => (
                <div key={index} className="code-snippet-container">
                  {cb.title && <span className="snippet-label">{cb.title} ({cb.language}):</span>}
                  {!cb.title && <span className="snippet-label">Snippet ({cb.language}):</span>}
                  <pre className="code-snippet"><code className={`language-${cb.language}`}>{cb.code.trim()}</code></pre>
                </div>
              ))}
              {item.subItems && renderTechnicalItems(item.subItems, 0)} {/* First level of subItems render as h5 */}
            </div>
          ))}
        </article>
      ))}
    </section>
  );
};


const Footer: React.FC = () => (
  <footer className="app-footer" role="contentinfo">
    <p>&copy; {new Date().getFullYear()} [Your Name/Student ID] - Algorithmic Problem Solving Portfolio</p>
    <p>Course: [Your Course Name/Number Here]</p>
  </footer>
);

const App: React.FC = () => {
  React.useEffect(() => {
    if (typeof (window as any).Prism !== 'undefined') {
      (window as any).Prism.highlightAll();
    }
  }, []);

  return (
    <>
      <Header />
      <main className="main-content" role="main">
        <HeroSection />
        <BusinessCasesSection />
        <FeaturesSection />
        <AlgorithmicDeepDiveSection />
        <StrategicAreasSection />
        <TechnicalImplementationGuideSection />
      </main>
      <Footer />
    </>
  );
};

const container = document.getElementById('root');
if (container) {
  const root = createRoot(container);
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
} else {
  console.error('Failed to find the root element');
}
