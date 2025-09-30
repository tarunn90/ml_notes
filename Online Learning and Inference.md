

  
# Levels of RTML

* **Level 1: Online predictions (real-time prediction):** The ML system makes predictions in real-time.

* **Level 2: Continual learning (real-time adaptation):** The system incorporates new data and updates the model in real-time.

  

# Problems with Batch Predictions

* **Latency/User Experience:** Models that take milliseconds too long to make predictions risk users clicking elsewhere.
* **Data Staleness:** Predictions generated offline in batches (e.g., every few hours) can become outdated.
* **Limited Scope:** Batch prediction, storing pre-computed results, works only if the input space is finite (e.g., predicting recommendations for a known set of users periodically). It fails for new or anonymous users where no precomputed personalized recommendations exist.

  

# Level 1: Online Inference, Offline Training
  
We can do online prediction with either **batch features** pulled from an offline store or **online features** accessed or generated in real-time. 

## Fast Inference (Model Optimization)

This focuses on enabling models to make predictions in the order of milliseconds:

* **Make Models Faster (Inference Optimization):** Techniques include fusing operations, distributing computations, memory footprint optimization, and writing high performance kernels targeting specific hardware.
* **Make Models Smaller (Model Compression):** Originally for edge devices, smaller models often run faster.
* **Quantization:** Using lower precision floats (e.g., 16-bit) or integers (e.g., 8-bit) instead of 32-bit floats for weights.
* **Knowledge Distillation:** Training a smaller model (student) to mimic a larger model or ensemble (teacher). Example: **DistilBERT** reduces BERT size by 40% while being 60% faster.
* **Pruning:** Setting parameters least useful to predictions to zero.
* **Low-rank Factorization:** Replacing over-parametric convolution filters with compact blocks.
* **Make Hardware Faster:** Developing hardware for faster inference and training on the cloud and edge devices is a booming research area.

  
## Real-Time Pipeline

This ensures data processing, model input, and prediction return happen in real-time:

* **Data Management:** Critical for fraud detection, which requires looking at recent user history, activities, and transactions happening concurrently.
* **In-Memory Storage:** Information about critical events (e.g., location choice, booking, adding credit card) should be kept in memory (usually for days) for quick access.
* **Tools:** Apache Kafka and Amazon Kinesis are common tools; Kafka is a stream storage.
* **Stream Processing vs. Batch Processing:**
* **Streams of Data Never Finish** (unlike static data like CSV files).
* **Batch Processing** (of static data) is efficient; **Stream Processing** (of events as they arrive) is fast
* **Event-driven vs. Request-driven:**
	* **Request-driven (REST APIs/Microservices):** Client sends requests, server responds. Prone to inter-service communication bottlenecks and lacks an overview of data flow, complicating monitoring and debugging.
	* **Event-driven (Pub/Sub Model, e.g., Kafka):** Services broadcast events to a stream; other services subscribe to necessary data. This improves speed and allows for system-wide data monitoring.
### Stages of Adoption for Online Prediction

Companies typically progress through stages when moving towards online inference:

* **Stage 1. Batch Prediction:** All predictions are precomputed offline (e.g., Netflix recommendations circa 2021).
* **Stage 2. Online Prediction with Batch Features:** Predictions are generated upon request, leveraging real-time user activities to retrieve pre-computed embeddings (batch features). This allows for **in-session adaptation** (e.g., recommending accessories based on what a new visitor has just viewed). Requires fast models and a feature store for batch features.
* **Stage 3. Online Prediction with Online Features:** Utilizes complex, real-time streaming features alongside batch features.
* **RT Features:** Computed in real-time as soon as a prediction request is generated (e.g., distance, recent view count calculated via Lambda function or SQL query).
* **Near RT Features:** Recomputed much more frequently (order of seconds) using a streaming processing engine.
* **Requirements:** Mature streaming infrastructure, efficient stream processing engine, and a feature store (for materialized feature consistency).


## Continual Learning (Level 2)

### Defining Continual Learning (CL)

* **Definition:** CL means the system can incorporate new data and update the model in real-time, generally defined as in the order of minutes.
* **Core Difference from Traditional Online Training:** CL typically uses **stateful training** (the model continues training, or fine-tuning, on new data) rather than **stateless retraining** (training from scratch each time).
* **Catastrophic Forgetting:** Traditional neural networks trained on new data abruptly forget previously learned information. CL algorithms are designed to mitigate this.
* **Benefits of Stateful Training:** Training frequency becomes a customizable "knob to twist" (e.g., hourly updates, or updates upon distribution shift detection). It requires less data for updates.

  

### Stages Towards Continual Learning (CL)

* **Stage 1. Manual Stateless Retraining:** Model is trained from scratch manually in an ad-hoc manner.
* **Stage 2. Automated Stateless Retraining:** A script automatically executes batch retraining from scratch. Frequency is often determined heuristically (e.g., nightly when compute is idle). This stage is mature for most companies with ML infrastructure.
* **Log and Wait (Feature Reuse):** Reusing features extracted during prediction for subsequent model retraining saves computation and reduces training-serving skew.
* **Stage 3. Automated, Stateful Training:** Automated fine-tuning (incremental learning) on new data. Requires a better model store capable of tracking **Model Lineage** (which model fine-tunes on which model) and ensuring **Streaming Features Reproducibility** (time-travel capability for debugging).
* **Stage 4. Continual Learning:** Model updates are triggered whenever data distributions shift or model performance plummets, rather than relying on a fixed schedule. The goal is to combine this with edge deployment for continuous device adaptation without centralized server cost.
  

### Continual Learning: Theory, Methods, and Applications

### Theoretical Foundation

* **Objectives:** A desirable solution for CL must ensure a proper balance between **stability** (retaining old knowledge) and **plasticity** (learning new knowledge), as well as **adequate intra/inter-task generalizability**.

* **Catastrophic Forgetting:** This dilemma reflects the trade-off between learning plasticity and memory stability.
* **Stability-Plasticity Trade-off:** Can be formulated through a Bayesian framework where the posterior distribution of old tasks acts as the prior for the new task.


### Continual Learning Methods Taxonomy

CL methods are conceptually separated into five main groups:

#### 1. Regularization-Based Approach

Adds explicit regularization terms to balance old and new tasks, often requiring a frozen copy of the old model.

* **Weight Regularization:** Penalizes the variation of parameters based on their "importance" to old tasks (e.g., EWC, SI, MAS).

* **Function Regularization (Knowledge Distillation - KD):** Uses the previously learned model as a "teacher" to mitigate forgetting in the current model ("student"). KD target data sources include new training samples, small fractions of old samples, unlabeled data, or generated data.

#### 2. Replay-Based Approach

Approximates and recovers old data distributions.

* **Experience Replay (Exemplar Replay):** Stores a few old training samples in a small memory buffer.

* **Construction:** Uses methods like Reservoir Sampling, or online continual compression (AQM, MRDC).

* **Exploitation:** Used to constrain optimization (e.g., ensuring non-increase in old task losses like GEM/A-GEM), align gradients (MER), or combined with KD (iCaRL, DER/DER++).

* **Generative Replay (Pseudo-rehearsal):** Trains an auxiliary generative model (e.g., GANs, VAEs) to sample generated data representing old distributions.

* **Feature Replay:** Focuses on maintaining feature-level distributions (e.g., prototypes, statistics, or generated features) to improve efficiency and privacy, but must counteract **representation shift**.


#### 3. Optimization-Based Approach

Designs and manipulates the optimization programs directly.

* **Gradient Projection:** Constrains parameter updates to align with or be orthogonal to the preserved knowledge (e.g., GEM, A-GEM use old samples; OWM, AOP use input space orthogonality; GPM maintains core gradient subspaces).

* **Meta-Learning:** Learning-to-learn strategies to obtain a data-driven inductive bias, often optimizing online updates to minimize interference (e.g., OML, MER).

* **Loss Landscape:** Modifying training regimes (e.g., Stable-SGD) or using methods like linear connector (MC-SGD) to converge to flat local minima, enhancing generalizability
  

#### 4. Representation-Based Approach

Creates and exploits representations for robustness.

* **Self-Supervised Learning (SSL):** Representations obtained via SSL (e.g., contrastive loss) tend to be more robust to catastrophic forgetting.

* **Pre-training for Downstream Tasks:** Downstream CL benefits from large-scale pre-training due to strong knowledge transfer and robustness.

* **Strategies:** Adapting fixed backbones using lightweight networks (Side-Tuning), parameter-efficient transfer techniques (Adapters), or prompt-based approaches (L2P, DualPrompt). Optimizing updatable backbones by seeking flat minima (F2M) or initial decorrelation (CwD).

* **Continual Pre-training (CPT):** Performing CL on the upstream pre-training data itself, which arrives incrementally.

  

#### 5. Architecture-Based Approach

Constructs task-specific/adaptive parameters using specialized architectures.

* **Parameter Allocation:** Assigns dedicated parameter subsets (using binary masks) to each task, freezing old task parameters (e.g., Piggyback, HAT). Scalability is challenged by network capacity limits.

* **Model Decomposition:** Explicitly separates models into task-sharing and task-specific components (e.g., parallel branches, adaptive layers, low-rank factorization).

* **Modular Network:** Leverages parallel sub-networks/modules, allowing explicit knowledge reuse and often requiring task identities for execution (e.g., Progressive Networks, Expert Gate, Model Zoo).

  

### Continual Learning Applications (Task Specificity and Scenario Complexity)

  

| Application/Scenario | Description & Challenges | Key Strategies |

| :--- | :--- | :--- |

| **Class-Incremental Learning (CIL)** | Tasks have disjoint label spaces; task identity is available during training but not testing (task-agnostic inference). Suffers from bias toward new classes and representation shift. | Mitigation across data (experience replay), feature (KD), and label spaces (KD). Data-Free CIL uses synthetic data (DeepInversion) or feature statistics. |

| **Few-Shot Continual Learning (FSCL/FSCIL)** | Learns a sequence of novel classes with only a small number of training samples after initial training. Leads to severe overfitting. | Strategies include preserving representation topology (TOPIC), fixing the backbone and adapting the classifier (SPPR), and meta-learning. |

| **General/Online CL (GCL/OCL/TFCL)** | Learning incremental data online, often task-free (TFCL: no task ID in training/testing) or observed in a one-pass stream (OCL). | Most OCL relies heavily on **Experience Replay**, focusing on intelligent construction and exploitation of the memory buffer (e.g., GSS, MIR, DER). |

| **Incremental Object Detection (IOD)** | Incrementally learns new object classes; old class instances in new data are often labeled as "background," exacerbating forgetting. | Knowledge Distillation (KD) is naturally powerful because old class objects can appear in new samples, constraining model response differences. |

| **Continual Semantic Segmentation (CSS)** | Pixel-wise prediction; new annotations treat old classes as background (**background shift**). | Adaptive KD from the old model is common to distinguish old classes from the background (MiB, ALIFE). Using pseudo-labels generated by the old model is also utilized. |

| **Continual Reinforcement Learning (CRL)** | Agents learn sequentially in non-stationary environments. | Uses memory replay (of old experiences) and on-policy learning (of novel experiences), often incorporating regularization (policy consolidation). |

| **Natural Language Processing (NLP) CL** | Diverse scenarios, characterized by dependence on **pre-trained Transformer architectures**. | **Parameter-efficient fine-tuning** techniques are adapted, such as adaptor-tuning (inserted layers) and prompt-tuning (trainable prompt tokens). KD and Experience Replay are also used extensively. |

  

## Online Learning: General Survey and Concepts

### Definitions and Principles

* **Online Learning (OL):** A field of ML where a learner attempts to predict/decide by learning from a sequential stream of data instances one by one.

* **Goal:** Maximize accuracy for the sequence of predictions given feedback on previous tasks.

* **Advantages over Batch Learning:** Models can be updated instantly and efficiently when new data arrives, offering better scalability for continuous data streams. OL algorithms are often simple to implement and founded on solid theory.
