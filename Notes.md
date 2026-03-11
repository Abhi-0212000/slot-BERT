While the specific Theia backbone paper you linked is outside of my provided sources, I can explain exactly how to build and train this system using the mathematical principles of Vision Transformers (ViTs), as the sources extensively detail how to apply Slot Attention on top of similar ViT foundation models like DINO and MAE. 

### Do you need to train the whole backbone again?
**No, you do not need to train the whole backbone again.** You keep the pre-trained ViT backbone (in your case, Theia) **completely frozen**. You only need to train the newly added, lightweight modules: the Slot Attention module, the temporal reasoning transformer, and a small decoder. This makes the training process much faster and more memory-efficient.

### The Math and Pipeline (How it works and how to code it)
Here is the step-by-step mathematical process of what happens once you get your embeddings from 1, 2, or $n$ images, designed so you can implement it in code.

#### 1. Feature Extraction (Frozen)
Pass your image(s) through your frozen backbone. For a single image, the backbone outputs a sequence of flattened patch embeddings, denoted as $X \in \mathbb{R}^{N \times D_{feature}}$, where $N$ is the number of patches and $D_{feature}$ is the embedding dimension. 

#### 2. Slot Initialization (At time $t=1$)
You define a fixed number of empty slots, $K$ (e.g., 7 slots), with a dimension of $d_{slot}$. For the very first image in your sequence, you initialize the slot matrix $S \in \mathbb{R}^{K \times d_{slot}}$ by randomly sampling from a learned Gaussian distribution with parameters $\mu$ and $\sigma$. 

#### 3. Iterative Slot Attention (The Core Loop)
This is the routing mechanism where unstructured features $X$ are grouped into the object slots $S$. You repeat the following steps for a set number of iterations (usually 2 or 3) per image:

*   **Linear Projections:** Create keys ($k$) and values ($v$) from the image features, and queries ($q$) from the slots using learned linear weight matrices:
    $$k = W_k X \in \mathbb{R}^{N \times d_{slot}}$$
    $$v = W_v X \in \mathbb{R}^{N \times d_{slot}}$$
    $$q = W_q S \in \mathbb{R}^{K \times d_{slot}}$$
*   **Calculate Attention:** Compute the dot-product similarity between the queries and keys. **Crucial code implementation detail:** The softmax normalization must be applied across the $K$ slot dimension (the columns), not the feature dimension. This forces the slots to compete for features:
    $$A = \text{softmax}_K \left( \frac{q k^T}{\sqrt{d_{slot}}} \right) \in \mathbb{R}^{K \times N}$$
*   **Normalize and Aggregate:** Normalize the attention map so the weights for each slot sum to 1 (adding a small $\epsilon$ for numerical stability). Then, multiply by the values $v$ to get the updates:
    $$\hat{A}_{m,n} = \frac{A_{m,n}}{\sum_{l=1}^N A_{m,l} + \epsilon}$$
    $$\text{Updates} = \hat{A} v \in \mathbb{R}^{K \times d_{slot}}$$
*   **Recurrent Update:** Update the slots' internal representations using a shared Gated Recurrent Unit (GRU) taking the aggregated features as input, followed by a residual Multi-Layer Perceptron (MLP) and Layer Normalization (LN):
    $$S \leftarrow \text{GRU}(\text{Updates}, S)$$
    $$S \leftarrow S + \text{MLP}(\text{LN}(S))$$

#### 4. Processing $n$ images (Temporal Dynamics)
If you are passing $n$ sequential images (a video), you do not re-initialize the slots randomly for images $2$ through $n$. 
Instead, the final slots $S_t$ from the current image are passed into a Temporal Slot Transformer (TST) to predict the slots for the next image. 
*   Add temporal positional embeddings: $S_{pos} = S_t + P_{temporal}$.
*   Pass them through a standard Transformer Encoder using multi-head self-attention: $S_{next} = \text{TransformerEncoder}(S_{pos})$.
*   $S_{next}$ is then used as the initial $S$ query matrix for the Slot Attention loop on image $t+1$. 

#### 5. How to Actually Train It (The Loss Function)
Because the backbone is frozen, you do not train the model to reconstruct raw image pixels. Instead, you train it to **reconstruct the backbone features**.

*   **Decoding:** Pass each updated slot independently through a simple shared MLP decoder. For each slot $k$, the decoder outputs predicted features $\hat{x}_k$ and a spatial alpha mask $\alpha_k$. 
*   **Combine:** Normalize the alpha masks across all slots using softmax to determine which slot is visible where, and calculate the final reconstructed features via a weighted sum:
    $$X_{recon} = \sum_{k=1}^K \text{softmax}_K(\alpha_k) \odot \hat{x}_k$$
*   **Objective (MSE):** The network is trained end-to-end by minimizing the Mean Squared Error (MSE) between your original frozen backbone features $X$ and the reconstructed features $X_{recon}$:
    $$\mathcal{L}_{recon} = \|X_{recon} - X\|_2^2$$
*   **Optional Contrastive Loss:** To prevent slots from collapsing into the same object, you can add a slot contrastive loss that calculates the cosine similarity between the slot vectors in a frame and penalizes them for being too mathematically similar, forcing them to capture distinct objects.