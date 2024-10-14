
# RAGSeg Project Progress Report

## Initial Approach (from paper)

### Core Concept
- Developed RAGSeg, a novel framework leveraging retrieval-augmented generation for few-shot off-road semantic segmentation
- Aimed to improve generalization across different off-road datasets

### Key Components
1. Image Embedding: Using CLIP to map input images to a semantic embedding space
2. Retrieval Database: Large collection of diverse off-road images and segmentation maps
3. Retrieval Mechanism: System to identify relevant examples from the database
4. Generalist Segmentation Model: SegGPT, capable of incorporating in-context examples

### Initial Results
- Showed promise in cross-dataset generalization
- Outperformed baseline in some scenarios (e.g., RUGD to RELLIS: 0.7403 MIoU)

## Post-Paper Progress and Iterations

### Embedding Models
1. CLIP
   - Initially used, grounded in text-image space
   - Performed well in certain scenarios
2. DINO v2
   - Implemented to focus on image feature space
   - Initially showed promise with better spread in embedding space
   - Significantly faster embedding process (10-12x improvement)
3. Google ViT
   - Added as another embedding option

### Retrieval Mechanisms
1. Cosine Similarity
   - Initial approach for finding similar images
2. Maximum Marginal Relevance (MMR)
   - Implemented to balance between relevance and diversity
   - Showed improvements in some cases (e.g., RELLIS to RUGD: 0.7624 MIoU, up from 0.7403)

### Dataset Expansion
- Added Yamaha-CMU and DeepScene datasets to increase diversity
- Considered adding IISERB dataset

### Challenges Encountered
1. Dataset Dominance
   - Observed DeepScene dominating prompts
   - Considered normalization and weighting strategies
2. Cross-Dataset Variability
   - Different datasets label same features (e.g., puddles) as traversable/non-traversable inconsistently
3. Retrieval Issues
   - Incorrect similarity scores in vector store
   - Batch processing logic errors leading to duplicate storage

### Key Insights and Learnings
1. Robot-Specific Segmentation
   - Realized segmentation might be a function of vehicle type
   - Considered using LLMs for dynamic label pooling based on robot specifications
2. Subjectivity in Ground Truth
   - Noted inconsistencies in labeling across datasets (e.g., small bushes labeled differently based on context)
   - Recognized this subjectivity as a potential reason for challenges in off-road segmentation
3. Importance of Concrete Classes
   - Observed better performance when segmenting specific features (e.g., grass, trees) rather than abstract concepts like "traversable region"

### Alternative Approaches Explored
1. CLIPSeg
   - Tested on RAGSeg failure cases (IoU < 0.5)
   - Showed promise with concrete classes but struggled with abstract concepts
   - Considered as a potential failure handler for RAGSeg

2. Image-to-Text for Prompt Generation
   - Proposed using image-to-text models to generate appropriate prompts for CLIPSeg

## Current Status and Future Directions

### Achievements
- Resolved major issues in retrieval pipeline
- Improved cross-dataset performance in some scenarios
- Gained deeper understanding of challenges in off-road segmentation

### Ongoing Challenges
- Balancing between retrieval relevance and diversity
- Handling dataset inconsistencies and labeling subjectivity
- Improving performance on abstract concepts like "traversable region"

### Potential Next Steps
1. Implement dynamic label pooling based on robot specifications
2. Explore hybrid approaches combining RAGSeg and CLIPSeg
3. Develop methods to handle labeling subjectivity and inconsistencies across datasets
4. Investigate fine-tuning or architectural modifications to SegGPT for improved performance
5. Create a more robust evaluation methodology that accounts for dataset variations

This report summarizes the significant progress and insights gained since the initial paper, highlighting both the achievements and the challenges encountered in developing a robust off-road semantic segmentation system.