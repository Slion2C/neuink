mod document_builder;
mod embedding_provider_trait;
mod embedding_representation;
mod error;
mod fastembed_provider;
mod memory_index;
mod persistent_semantic_index;
mod query;
mod query_plan;
mod result;
mod retrieval_pipeline;
#[cfg(test)]
mod retrieval_pipeline_tests;
mod rrf;
mod search_index_trait;
mod section_tree_index;
mod semantic_index;
mod token;
mod vector_store_trait;

pub use document_builder::{compose_search_document, SearchDocumentInput};
pub use embedding_provider_trait::{
    EmbeddingInput, EmbeddingProvider, EmbeddingProviderStatus, EmbeddingVector,
};
pub use embedding_representation::{
    embed_query_vector, embedding_text, semantic_embedding_identity, stable_document_key,
    SemanticVectorDocument, EMBEDDING_REPRESENTATION_VERSION,
};
pub use error::{SearchError, SearchResult};
pub use fastembed_provider::{
    FastEmbedProvider, DEFAULT_EMBEDDING_MODEL_RESOURCE_DIR, LOCAL_EMBEDDING_BATCH_SIZE,
};
pub use memory_index::MemorySearchIndex;
pub use persistent_semantic_index::{
    merge_hybrid_results, semantic_index_path, semantic_result_mode, PersistentSemanticSearchIndex,
    INDEX_FORMAT_VERSION, SEMANTIC_INDEX_FORMAT_VERSION,
};
pub use query::{SearchInclude, SearchMode, SearchQuery, SearchScope};
pub use query_plan::{infer_query_plan, QueryModality, QueryPlan};
pub use result::{
    SearchDocument, SearchDocumentSource, SearchDocumentSourceKind, SearchEntryGroup, SearchHit,
    SearchResults, SearchTarget, SearchTextSection,
};
pub use retrieval_pipeline::*;
pub use rrf::{
    adaptive_hybrid_weights, reciprocal_rank_fusion, weighted_reciprocal_rank_fusion, DEFAULT_RRF_K,
};
pub use search_index_trait::SearchIndex;
pub use section_tree_index::{
    build_section_tree_index, segment_type_matches_filter, SectionTreeEntry, SectionTreeIndex,
    NEUTRAL_STRUCTURE_SCORE,
};
pub use semantic_index::InMemorySemanticSearchIndex;
pub use vector_store_trait::{VectorRecord, VectorSearchHit, VectorStore};
