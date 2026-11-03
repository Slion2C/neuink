use std::{
    collections::{HashMap, HashSet},
    fs,
    hash::{DefaultHasher, Hash, Hasher},
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
    thread,
    time::{Duration, Instant, SystemTime},
};
use rayon::prelude::*;

use neuink_domain::{CanonicalSectionRole, EntryId, SegmentType};
use neuink_search::{
    semantic_index_path, semantic_result_mode, EmbeddingProvider, FastEmbedProvider,
    MemorySearchIndex, PersistentSemanticSearchIndex, SearchDocument, SearchDocumentSource,
    SearchDocumentSourceKind, SearchEntryGroup, SearchInclude, SearchIndex, SearchMode,
    SearchQuery, SearchResults, SearchScope, SearchTarget, SearchTextSection,
};
use neuink_workspace::{
    Workspace, WorkspaceLayout, WorkspaceSearchOptions, WorkspaceSearchRecord,
    WorkspaceSearchRecordKind,
};
use serde::{de::IgnoredAny, Deserialize, Serialize};

use super::embedding_resources::embedding_model_dir;

const SEARCH_INDEX_CACHE_LIMIT: usize = 8;
const SEMANTIC_INDEX_CACHE_LIMIT: usize = 3;
const EMBEDDING_PROVIDER_CACHE_LIMIT: usize = 1;
const CACHE_TTL: Duration = Duration::from_secs(60 * 30);
const CACHE_CLEANUP_INTERVAL: Duration = Duration::from_secs(60 * 5);
const DISK_VECTOR_CACHE_TTL: Duration = Duration::from_secs(60 * 60 * 24 * 14);
const SEMANTIC_INDEX_FORMAT_VERSION: u32 = 5;

#[derive(Debug, Deserialize)]
pub struct SearchEntriesRequest {
    pub root: PathBuf,
    pub query: String,
    #[serde(default)]
    pub english_query: Option<String>,
    #[serde(default)]
    pub sub_queries: Vec<String>,
    #[serde(default)]
    pub section_filter: Option<String>,
    #[serde(default)]
    pub segment_types: Vec<String>,
    #[serde(default)]
    pub scope_entry_ids: Vec<EntryId>,
    #[serde(default)]
    pub mode: SearchMode,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct SearchSegmentsRequest {
    pub root: PathBuf,
    pub query: String,
    #[serde(default)]
    pub english_query: Option<String>,
    #[serde(default)]
    pub sub_queries: Vec<String>,
    #[serde(default)]
    pub section_filter: Option<String>,
    #[serde(default)]
    pub segment_types: Vec<String>,
    #[serde(default)]
    pub scope_entry_ids: Vec<EntryId>,
    #[serde(default)]
    pub mode: SearchMode,
    #[serde(default)]
    pub top_k: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct SearchIndexStatusRequest {
    pub root: PathBuf,
    #[serde(default)]
    pub segments_only: bool,
}

#[derive(Debug, Deserialize)]
pub struct RebuildSearchIndexRequest {
    pub root: PathBuf,
    #[serde(default)]
    pub segments_only: bool,
}

#[derive(Debug, Serialize)]
pub struct SearchIndexStatus {
    pub scope: String,
    pub semantic_status: SemanticIndexState,
    pub document_count: usize,
    pub semantic_document_count: usize,
    pub records_fingerprint: String,
    pub keyword_memory_cache_ready: bool,
    pub semantic_memory_cache_ready: bool,
    pub semantic_disk_cache_ready: bool,
    pub semantic_disk_cache_path: String,
    pub semantic_disk_cache_record_count: Option<usize>,
    pub semantic_disk_cache_modified_at_ms: Option<u64>,
    pub message: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticIndexState {
    Empty,
    ReadyMemory,
    ReadyDisk,
    NeedsBuild,
}

#[derive(Debug, Serialize)]
pub struct RebuildSearchIndexResponse {
    pub status: SearchIndexStatus,
    pub rebuilt_vector_count: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchIndexBuildState {
    Idle,
    Queued,
    Running,
    Ready,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SearchIndexBuildStatus {
    pub root: String,
    pub state: SearchIndexBuildState,
    pub scope: String,
    pub phase: String,
    pub completed: usize,
    pub total: usize,
    pub message: String,
    pub error: Option<String>,
    pub started_at_ms: u64,
    pub updated_at_ms: u64,
}

struct CachedWorkspaceIndex {
    fingerprint: u64,
    index: Arc<MemorySearchIndex>,
    last_used: Instant,
}

struct CachedSemanticIndex {
    fingerprint: u64,
    index: Arc<PersistentSemanticSearchIndex>,
    last_used: Instant,
}

struct CachedEmbeddingProvider {
    provider: Arc<FastEmbedProvider>,
    last_used: Instant,
}

static SEARCH_INDEX_CACHE: OnceLock<Mutex<HashMap<String, CachedWorkspaceIndex>>> = OnceLock::new();
static SEMANTIC_INDEX_CACHE: OnceLock<Mutex<HashMap<String, CachedSemanticIndex>>> =
    OnceLock::new();
static EMBEDDING_PROVIDER_CACHE: OnceLock<Mutex<HashMap<String, CachedEmbeddingProvider>>> =
    OnceLock::new();
static DISK_VECTOR_CACHE_LAST_CLEANUP: OnceLock<Mutex<HashMap<String, Instant>>> = OnceLock::new();
static SEARCH_BUILD_STATUS: OnceLock<Mutex<HashMap<String, SearchIndexBuildStatus>>> =
    OnceLock::new();
static SEMANTIC_BUILD_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
static SEMANTIC_AUTO_REBUILD_KEYS: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

pub fn spawn_search_cache_workers<R: tauri::Runtime>(_app: tauri::AppHandle<R>) {
    // Keep cache eviction alive at startup. Legacy semantic snapshots are
    // rebuilt lazily in a detached worker on the first semantic request, so
    // keyword search remains available while the new index is prepared.
    thread::Builder::new()
        .name("neuink-search-cache-cleanup".to_string())
        .spawn(clean_search_caches_loop)
        .ok();
}

#[tauri::command]
pub async fn search_entries<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
    request: SearchEntriesRequest,
) -> Result<SearchResults, String> {
    let embedding_model_dir = embedding_model_dir(&app)?;
    tokio::task::spawn_blocking(move || {
        let primary_query = request.query.trim().to_string();
        if primary_query.is_empty() {
            return Err("query is required".to_string());
        }
        let mut queries = vec![(primary_query.clone(), 1.0_f32)];
        if let Some(english_query) = request
            .english_query
            .as_deref()
            .map(str::trim)
            .filter(|query| !query.is_empty())
        {
            queries.push((english_query.to_string(), 0.9));
        }
        for sub_query in request
            .sub_queries
            .into_iter()
            .map(|query| query.trim().to_string())
            .filter(|query| !query.is_empty())
            .take(3)
        {
            queries.push((sub_query, 0.75));
        }
        let scope_entry_ids = request.scope_entry_ids;
        let section_filter = request.section_filter.clone();
        let segment_types = parse_segment_types(&request.segment_types);
        let search_results = queries
            .par_iter()
            .map(|(query, _)| {
                run_search(
                    embedding_model_dir.clone(),
                    request.root.clone(),
                    query.clone(),
                    scope_entry_ids.clone(),
                    request.mode,
                    request.limit.unwrap_or(40),
                    WorkspaceSearchOptions::default(),
                    SearchInclude::default(),
                    section_filter.clone(),
                    segment_types.clone(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let weights = queries.iter().map(|(_, weight)| *weight).collect::<Vec<_>>();
        let mut results = merge_query_results(primary_query, search_results, &weights);
        rerank_section_hits(&mut results, section_filter.as_deref());
        if contains_cjk(&results.query)
            && request.english_query.as_deref().is_none_or(|query| query.trim().is_empty())
            && (results.mode.contains("fallback") || results.total_hit_count == 0)
        {
            results.warnings.push(
                "中文查询未提供 english_query；若目标是英文论文，当前结果可能缺少跨语种匹配。"
                    .to_string(),
            );
        }
        Ok(results)
    })
    .await
    .map_err(|error| error.to_string())?
}

#[tauri::command]
pub async fn search_segments<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
    request: SearchSegmentsRequest,
) -> Result<SearchResults, String> {
    let embedding_model_dir = embedding_model_dir(&app)?;
    // Must match the scope `rebuild_search_index` embeds, otherwise the semantic
    // cache namespace never exists and hybrid search silently stays keyword-only.
    let (options, include) = search_scope_options(true);
    let top_k = request.top_k.unwrap_or(20).clamp(1, 20);
    let candidate_k = top_k.saturating_mul(5).max(60);
    let primary_query = request.query.trim().to_string();
    if primary_query.is_empty() {
        return Err("query is required".to_string());
    }
    let mut queries = vec![(primary_query.clone(), 1.0_f32)];
    if let Some(english_query) = request
        .english_query
        .as_deref()
        .map(str::trim)
        .filter(|query| !query.is_empty())
    {
        queries.push((english_query.to_string(), 0.9));
    }
    for sub_query in request
        .sub_queries
        .into_iter()
        .map(|query| query.trim().to_string())
        .filter(|query| !query.is_empty())
        .take(3)
    {
        queries.push((sub_query, 0.75));
    }
    let scope_entry_ids = request.scope_entry_ids;
    let section_filter = request.section_filter.clone();
    let segment_types = parse_segment_types(&request.segment_types);
    let mode = request.mode;
    let has_english_query = queries.len() > 1;
    tokio::task::spawn_blocking(move || {
        let search_results = queries
            .par_iter()
            .map(|(query, _)| {
                run_search(
                    embedding_model_dir.clone(),
                    request.root.clone(),
                    query.clone(),
                    scope_entry_ids.clone(),
                    mode,
                    candidate_k,
                    options.clone(),
                    include.clone(),
                    section_filter.clone(),
                    segment_types.clone(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let weights = queries.iter().map(|(_, weight)| *weight).collect::<Vec<_>>();
        let mut results = merge_query_results(primary_query, search_results, &weights);
        if contains_cjk(&results.query)
            && !has_english_query
            && (results.mode.contains("fallback") || results.total_hit_count == 0)
        {
            results.warnings.push(
                "中文查询未提供 english_query，且语义索引不可用；英文论文可能没有关键词交集，请配置模型或同时提供英文检索词。"
                    .to_string(),
            );
        }
        Ok(filter_segment_search_results(results, top_k))
    })
    .await
    .map_err(|error| error.to_string())?
}

fn contains_cjk(text: &str) -> bool {
    text.chars().any(|ch| matches!(ch, '\u{3400}'..='\u{4dbf}' | '\u{4e00}'..='\u{9fff}'))
}

fn parse_segment_types(values: &[String]) -> Vec<SegmentType> {
    values
        .iter()
        .filter_map(|value| match value.trim().to_ascii_lowercase().as_str() {
            "paragraph" => Some(SegmentType::Paragraph),
            "heading" => Some(SegmentType::Heading),
            "table" => Some(SegmentType::Table),
            "figure" => Some(SegmentType::Figure),
            "math" | "equation" => Some(SegmentType::Math),
            "code" => Some(SegmentType::Code),
            "list" => Some(SegmentType::List),
            "aside_text" | "aside" => Some(SegmentType::AsideText),
            "page_footnote" | "footnote" => Some(SegmentType::PageFootnote),
            _ => None,
        })
        .collect()
}

fn merge_query_results(
    query: String,
    results: Vec<SearchResults>,
    weights: &[f32],
) -> SearchResults {
    let Some(first) = results.first() else {
        return SearchResults {
            query,
            mode: "keyword".to_string(),
            index_generation: 0,
            total_hit_count: 0,
            entries: Vec::new(),
            warnings: Vec::new(),
        };
    };
    if results.len() == 1 {
        let mut result = first.clone();
        result.query = query;
        return result;
    }

    let ranked = results
        .iter()
        .map(|result| {
            result
                .entries
                .iter()
                .flat_map(|entry| entry.hits.iter())
                .map(|hit| search_hit_key(hit))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let fused = neuink_search::weighted_reciprocal_rank_fusion(&ranked, weights, 60.0);
    let mut hits_by_key = std::collections::BTreeMap::new();
    for result in results.iter() {
        for hit in result.entries.iter().flat_map(|entry| entry.hits.iter()) {
            hits_by_key
                .entry(search_hit_key(hit))
                .or_insert_with(|| hit.clone());
        }
    }
    let mut hits = fused
        .into_iter()
        .filter_map(|(key, score)| {
            let mut hit = hits_by_key.remove(&key)?;
            hit.score = score;
            Some(hit)
        })
        .collect::<Vec<_>>();
    hits.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.entry_title.cmp(&right.entry_title))
            .then_with(|| left.title.cmp(&right.title))
    });
    let mut warnings = results
        .iter()
        .flat_map(|result| result.warnings.iter().cloned())
        .collect::<Vec<_>>();
    warnings.sort();
    warnings.dedup();
    let mode = if first.mode.contains("fallback") {
        first.mode.clone()
    } else {
        "hybrid".to_string()
    };
    let mut grouped = std::collections::BTreeMap::<String, SearchEntryGroup>::new();
    for hit in hits {
        let key = hit.entry_id.as_str().to_string();
        let group = grouped.entry(key).or_insert_with(|| SearchEntryGroup {
            entry_id: hit.entry_id.clone(),
            entry_title: hit.entry_title.clone(),
            hit_count: 0,
            max_score: hit.score,
            hits: Vec::new(),
        });
        group.hit_count += 1;
        group.max_score = group.max_score.max(hit.score);
        group.hits.push(hit);
    }
    let mut entries = grouped.into_values().collect::<Vec<_>>();
    entries.sort_by(|left, right| {
        right
            .max_score
            .partial_cmp(&left.max_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.entry_title.cmp(&right.entry_title))
    });
    SearchResults {
        query,
        mode,
        index_generation: first.index_generation,
        total_hit_count: entries.iter().map(|entry| entry.hit_count).sum(),
        entries,
        warnings,
    }
}

fn search_hit_key(hit: &neuink_search::SearchHit) -> String {
    match &hit.target {
        neuink_search::SearchTarget::Segment {
            entry_id,
            segment_uid,
            ..
        } => format!("segment:{}:{}", entry_id, segment_uid),
        neuink_search::SearchTarget::Page { entry_id, page_idx } => {
            format!("page:{}:{}", entry_id, page_idx)
        }
        neuink_search::SearchTarget::Note { entry_id, note_id } => {
            format!("note:{}:{}", entry_id, note_id)
        }
        neuink_search::SearchTarget::Entry { entry_id } => format!("entry:{}", entry_id),
    }
}

#[tauri::command]
pub async fn get_search_index_status(
    request: SearchIndexStatusRequest,
) -> Result<SearchIndexStatus, String> {
    tokio::task::spawn_blocking(move || get_search_index_status_impl(request))
        .await
        .map_err(|error| error.to_string())?
}

#[tauri::command]
pub async fn get_search_index_build_status(
    request: SearchIndexStatusRequest,
) -> Result<SearchIndexBuildStatus, String> {
    tokio::task::spawn_blocking(move || load_search_build_status(&request.root))
        .await
        .map_err(|error| error.to_string())?
}

#[tauri::command]
pub async fn rebuild_search_index<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
    request: RebuildSearchIndexRequest,
) -> Result<RebuildSearchIndexResponse, String> {
    let embedding_model_dir = embedding_model_dir(&app)?;
    tokio::task::spawn_blocking(move || {
        begin_search_build(&request.root, "手动重建向量索引");
        let root = request.root.clone();
        let result = rebuild_search_index_impl(embedding_model_dir, request);
        if let Err(error) = &result {
            fail_search_build(&root, error);
        }
        result
    })
    .await
    .map_err(|error| error.to_string())?
}

fn run_search(
    embedding_model_dir: PathBuf,
    root: PathBuf,
    query: String,
    scope_entry_ids: Vec<EntryId>,
    mode: SearchMode,
    limit: usize,
    options: WorkspaceSearchOptions,
    include: SearchInclude,
    section_filter: Option<String>,
    segment_types: Vec<SegmentType>,
) -> Result<SearchResults, String> {
    let workspace = Workspace::open(&root).map_err(|error| error.to_string())?;
    clean_disk_vector_cache(&root);
    let records = workspace
        .collect_search_records(options.clone())
        .map_err(|error| error.to_string())?;
    let documents = records
        .iter()
        .cloned()
        .filter_map(search_document)
        .collect::<Vec<_>>();
    let records_fingerprint = records_fingerprint(&records);
    let semantic_fingerprint = semantic_records_fingerprint(&records);
    let index = cached_index(&root, &options, records_fingerprint, documents.clone())?;
    let search_query = SearchQuery {
        text: query,
        mode,
        scope: SearchScope {
            entry_ids: scope_entry_ids,
        },
        include,
        section_filter,
        segment_types,
        limit,
    };

    if mode == SearchMode::Keyword {
        return index
            .search(search_query)
            .map_err(|error| error.to_string());
    }

    match run_semantic_search(
        embedding_model_dir,
        &index,
        &documents,
        &root,
        &options,
        semantic_fingerprint,
        search_query.clone(),
    ) {
        Ok(results) => Ok(results),
        Err(error) => keyword_fallback(&index, search_query, error),
    }
}

fn get_search_index_status_impl(
    request: SearchIndexStatusRequest,
) -> Result<SearchIndexStatus, String> {
    let (scope, options, include) = search_scope_status_options(request.segments_only);
    let workspace = Workspace::open(&request.root).map_err(|error| error.to_string())?;
    let records = workspace
        .collect_search_records(options.clone())
        .map_err(|error| error.to_string())?;
    let documents = records
        .iter()
        .cloned()
        .filter_map(search_document)
        .collect::<Vec<_>>();
    let semantic_document_count = documents
        .iter()
        .filter(|document| search_include_contains(&include, document.source.kind))
        .count();
    let records_fingerprint = records_fingerprint(&records);
    let semantic_fingerprint = semantic_records_fingerprint(&records);
    let keyword_cache_key = workspace_cache_key(&request.root, &options);
    let keyword_memory_cache_ready = SEARCH_INDEX_CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .ok()
        .and_then(|cache| {
            cache
                .get(&keyword_cache_key)
                .map(|cached| cached.fingerprint == records_fingerprint)
        })
        .unwrap_or(false);

    let semantic_cache_key = semantic_cache_key(&request.root, &options, &include);
    let semantic_memory_cache_ready = SEMANTIC_INDEX_CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .ok()
        .and_then(|cache| {
            cache
                .get(&semantic_cache_key)
                .map(|cached| cached.fingerprint == semantic_fingerprint)
        })
        .unwrap_or(false);

    let cache_dir = WorkspaceLayout::new(request.root.clone()).cache_dir();
    let namespace = semantic_cache_namespace(&request.root, &options, &include);
    let semantic_disk_cache_path = semantic_index_path(cache_dir, &namespace);
    let disk_cache = read_semantic_disk_cache_status(&semantic_disk_cache_path);
    let semantic_disk_cache_ready = disk_cache
        .as_ref()
        .is_some_and(|status| status.current_for(semantic_fingerprint));
    let semantic_disk_cache_modified_at_ms = semantic_disk_cache_path
        .metadata()
        .ok()
        .and_then(|metadata| metadata.modified().ok())
        .and_then(|modified| modified.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|duration| duration.as_millis() as u64);
    let semantic_disk_cache_record_count = disk_cache.map(|status| status.record_count);
    let semantic_status = semantic_index_state(
        semantic_document_count,
        semantic_memory_cache_ready,
        semantic_disk_cache_ready,
    );
    let message = search_index_status_message(semantic_document_count, &semantic_status);

    Ok(SearchIndexStatus {
        scope,
        semantic_status,
        document_count: documents.len(),
        semantic_document_count,
        records_fingerprint: records_fingerprint.to_string(),
        keyword_memory_cache_ready,
        semantic_memory_cache_ready,
        semantic_disk_cache_ready,
        semantic_disk_cache_path: semantic_disk_cache_path.display().to_string(),
        semantic_disk_cache_record_count,
        semantic_disk_cache_modified_at_ms,
        message,
    })
}

fn rebuild_search_index_impl(
    embedding_model_dir: PathBuf,
    request: RebuildSearchIndexRequest,
) -> Result<RebuildSearchIndexResponse, String> {
    // The manual build intentionally keeps the previous disk snapshot so
    // `open_or_build_with_progress` only embeds documents that are new or
    // changed since the last build instead of re-embedding the workspace.
    let workspace = Workspace::open(&request.root).map_err(|error| error.to_string())?;
    let provider =
        cached_embedding_provider(embedding_model_dir).map_err(|error| error.to_string())?;

    // A global build also refreshes the segments-only scope so assistant
    // segment search keeps working after the startup warmup was removed.
    let scopes = if request.segments_only {
        vec![search_scope_options(true)]
    } else {
        vec![search_scope_options(false), search_scope_options(true)]
    };
    let mut semantic_document_count = 0;
    for (options, include) in scopes {
        // Each scope must fingerprint the exact record set its searches use,
        // otherwise the disk snapshot never matches at query time.
        let records = workspace
            .collect_search_records(options.clone())
            .map_err(|error| error.to_string())?;
        let documents = records
            .iter()
            .cloned()
            .filter_map(search_document)
            .collect::<Vec<_>>();
        semantic_document_count = semantic_document_count.max(documents.len());
        if documents.is_empty() {
            continue;
        }

        let records_fingerprint = records_fingerprint(&records);
        let semantic_fingerprint = semantic_records_fingerprint(&records);
        let keyword_index = cached_index(
            &request.root,
            &options,
            records_fingerprint,
            documents.clone(),
        )?;
        let _ = keyword_index.generation();
        let _ = cached_semantic_index(
            &request.root,
            &options,
            semantic_fingerprint,
            &include,
            &documents,
            provider.clone(),
            Some(if options.include_segments && !options.include_entry_meta {
                "segments"
            } else {
                "global"
            }),
        )
        .map_err(|error| error.to_string())?;
    }

    let status = get_search_index_status_impl(SearchIndexStatusRequest {
        root: request.root,
        segments_only: request.segments_only,
    })?;
    Ok(RebuildSearchIndexResponse {
        status,
        rebuilt_vector_count: semantic_document_count,
    })
}

fn cached_index(
    root: &PathBuf,
    options: &WorkspaceSearchOptions,
    fingerprint: u64,
    documents: Vec<SearchDocument>,
) -> Result<Arc<MemorySearchIndex>, String> {
    let cache_key = workspace_cache_key(root, options);
    let cache = SEARCH_INDEX_CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    if let Some(index) = cache
        .lock()
        .map_err(|_| "search cache lock is poisoned".to_string())?
        .get_mut(&cache_key)
        .filter(|cached| cached.fingerprint == fingerprint)
        .map(|cached| {
            cached.last_used = Instant::now();
            cached.index.clone()
        })
    {
        return Ok(index);
    }

    let index = Arc::new(MemorySearchIndex::default());
    index
        .replace_documents(documents)
        .map_err(|error| error.to_string())?;
    let mut cache = cache
        .lock()
        .map_err(|_| "search cache lock is poisoned".to_string())?;
    cache.insert(
        cache_key,
        CachedWorkspaceIndex {
            fingerprint,
            index: index.clone(),
            last_used: Instant::now(),
        },
    );
    trim_workspace_cache(&mut cache);
    Ok(index)
}

fn run_semantic_search(
    embedding_model_dir: PathBuf,
    keyword_index: &MemorySearchIndex,
    documents: &[SearchDocument],
    root: &PathBuf,
    options: &WorkspaceSearchOptions,
    records_fingerprint: u64,
    query: SearchQuery,
) -> Result<SearchResults, neuink_search::SearchError> {
    let normalized_query = query.normalized_text();
    if normalized_query.is_empty() {
        return Err(neuink_search::SearchError::EmptyQuery);
    }

    let provider = cached_embedding_provider(embedding_model_dir)?;
    let Some(semantic_index) = try_cached_semantic_index(
        root,
        options,
        records_fingerprint,
        &query.include,
        documents,
    ) else {
        schedule_legacy_semantic_rebuild(
            root,
            options,
            records_fingerprint,
            &query.include,
            documents,
            provider,
        );
        return Err(neuink_search::SearchError::EmbeddingUnavailable(
            "向量索引尚未构建或正在后台重建；当前返回关键词结果。".to_string(),
        ));
    };
    let candidate_limit = query.limit.saturating_mul(5).max(60);
    let mut candidate_query = query.clone();
    candidate_query.limit = candidate_limit;
    let semantic_query = if query.mode == SearchMode::Hybrid {
        &candidate_query
    } else {
        &query
    };
    let semantic_results = semantic_index.search(
        semantic_query,
        &normalized_query,
        provider.as_ref(),
        semantic_result_mode(query.mode),
        Vec::new(),
        keyword_index.generation()?,
    )?;

    if query.mode == SearchMode::Semantic {
        return Ok(semantic_results);
    }

    let mut keyword_query = candidate_query;
    keyword_query.mode = SearchMode::Keyword;
    let keyword_results = keyword_index.search(keyword_query)?;
    Ok(neuink_search::merge_hybrid_results(
        normalized_query,
        keyword_index.generation()?,
        keyword_results,
        semantic_results,
        query.limit,
        Vec::new(),
    ))
}

fn cached_embedding_provider(
    embedding_model_dir: PathBuf,
) -> Result<Arc<FastEmbedProvider>, neuink_search::SearchError> {
    let cache_key = embedding_model_dir.display().to_string();
    let cache = EMBEDDING_PROVIDER_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| neuink_search::SearchError::LockPoisoned)?;
    if let Some(cached) = cache.get_mut(&cache_key) {
        cached.last_used = Instant::now();
        return Ok(cached.provider.clone());
    }

    let provider = Arc::new(FastEmbedProvider::from_model_dir(embedding_model_dir));
    if !provider.status().available {
        return Ok(provider);
    }
    cache.insert(
        cache_key,
        CachedEmbeddingProvider {
            provider: provider.clone(),
            last_used: Instant::now(),
        },
    );
    trim_embedding_provider_cache(&mut cache);
    Ok(provider)
}

fn cached_semantic_index(
    root: &PathBuf,
    options: &WorkspaceSearchOptions,
    records_fingerprint: u64,
    include: &SearchInclude,
    documents: &[SearchDocument],
    provider: Arc<FastEmbedProvider>,
    progress_scope: Option<&str>,
) -> Result<Arc<PersistentSemanticSearchIndex>, neuink_search::SearchError> {
    let cache_key = semantic_cache_key(root, options, include);
    let cache = SEMANTIC_INDEX_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(index) = cache
        .lock()
        .map_err(|_| neuink_search::SearchError::LockPoisoned)?
        .get_mut(&cache_key)
        .filter(|cached| cached.fingerprint == records_fingerprint)
        .map(|cached| {
            cached.last_used = Instant::now();
            cached.index.clone()
        })
    {
        return Ok(index);
    }

    let _build_guard = SEMANTIC_BUILD_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .map_err(|_| neuink_search::SearchError::LockPoisoned)?;
    if let Some(index) = cache
        .lock()
        .map_err(|_| neuink_search::SearchError::LockPoisoned)?
        .get_mut(&cache_key)
        .filter(|cached| cached.fingerprint == records_fingerprint)
        .map(|cached| {
            cached.last_used = Instant::now();
            cached.index.clone()
        })
    {
        return Ok(index);
    }

    let cache_dir = WorkspaceLayout::new(root.clone()).cache_dir();
    let namespace = semantic_cache_namespace(root, options, include);
    let index_path = semantic_index_path(cache_dir, &namespace);
    let scope = progress_scope.unwrap_or("semantic");
    let semantic_total = documents
        .iter()
        .filter(|document| search_include_contains(include, document.source.kind))
        .count();
    update_search_build_progress(
        root,
        scope,
        "embedding",
        0,
        semantic_total,
        format!("正在构建{}向量索引", search_scope_display_name(scope)),
    );
    let progress = |completed: usize, total: usize| {
        update_search_build_progress(
            root,
            scope,
            "embedding",
            completed,
            total,
            format!(
                "正在构建{}向量索引 · {completed}/{total}",
                search_scope_display_name(scope)
            ),
        );
    };
    let index_result = PersistentSemanticSearchIndex::open_or_build_with_progress(
        index_path,
        documents,
        provider.as_ref(),
        include,
        records_fingerprint,
        &progress,
    );
    let index = match index_result {
        Ok(index) => Arc::new(index),
        Err(error) => {
            fail_search_build(root, &error.to_string());
            return Err(error);
        }
    };
    let mut cache = cache
        .lock()
        .map_err(|_| neuink_search::SearchError::LockPoisoned)?;
    cache.insert(
        cache_key,
        CachedSemanticIndex {
            fingerprint: records_fingerprint,
            index: index.clone(),
            last_used: Instant::now(),
        },
    );
    trim_semantic_cache(&mut cache);
    finish_search_build(
        root,
        format!(
            "{}向量索引已就绪 · {}",
            search_scope_display_name(scope),
            index.document_count()
        ),
    );
    Ok(index)
}

/// Returns a semantic index only when it is already available (memory cache or
/// a fingerprint-current disk snapshot). Never triggers embedding work, so
/// search requests stay cheap until the user explicitly builds the index.
fn try_cached_semantic_index(
    root: &PathBuf,
    options: &WorkspaceSearchOptions,
    records_fingerprint: u64,
    include: &SearchInclude,
    documents: &[SearchDocument],
) -> Option<Arc<PersistentSemanticSearchIndex>> {
    let cache_key = semantic_cache_key(root, options, include);
    let cache = SEMANTIC_INDEX_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(index) = cache
        .lock()
        .ok()?
        .get_mut(&cache_key)
        .filter(|cached| cached.fingerprint == records_fingerprint)
        .map(|cached| {
            cached.last_used = Instant::now();
            cached.index.clone()
        })
    {
        return Some(index);
    }

    let cache_dir = WorkspaceLayout::new(root.clone()).cache_dir();
    let namespace = semantic_cache_namespace(root, options, include);
    let index_path = semantic_index_path(cache_dir, &namespace);
    let index = PersistentSemanticSearchIndex::open_cached(
        &index_path,
        documents,
        include,
        records_fingerprint,
    )
    .ok()??;
    let index = Arc::new(index);
    if let Ok(mut cache) = cache.lock() {
        cache.insert(
            cache_key,
            CachedSemanticIndex {
                fingerprint: records_fingerprint,
                index: index.clone(),
                last_used: Instant::now(),
            },
        );
        trim_semantic_cache(&mut cache);
    }
    Some(index)
}

fn schedule_legacy_semantic_rebuild(
    root: &PathBuf,
    options: &WorkspaceSearchOptions,
    records_fingerprint: u64,
    include: &SearchInclude,
    documents: &[SearchDocument],
    provider: Arc<FastEmbedProvider>,
) {
    let cache_dir = WorkspaceLayout::new(root.clone()).cache_dir();
    let namespace = semantic_cache_namespace(root, options, include);
    let index_path = semantic_index_path(cache_dir, &namespace);
    let cache_status = read_semantic_disk_cache_status(&index_path);
    if !should_schedule_legacy_semantic_rebuild(cache_status.as_ref(), provider.status().available)
    {
        return;
    }

    let active_key = semantic_cache_key(root, options, include);
    let active = SEMANTIC_AUTO_REBUILD_KEYS.get_or_init(|| Mutex::new(HashSet::new()));
    let Ok(mut active_keys) = active.lock() else {
        return;
    };
    if !active_keys.insert(active_key.clone()) {
        return;
    }
    drop(active_keys);

    let root = root.clone();
    let options = options.clone();
    let include = include.clone();
    let documents = documents.to_vec();
    let scope = if options.include_segments && !options.include_entry_meta {
        "segments"
    } else {
        "global"
    };
    let worker_key = active_key.clone();
    let spawn = thread::Builder::new()
        .name("neuink-search-semantic-auto-rebuild".to_string())
        .spawn(move || {
            let result = cached_semantic_index(
                &root,
                &options,
                records_fingerprint,
                &include,
                &documents,
                provider,
                Some(scope),
            );
            if let Err(error) = result {
                fail_search_build(&root, &error.to_string());
            }
            if let Ok(mut active_keys) = SEMANTIC_AUTO_REBUILD_KEYS
                .get_or_init(|| Mutex::new(HashSet::new()))
                .lock()
            {
                active_keys.remove(&worker_key);
            }
        });
    if spawn.is_err() {
        if let Ok(mut active_keys) = active.lock() {
            active_keys.remove(&active_key);
        }
    }
}

fn should_schedule_legacy_semantic_rebuild(
    cache_status: Option<&SemanticDiskCacheStatus>,
    provider_available: bool,
) -> bool {
    provider_available
        && cache_status.is_some_and(|status| status.version != SEMANTIC_INDEX_FORMAT_VERSION)
}

fn semantic_cache_key(
    root: &PathBuf,
    options: &WorkspaceSearchOptions,
    include: &SearchInclude,
) -> String {
    format!(
        "{}|{}:{}:{}",
        workspace_cache_key(root, options),
        include.entry_meta,
        include.notes,
        include.segments
    )
}

/// The segments scope is the assistant's `search_segments` corpus: parsed
/// segments plus the user's notes and annotations anchored to them.
fn search_scope_options(segments_only: bool) -> (WorkspaceSearchOptions, SearchInclude) {
    if segments_only {
        (
            WorkspaceSearchOptions {
                include_entry_meta: false,
                include_notes: true,
                include_segments: true,
            },
            SearchInclude {
                entry_meta: false,
                notes: true,
                segments: true,
            },
        )
    } else {
        (WorkspaceSearchOptions::default(), SearchInclude::default())
    }
}

fn search_scope_status_options(
    segments_only: bool,
) -> (String, WorkspaceSearchOptions, SearchInclude) {
    let (options, include) = search_scope_options(segments_only);
    (
        if segments_only { "segments" } else { "global" }.to_string(),
        options,
        include,
    )
}

fn search_scope_display_name(scope: &str) -> &'static str {
    if scope == "segments" {
        "片段"
    } else {
        "全局"
    }
}

fn keyword_fallback(
    keyword_index: &MemorySearchIndex,
    mut query: SearchQuery,
    error: neuink_search::SearchError,
) -> Result<SearchResults, String> {
    let requested_mode = query.mode;
    query.mode = SearchMode::Keyword;
    let mut results = keyword_index
        .search(query)
        .map_err(|keyword_error| keyword_error.to_string())?;
    results.mode = match requested_mode {
        SearchMode::Semantic => "semantic_fallback_keyword".to_string(),
        SearchMode::Hybrid => "hybrid_fallback_keyword".to_string(),
        SearchMode::Keyword => "keyword".to_string(),
    };
    results.warnings.push(format!(
        "Embedding search is unavailable: {error}. Showing keyword fallback results."
    ));
    Ok(results)
}

fn load_search_build_status(root: &PathBuf) -> Result<SearchIndexBuildStatus, String> {
    let key = root.display().to_string();
    let cache = SEARCH_BUILD_STATUS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(status) = cache
        .lock()
        .map_err(|_| "search build status lock is poisoned".to_string())?
        .get(&key)
        .cloned()
    {
        return Ok(status);
    }

    let status = fs::read(search_build_status_path(root))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<SearchIndexBuildStatus>(&bytes).ok())
        .unwrap_or_else(|| idle_search_build_status(root));
    if matches!(
        &status.state,
        SearchIndexBuildState::Queued | SearchIndexBuildState::Running
    ) {
        // Builds run inside the current application process and cannot survive
        // a restart. If an in-progress state exists only on disk, the previous
        // process stopped before it could record a terminal state. Recover to
        // idle instead of presenting or resuming a phantom background build.
        let recovered = interrupted_search_build_status(root);
        store_search_build_status(root, recovered.clone());
        return Ok(recovered);
    }
    cache
        .lock()
        .map_err(|_| "search build status lock is poisoned".to_string())?
        .insert(key, status.clone());
    Ok(status)
}

fn idle_search_build_status(root: &PathBuf) -> SearchIndexBuildStatus {
    let now = unix_time_ms();
    SearchIndexBuildStatus {
        root: root.display().to_string(),
        state: SearchIndexBuildState::Idle,
        scope: "all".to_string(),
        phase: "idle".to_string(),
        completed: 0,
        total: 0,
        message: "等待构建向量索引".to_string(),
        error: None,
        started_at_ms: now,
        updated_at_ms: now,
    }
}

fn interrupted_search_build_status(root: &PathBuf) -> SearchIndexBuildStatus {
    let mut status = idle_search_build_status(root);
    status.message = "上次向量索引构建已中断；等待用户手动构建".to_string();
    status
}

fn begin_search_build(root: &PathBuf, message: &str) {
    let now = unix_time_ms();
    store_search_build_status(
        root,
        SearchIndexBuildStatus {
            root: root.display().to_string(),
            state: SearchIndexBuildState::Queued,
            scope: "all".to_string(),
            phase: "queued".to_string(),
            completed: 0,
            total: 0,
            message: message.to_string(),
            error: None,
            started_at_ms: now,
            updated_at_ms: now,
        },
    );
}

fn update_search_build_progress(
    root: &PathBuf,
    scope: &str,
    phase: &str,
    completed: usize,
    total: usize,
    message: String,
) {
    let previous =
        load_search_build_status(root).unwrap_or_else(|_| idle_search_build_status(root));
    store_search_build_status(
        root,
        SearchIndexBuildStatus {
            root: root.display().to_string(),
            state: SearchIndexBuildState::Running,
            scope: scope.to_string(),
            phase: phase.to_string(),
            completed,
            total,
            message,
            error: None,
            started_at_ms: previous.started_at_ms,
            updated_at_ms: unix_time_ms(),
        },
    );
}

fn finish_search_build(root: &PathBuf, message: String) {
    let previous =
        load_search_build_status(root).unwrap_or_else(|_| idle_search_build_status(root));
    store_search_build_status(
        root,
        SearchIndexBuildStatus {
            root: root.display().to_string(),
            state: SearchIndexBuildState::Ready,
            scope: previous.scope,
            phase: "complete".to_string(),
            completed: previous.total,
            total: previous.total,
            message,
            error: None,
            started_at_ms: previous.started_at_ms,
            updated_at_ms: unix_time_ms(),
        },
    );
}

fn fail_search_build(root: &PathBuf, error: &str) {
    let previous =
        load_search_build_status(root).unwrap_or_else(|_| idle_search_build_status(root));
    store_search_build_status(
        root,
        SearchIndexBuildStatus {
            root: root.display().to_string(),
            state: SearchIndexBuildState::Failed,
            scope: previous.scope,
            phase: "failed".to_string(),
            completed: previous.completed,
            total: previous.total,
            message: "向量索引构建失败".to_string(),
            error: Some(error.to_string()),
            started_at_ms: previous.started_at_ms,
            updated_at_ms: unix_time_ms(),
        },
    );
}

fn store_search_build_status(root: &PathBuf, status: SearchIndexBuildStatus) {
    let key = root.display().to_string();
    if let Ok(mut cache) = SEARCH_BUILD_STATUS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
    {
        cache.insert(key, status.clone());
        while cache.len() > 4 {
            let Some(oldest_key) = cache
                .iter()
                .min_by_key(|(_, cached)| cached.updated_at_ms)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            cache.remove(&oldest_key);
        }
    }

    let path = search_build_status_path(root);
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(bytes) = serde_json::to_vec_pretty(&status) {
        let _ = fs::write(path, bytes);
    }
}

fn search_build_status_path(root: &PathBuf) -> PathBuf {
    WorkspaceLayout::new(root.clone())
        .cache_dir()
        .join("search")
        .join("build-status.json")
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}

fn clean_search_caches_loop() {
    loop {
        thread::sleep(CACHE_CLEANUP_INTERVAL);
        clean_search_caches();
    }
}

fn clean_search_caches() {
    let cutoff = Instant::now() - CACHE_TTL;
    if let Some(cache) = SEARCH_INDEX_CACHE.get() {
        if let Ok(mut cache) = cache.lock() {
            cache.retain(|_, cached| cached.last_used >= cutoff);
            trim_workspace_cache(&mut cache);
        }
    }
    if let Some(cache) = SEMANTIC_INDEX_CACHE.get() {
        if let Ok(mut cache) = cache.lock() {
            cache.retain(|_, cached| cached.last_used >= cutoff);
            trim_semantic_cache(&mut cache);
        }
    }
    if let Some(cache) = EMBEDDING_PROVIDER_CACHE.get() {
        if let Ok(mut cache) = cache.lock() {
            cache.retain(|_, cached| cached.last_used >= cutoff);
            trim_embedding_provider_cache(&mut cache);
        }
    }
}

fn clean_disk_vector_cache(root: &PathBuf) {
    let root_key = root.display().to_string();
    let last_cleanup = DISK_VECTOR_CACHE_LAST_CLEANUP.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(mut last_cleanup) = last_cleanup.lock() {
        if last_cleanup
            .get(&root_key)
            .is_some_and(|last| last.elapsed() < CACHE_CLEANUP_INTERVAL)
        {
            return;
        }
        last_cleanup.insert(root_key, Instant::now());
    }

    let search_dir = WorkspaceLayout::new(root.clone())
        .cache_dir()
        .join("search");
    let Ok(entries) = fs::read_dir(search_dir) else {
        return;
    };
    let cutoff = SystemTime::now()
        .checked_sub(DISK_VECTOR_CACHE_TTL)
        .unwrap_or(SystemTime::UNIX_EPOCH);

    for entry in entries.flatten() {
        let path = entry.path();
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !file_name.starts_with("semantic-") || !file_name.ends_with(".vectors.json") {
            continue;
        }
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        let Ok(modified) = metadata.modified() else {
            continue;
        };
        if modified < cutoff {
            let _ = fs::remove_file(path);
        }
    }
}

fn trim_workspace_cache(cache: &mut HashMap<String, CachedWorkspaceIndex>) {
    trim_cache_by_last_used(cache, SEARCH_INDEX_CACHE_LIMIT, |cached| cached.last_used);
}

fn trim_semantic_cache(cache: &mut HashMap<String, CachedSemanticIndex>) {
    trim_cache_by_last_used(cache, SEMANTIC_INDEX_CACHE_LIMIT, |cached| cached.last_used);
}

fn trim_embedding_provider_cache(cache: &mut HashMap<String, CachedEmbeddingProvider>) {
    trim_cache_by_last_used(cache, EMBEDDING_PROVIDER_CACHE_LIMIT, |cached| {
        cached.last_used
    });
}

fn trim_cache_by_last_used<T>(
    cache: &mut HashMap<String, T>,
    limit: usize,
    last_used: impl Fn(&T) -> Instant,
) {
    while cache.len() > limit {
        let Some(oldest_key) = cache
            .iter()
            .min_by_key(|(_, cached)| last_used(cached))
            .map(|(key, _)| key.clone())
        else {
            return;
        };
        cache.remove(&oldest_key);
    }
}

fn search_document(record: WorkspaceSearchRecord) -> Option<SearchDocument> {
    let source_kind = source_kind(record.kind);
    let target = match record.kind {
        WorkspaceSearchRecordKind::EntryTitle
        | WorkspaceSearchRecordKind::EntryField
        | WorkspaceSearchRecordKind::EntryTag => SearchTarget::Entry {
            entry_id: record.entry_id.clone(),
        },
        WorkspaceSearchRecordKind::NoteTitle | WorkspaceSearchRecordKind::NoteBody => {
            SearchTarget::Note {
                entry_id: record.entry_id.clone(),
                note_id: record.note_id.clone()?,
            }
        }
        WorkspaceSearchRecordKind::PdfPage => SearchTarget::Page {
            entry_id: record.entry_id.clone(),
            page_idx: record.page_idx?,
        },
        WorkspaceSearchRecordKind::Segment
        | WorkspaceSearchRecordKind::SegmentNote
        | WorkspaceSearchRecordKind::Annotation => SearchTarget::Segment {
            entry_id: record.entry_id.clone(),
            segment_uid: record.segment_uid.clone()?,
            page_idx: record.page_idx?,
        },
    };
    let label = source_label(&record);

    let title_section = if record.kind == WorkspaceSearchRecordKind::Segment {
        SearchTextSection::with_boost("title", record.title.clone(), 0.25)
    } else {
        SearchTextSection::title(record.title.clone())
    };

    Some(SearchDocument {
        entry_id: record.entry_id,
        entry_title: record.entry_title,
        source: SearchDocumentSource {
            kind: source_kind,
            label,
            field_name: record.field_name,
            tag_id: record.tag_id,
            note_id: record.note_id,
            segment_uid: record.segment_uid,
            page_idx: record.page_idx,
            section_id: record.section_id.clone(),
            section_path: record.section_path.clone(),
            section_role: record.section_role,
            segment_type: record.segment_type,
        },
        target,
        title: record.title,
        sections: {
            let mut sections = vec![title_section];
            if let Some(path) = record.section_path.clone().filter(|path| !path.trim().is_empty()) {
                sections.push(SearchTextSection::with_boost("section_path", path, 1.6));
            }
            if let Some(caption) = record.caption.clone().filter(|caption| !caption.trim().is_empty()) {
                sections.push(SearchTextSection::with_boost("caption", caption, 1.8));
            }
            sections.push(SearchTextSection::body(record.text));
            if let Some(translation) = record.translation {
                sections.push(SearchTextSection::with_boost(
                    "translation",
                    translation,
                    0.85,
                ));
            }
            sections
        },
        boost: source_boost(record.kind)
            * if record.section_role == Some(CanonicalSectionRole::References) {
                0.35
            } else {
                1.0
            },
    })
}

fn filter_segment_search_results(mut results: SearchResults, limit: usize) -> SearchResults {
    let mut hits = results
        .entries
        .into_iter()
        .flat_map(|entry| entry.hits)
        .filter(|hit| matches!(&hit.target, SearchTarget::Segment { .. }))
        .collect::<Vec<_>>();
    hits.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.entry_title.cmp(&right.entry_title))
            .then_with(|| left.title.cmp(&right.title))
    });
    hits.truncate(limit);

    let mut grouped = std::collections::BTreeMap::<String, SearchEntryGroup>::new();
    for hit in hits {
        let key = hit.entry_id.as_str().to_string();
        let group = grouped.entry(key).or_insert_with(|| SearchEntryGroup {
            entry_id: hit.entry_id.clone(),
            entry_title: hit.entry_title.clone(),
            hit_count: 0,
            max_score: hit.score,
            hits: Vec::new(),
        });
        group.hit_count += 1;
        group.max_score = group.max_score.max(hit.score);
        group.hits.push(hit);
    }
    let mut entries = grouped.into_values().collect::<Vec<_>>();
    entries.sort_by(|left, right| {
        right
            .max_score
            .partial_cmp(&left.max_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.entry_title.cmp(&right.entry_title))
    });
    results.entries = entries;
    results.total_hit_count = results.entries.iter().map(|entry| entry.hit_count).sum();
    results
}

fn rerank_section_hits(results: &mut SearchResults, section_filter: Option<&str>) {
    let plan = neuink_search::infer_query_plan(&results.query);
    let filter_mentions_references = section_filter
        .unwrap_or_default()
        .to_ascii_lowercase()
        .contains("reference");
    for hit in results.entries.iter_mut().flat_map(|entry| entry.hits.iter_mut()) {
        let role_bonus = plan
            .section_role
            .zip(hit.source.section_role)
            .map(|(expected, actual)| if expected == actual { 1.15 } else { 1.0 })
            .unwrap_or(1.0);
        let modality_bonus = match (plan.modality, hit.source.segment_type) {
            (neuink_search::QueryModality::Table, Some(SegmentType::Table))
            | (neuink_search::QueryModality::Figure, Some(SegmentType::Figure))
            | (neuink_search::QueryModality::Math, Some(SegmentType::Math)) => 1.15,
            _ => 1.0,
        };
        let reference_penalty = if hit.source.section_role == Some(CanonicalSectionRole::References)
            && !filter_mentions_references
        {
            0.35
        } else {
            1.0
        };
        hit.score *= role_bonus * modality_bonus * reference_penalty;
    }
    for entry in &mut results.entries {
        entry.hits.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        entry.max_score = entry.hits.first().map(|hit| hit.score).unwrap_or(0.0);
    }
    results.entries.sort_by(|left, right| {
        right
            .max_score
            .partial_cmp(&left.max_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

fn source_kind(kind: WorkspaceSearchRecordKind) -> SearchDocumentSourceKind {
    match kind {
        WorkspaceSearchRecordKind::EntryTitle => SearchDocumentSourceKind::EntryTitle,
        WorkspaceSearchRecordKind::EntryField => SearchDocumentSourceKind::EntryField,
        WorkspaceSearchRecordKind::EntryTag => SearchDocumentSourceKind::EntryTag,
        WorkspaceSearchRecordKind::NoteTitle => SearchDocumentSourceKind::NoteTitle,
        WorkspaceSearchRecordKind::NoteBody => SearchDocumentSourceKind::NoteBody,
        WorkspaceSearchRecordKind::SegmentNote => SearchDocumentSourceKind::SegmentNote,
        WorkspaceSearchRecordKind::Annotation => SearchDocumentSourceKind::Annotation,
        WorkspaceSearchRecordKind::PdfPage => SearchDocumentSourceKind::PdfPage,
        WorkspaceSearchRecordKind::Segment => SearchDocumentSourceKind::Segment,
    }
}

fn source_boost(kind: WorkspaceSearchRecordKind) -> f32 {
    match kind {
        WorkspaceSearchRecordKind::EntryTitle => 3.0,
        WorkspaceSearchRecordKind::EntryTag => 2.4,
        WorkspaceSearchRecordKind::EntryField => 1.8,
        WorkspaceSearchRecordKind::NoteTitle => 1.8,
        WorkspaceSearchRecordKind::NoteBody => 1.1,
        WorkspaceSearchRecordKind::SegmentNote => 1.35,
        WorkspaceSearchRecordKind::Annotation => 1.45,
        WorkspaceSearchRecordKind::PdfPage => 1.15,
        WorkspaceSearchRecordKind::Segment => 1.0,
    }
}

fn source_label(record: &WorkspaceSearchRecord) -> String {
    match record.kind {
        WorkspaceSearchRecordKind::EntryTitle => "Entry".to_string(),
        WorkspaceSearchRecordKind::EntryField => record
            .field_name
            .as_ref()
            .map(|name| format!("Field · {name}"))
            .unwrap_or_else(|| "Field".to_string()),
        WorkspaceSearchRecordKind::EntryTag => record
            .tag_path
            .as_ref()
            .map(|path| format!("Tag · {path}"))
            .unwrap_or_else(|| "Tag".to_string()),
        WorkspaceSearchRecordKind::NoteTitle => "Note title".to_string(),
        WorkspaceSearchRecordKind::NoteBody => "Note".to_string(),
        WorkspaceSearchRecordKind::SegmentNote => record
            .page_idx
            .map(|page_idx| format!("Segment Note · Page {}", page_idx + 1))
            .unwrap_or_else(|| "Segment Note".to_string()),
        WorkspaceSearchRecordKind::Annotation => record
            .page_idx
            .map(|page_idx| format!("Annotation · Page {}", page_idx + 1))
            .unwrap_or_else(|| "Annotation".to_string()),
        WorkspaceSearchRecordKind::PdfPage => record
            .page_idx
            .map(|page_idx| format!("PDF Page · Page {}", page_idx + 1))
            .unwrap_or_else(|| "PDF Page".to_string()),
        WorkspaceSearchRecordKind::Segment => record
            .page_idx
            .map(|page_idx| match record.segment_type {
                Some(segment_type) => {
                    format!(
                        "PDF Segment · Page {} · {}",
                        page_idx + 1,
                        segment_type_label(segment_type)
                    )
                }
                None => format!("PDF Segment · Page {}", page_idx + 1),
            })
            .unwrap_or_else(|| "PDF Segment".to_string()),
    }
}

fn segment_type_label(segment_type: neuink_domain::SegmentType) -> &'static str {
    match segment_type {
        neuink_domain::SegmentType::Paragraph => "paragraph",
        neuink_domain::SegmentType::Heading => "heading",
        neuink_domain::SegmentType::Table => "table",
        neuink_domain::SegmentType::Math => "math",
        neuink_domain::SegmentType::Figure => "figure",
        neuink_domain::SegmentType::Code => "code",
        neuink_domain::SegmentType::List => "list",
        neuink_domain::SegmentType::PageHeader => "page_header",
        neuink_domain::SegmentType::PageFooter => "page_footer",
        neuink_domain::SegmentType::PageNumber => "page_number",
        neuink_domain::SegmentType::AsideText => "aside_text",
        neuink_domain::SegmentType::PageFootnote => "page_footnote",
    }
}

fn search_include_contains(include: &SearchInclude, kind: SearchDocumentSourceKind) -> bool {
    match kind {
        SearchDocumentSourceKind::EntryTitle
        | SearchDocumentSourceKind::EntryField
        | SearchDocumentSourceKind::EntryTag => include.entry_meta,
        SearchDocumentSourceKind::NoteTitle
        | SearchDocumentSourceKind::NoteBody
        | SearchDocumentSourceKind::SegmentNote
        | SearchDocumentSourceKind::Annotation => include.notes,
        SearchDocumentSourceKind::PdfPage | SearchDocumentSourceKind::Segment => include.segments,
    }
}

fn semantic_index_state(
    semantic_document_count: usize,
    semantic_memory_cache_ready: bool,
    semantic_disk_cache_ready: bool,
) -> SemanticIndexState {
    if semantic_document_count == 0 {
        SemanticIndexState::Empty
    } else if semantic_memory_cache_ready {
        SemanticIndexState::ReadyMemory
    } else if semantic_disk_cache_ready {
        SemanticIndexState::ReadyDisk
    } else {
        SemanticIndexState::NeedsBuild
    }
}

fn search_index_status_message(
    semantic_document_count: usize,
    semantic_status: &SemanticIndexState,
) -> String {
    match semantic_status {
        SemanticIndexState::Empty => "暂无可向量化内容".to_string(),
        SemanticIndexState::ReadyMemory => {
            format!("向量已就绪 · {semantic_document_count}")
        }
        SemanticIndexState::ReadyDisk => format!("磁盘缓存 · {semantic_document_count}"),
        SemanticIndexState::NeedsBuild => {
            format!("待构建向量 · {semantic_document_count}")
        }
    }
}

fn workspace_cache_key(root: &PathBuf, options: &WorkspaceSearchOptions) -> String {
    format!(
        "{}|{}:{}:{}",
        root.display(),
        options.include_entry_meta,
        options.include_notes,
        options.include_segments
    )
}

fn semantic_cache_namespace(
    root: &PathBuf,
    options: &WorkspaceSearchOptions,
    include: &SearchInclude,
) -> String {
    let mut hasher = DefaultHasher::new();
    workspace_cache_key(root, options).hash(&mut hasher);
    include.entry_meta.hash(&mut hasher);
    include.notes.hash(&mut hasher);
    include.segments.hash(&mut hasher);
    format!("semantic-{:016x}", hasher.finish())
}

fn records_fingerprint(records: &[WorkspaceSearchRecord]) -> u64 {
    records_fingerprint_with_translation(records, true)
}

fn semantic_records_fingerprint(records: &[WorkspaceSearchRecord]) -> u64 {
    records_fingerprint_with_translation(records, false)
}

fn records_fingerprint_with_translation(
    records: &[WorkspaceSearchRecord],
    include_translation: bool,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    records.len().hash(&mut hasher);
    let mut record_fingerprints = Vec::with_capacity(records.len());
    for record in records {
        let mut record_hasher = DefaultHasher::new();
        record.entry_id.hash(&mut record_hasher);
        record.entry_title.hash(&mut record_hasher);
        record.kind.hash(&mut record_hasher);
        record.title.hash(&mut record_hasher);
        record.text.hash(&mut record_hasher);
        record.field_name.hash(&mut record_hasher);
        record.tag_id.hash(&mut record_hasher);
        record.tag_path.hash(&mut record_hasher);
        record.note_id.hash(&mut record_hasher);
        record.segment_uid.hash(&mut record_hasher);
        record.page_idx.hash(&mut record_hasher);
        record.segment_type.hash(&mut record_hasher);
        record.section_id.hash(&mut record_hasher);
        record.section_path.hash(&mut record_hasher);
        record.section_role.hash(&mut record_hasher);
        record.caption.hash(&mut record_hasher);
        if include_translation {
            record.translation.hash(&mut record_hasher);
        }
        record_fingerprints.push(record_hasher.finish());
    }
    record_fingerprints.sort_unstable();
    for fingerprint in record_fingerprints {
        fingerprint.hash(&mut hasher);
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use std::{fs, time::SystemTime};

    use super::{
        begin_search_build, filter_segment_search_results, load_search_build_status,
        records_fingerprint, search_build_status_path, search_scope_options,
        semantic_cache_namespace, update_search_build_progress, SearchIndexBuildState,
        SearchIndexBuildStatus, WorkspaceSearchRecord, WorkspaceSearchRecordKind,
    };
    use neuink_domain::EntryId;
    use neuink_search::{
        SearchDocumentSource, SearchDocumentSourceKind, SearchEntryGroup, SearchHit, SearchInclude,
        SearchResults, SearchTarget,
    };
    use neuink_workspace::WorkspaceSearchOptions;

    #[test]
    fn segments_scope_matches_assistant_search_corpus() {
        // `search_segments` searches segments plus anchored notes; the manual
        // segments build must embed the same namespace or hybrid never finds it.
        let root = std::path::PathBuf::from("workspace");
        let assistant_options = WorkspaceSearchOptions {
            include_entry_meta: false,
            include_notes: true,
            include_segments: true,
        };
        let assistant_include = SearchInclude {
            entry_meta: false,
            notes: true,
            segments: true,
        };
        let (options, include) = search_scope_options(true);

        assert_eq!(include, assistant_include);
        assert_eq!(
            semantic_cache_namespace(&root, &options, &include),
            semantic_cache_namespace(&root, &assistant_options, &assistant_include)
        );
    }

    #[test]
    fn segment_scope_filters_pages_and_unanchored_notes_after_candidate_expansion() {
        let entry_id = EntryId::from_string("entry-a");
        let segment_uid = neuink_domain::SegmentUid::from_string("segment-a");
        let hit = |target, kind| SearchHit {
            entry_id: entry_id.clone(),
            entry_title: "Paper".to_string(),
            source: SearchDocumentSource {
                kind,
                ..Default::default()
            },
            target,
            title: "hit".to_string(),
            snippet: "body".to_string(),
            score: 1.0,
            matched_terms: Vec::new(),
        };
        let results = SearchResults {
            query: "q".to_string(),
            mode: "keyword".to_string(),
            index_generation: 1,
            total_hit_count: 3,
            entries: vec![SearchEntryGroup {
                entry_id: entry_id.clone(),
                entry_title: "Paper".to_string(),
                hit_count: 3,
                max_score: 1.0,
                hits: vec![
                    hit(
                        SearchTarget::Page {
                            entry_id: entry_id.clone(),
                            page_idx: 0,
                        },
                        SearchDocumentSourceKind::PdfPage,
                    ),
                    hit(
                        SearchTarget::Note {
                            entry_id: entry_id.clone(),
                            note_id: neuink_domain::NoteId::from_string("note-a"),
                        },
                        SearchDocumentSourceKind::NoteBody,
                    ),
                    hit(
                        SearchTarget::Segment {
                            entry_id: entry_id.clone(),
                            segment_uid,
                            page_idx: 0,
                        },
                        SearchDocumentSourceKind::Segment,
                    ),
                ],
            }],
            warnings: Vec::new(),
        };

        let filtered = filter_segment_search_results(results, 1);

        assert_eq!(filtered.total_hit_count, 1);
        assert!(matches!(
            filtered.entries[0].hits[0].target,
            SearchTarget::Segment { .. }
        ));
    }

    #[test]
    fn records_fingerprint_is_independent_of_directory_iteration_order() {
        let first = record("entry-a", "Alpha");
        let second = record("entry-b", "Beta");
        assert_eq!(
            records_fingerprint(&[first.clone(), second.clone()]),
            records_fingerprint(&[second, first])
        );
    }

    #[test]
    fn records_fingerprint_changes_when_searchable_content_changes() {
        let first = record("entry-a", "Alpha");
        let mut changed = first.clone();
        changed.text = "Changed".to_string();
        assert_ne!(
            records_fingerprint(&[first]),
            records_fingerprint(&[changed])
        );
    }

    #[test]
    fn legacy_semantic_cache_is_rebuilt_only_when_provider_and_version_require_it() {
        let legacy = super::SemanticDiskCacheStatus {
            version: super::SEMANTIC_INDEX_FORMAT_VERSION - 1,
            fingerprint: 1,
            record_count: 2,
        };
        let current = super::SemanticDiskCacheStatus {
            version: super::SEMANTIC_INDEX_FORMAT_VERSION,
            fingerprint: 1,
            record_count: 2,
        };

        assert!(super::should_schedule_legacy_semantic_rebuild(
            Some(&legacy),
            true
        ));
        assert!(!super::should_schedule_legacy_semantic_rebuild(
            Some(&legacy),
            false
        ));
        assert!(!super::should_schedule_legacy_semantic_rebuild(
            Some(&current),
            true
        ));
        assert!(!super::should_schedule_legacy_semantic_rebuild(None, true));
    }

    #[test]
    fn search_build_progress_is_persisted_outside_the_ui() {
        let root = std::env::temp_dir().join(format!(
            "neuink-search-build-status-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&root).expect("test root");

        begin_search_build(&root, "queued");
        update_search_build_progress(&root, "global", "embedding", 4, 10, "building".to_string());
        let status = load_search_build_status(&root).expect("persisted status");

        assert!(matches!(status.state, SearchIndexBuildState::Running));
        assert_eq!(status.completed, 4);
        assert_eq!(status.total, 10);
        assert!(search_build_status_path(&root).is_file());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn persisted_in_progress_build_is_recovered_as_manual_idle_state() {
        for (suffix, state) in [
            ("queued", SearchIndexBuildState::Queued),
            ("running", SearchIndexBuildState::Running),
        ] {
            let root = std::env::temp_dir().join(format!(
                "neuink-search-interrupted-status-{suffix}-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .expect("clock")
                    .as_nanos()
            ));
            let path = search_build_status_path(&root);
            fs::create_dir_all(path.parent().expect("status parent")).expect("status directory");
            let persisted = SearchIndexBuildStatus {
                root: root.display().to_string(),
                state,
                scope: "global".to_string(),
                phase: "embedding".to_string(),
                completed: 4,
                total: 10,
                message: "building".to_string(),
                error: None,
                started_at_ms: 1,
                updated_at_ms: 2,
            };
            fs::write(
                &path,
                serde_json::to_vec_pretty(&persisted).expect("serialize persisted status"),
            )
            .expect("write persisted status");

            let recovered = load_search_build_status(&root).expect("recover interrupted status");
            assert!(matches!(recovered.state, SearchIndexBuildState::Idle));
            assert_eq!(recovered.completed, 0);
            assert_eq!(recovered.total, 0);
            assert_eq!(
                recovered.message,
                "上次向量索引构建已中断；等待用户手动构建"
            );

            let stored = serde_json::from_slice::<SearchIndexBuildStatus>(
                &fs::read(&path).expect("read recovered status"),
            )
            .expect("deserialize recovered status");
            assert!(matches!(stored.state, SearchIndexBuildState::Idle));
            let _ = fs::remove_dir_all(root);
        }
    }

    fn record(entry_id: &str, text: &str) -> WorkspaceSearchRecord {
        WorkspaceSearchRecord {
            entry_id: EntryId::from_string(entry_id.to_string()),
            entry_title: text.to_string(),
            kind: WorkspaceSearchRecordKind::EntryTitle,
            title: text.to_string(),
            text: text.to_string(),
            field_name: None,
            tag_id: None,
            tag_path: None,
            note_id: None,
            segment_uid: None,
            page_idx: None,
            segment_type: None,
            section_id: None,
            section_path: None,
            section_role: None,
            caption: None,
            translation: None,
        }
    }
}

#[derive(Debug, Deserialize)]
struct SemanticDiskCacheSnapshot {
    version: u32,
    fingerprint: u64,
    records: Vec<IgnoredAny>,
}

#[derive(Debug)]
struct SemanticDiskCacheStatus {
    version: u32,
    fingerprint: u64,
    record_count: usize,
}

impl SemanticDiskCacheStatus {
    fn current_for(&self, fingerprint: u64) -> bool {
        self.version == SEMANTIC_INDEX_FORMAT_VERSION && self.fingerprint == fingerprint
    }
}

fn read_semantic_disk_cache_status(path: &PathBuf) -> Option<SemanticDiskCacheStatus> {
    let bytes = fs::read(path).ok()?;
    let snapshot = serde_json::from_slice::<SemanticDiskCacheSnapshot>(&bytes).ok()?;
    Some(SemanticDiskCacheStatus {
        version: snapshot.version,
        fingerprint: snapshot.fingerprint,
        record_count: snapshot.records.len(),
    })
}
