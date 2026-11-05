import type { ConversationSourceLink } from '@/shared/ipc/assistantApi';
import { searchSegmentsBatch, type SearchSegmentsBatchItem } from '@/shared/ipc/workspaceApi';
import { formatSearchSegmentsOutput } from './toolSupport';
import type { TaskSearchItem, TaskSearchOutcome, TaskSearchRequest } from './searchLibrary';

/**
 * Host adapter that sends every `search_library` task through one `search_segments_batch`
 * call. The backend prepares the index once for the batch. Per-task failures come back
 * as fixed codes, and their messages are replaced with host-owned copy here.
 *
 * Raw hits outside a task's allowed entries are removed before formatting, so no source
 * marker is registered for them. A cancelled run rejects at once, even when the native
 * invoke itself cannot be cancelled, and nothing is registered after cancellation.
 */

const BACKEND_ERROR_COPY: Record<string, string> = {
  invalid_query: '检索词为空，该子问题未检索。',
  search_failed: '该子问题检索失败，请稍后重试。'
};

const MALFORMED_RESULT: TaskSearchOutcome = {
  status: 'error',
  code: 'malformed_result',
  message: '检索结果格式异常，该子问题未返回证据。'
};

function abortReason(signal: AbortSignal) {
  return signal.reason instanceof Error ? signal.reason : new DOMException('The operation was aborted.', 'AbortError');
}

/** Rejects as soon as the signal aborts. The native call may still finish later, and its result is ignored. */
function raceWithAbort<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return promise;
  if (signal.aborted) return Promise.reject(abortReason(signal));
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(abortReason(signal));
    signal.addEventListener('abort', onAbort, { once: true });
    promise.then(
      (value) => {
        signal.removeEventListener('abort', onAbort);
        resolve(value);
      },
      (error: unknown) => {
        signal.removeEventListener('abort', onAbort);
        reject(error);
      }
    );
  });
}

/** Keeps only the groups and hits whose cited entry is in the allowed set. */
function restrictResults(
  results: NonNullable<SearchSegmentsBatchItem['results']>,
  allowedIds: ReadonlySet<string>
): NonNullable<SearchSegmentsBatchItem['results']> {
  const entries = results.entries.flatMap((group) => {
    if (!allowedIds.has(group.entry_id)) return [];
    const hits = group.hits.filter((hit) =>
      allowedIds.has(hit.entry_id) && allowedIds.has(hit.target.entry_id)
    );
    return hits.length > 0 ? [{ ...group, hits, hit_count: hits.length }] : [];
  });
  return { ...results, entries };
}

function outcomeFromItem(
  item: SearchSegmentsBatchItem,
  request: TaskSearchRequest,
  addSource: (source: ConversationSourceLink) => number,
  signal?: AbortSignal
): TaskSearchOutcome {
  if (item.error) {
    return {
      status: 'error',
      code: item.error.code,
      message: BACKEND_ERROR_COPY[item.error.code] ?? '该子问题检索失败。'
    };
  }
  if (!item.results) return MALFORMED_RESULT;
  signal?.throwIfAborted();
  try {
    const allowed = new Set(request.allowed_entry_ids);
    const restricted = restrictResults(item.results, allowed);
    const formatted = formatSearchSegmentsOutput(restricted, addSource, request.section_filter);
    return {
      status: 'ok',
      mode: formatted.modelOutput.mode,
      evidence: formatted.modelOutput.evidence,
      sources: formatted.sources,
      warnings: formatted.modelOutput.warnings
    };
  } catch (error) {
    if (signal?.aborted) throw error;
    return MALFORMED_RESULT;
  }
}

export function createSearchLibraryBatch({
  root,
  addSource,
  abortSignal
}: {
  root: string;
  addSource: (source: ConversationSourceLink) => number;
  abortSignal?: AbortSignal;
}) {
  return async (requests: TaskSearchRequest[]): Promise<TaskSearchItem[]> => {
    if (requests.length === 0) return [];
    abortSignal?.throwIfAborted();
    const items = await raceWithAbort(
      searchSegmentsBatch(
        root,
        requests.map((request) => ({
          task_id: request.task_id,
          query: request.query,
          mode: 'hybrid',
          top_k: request.top_k,
          scope_entry_ids: request.scope_entry_ids,
          english_query: request.english_query,
          sub_queries: request.sub_queries,
          section_filter: request.section_filter,
          segment_types: request.segment_types
        }))
      ),
      abortSignal
    );
    abortSignal?.throwIfAborted();
    const itemByTask = new Map(items.map((item) => [item.task_id, item]));
    return requests.flatMap((request) => {
      const item = itemByTask.get(request.task_id);
      return item ? [{ task_id: request.task_id, outcome: outcomeFromItem(item, request, addSource, abortSignal) }] : [];
    });
  };
}
