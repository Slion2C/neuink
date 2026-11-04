import type { ScopeKind } from './retrievalPlanning';

/**
 * Host-side scope resolution for the unified `search_library` tool.
 *
 * The planner model only guesses `scope_kind`; the host is authoritative and
 * must never let a model widen the frozen assistant scope. Backend search
 * treats an empty `scope_entry_ids` as "full library" (`SearchScope` in
 * `neuink-search`), so this module converts the tool's tri-state intent into
 * an explicit result the caller can act on without overloading that meaning.
 */

export type ScopeRequest =
  | { kind: 'unspecified' }
  | { kind: 'explicit_single' | 'explicit_multi'; entry_ids: string[] };

export type ResolvedSearchScope =
  | { status: 'full_library' }
  | { status: 'resolved'; scope_kind: ScopeKind; entry_ids: string[] }
  | { status: 'empty'; scope_kind: ScopeKind };

function uniqueNonEmpty(ids: readonly string[]) {
  return [...new Set(ids.map((id) => id.trim()).filter(Boolean))];
}

function scopeKindFor(ids: string[]): 'explicit_single' | 'explicit_multi' {
  return ids.length === 1 ? 'explicit_single' : 'explicit_multi';
}

export function resolveSearchScope(
  request: ScopeRequest,
  frozenEntryIds: readonly string[]
): ResolvedSearchScope {
  const frozen = uniqueNonEmpty(frozenEntryIds);
  const allowed = new Set(frozen);

  if (request.kind === 'unspecified') {
    if (frozen.length > 0) return { status: 'resolved', scope_kind: scopeKindFor(frozen), entry_ids: frozen };
    return { status: 'full_library' };
  }

  const requested = uniqueNonEmpty(request.entry_ids);
  if (requested.length === 0) {
    // Explicitly empty scope: report no results, never fall back to full library.
    return { status: 'empty', scope_kind: request.kind };
  }
  if (frozen.length > 0) {
    const outOfScope = requested.find((id) => !allowed.has(id));
    if (outOfScope) throw new Error('检索条目超出冻结范围，不能自动扩大到其他论文。');
  }
  return { status: 'resolved', scope_kind: scopeKindFor(requested), entry_ids: requested };
}

/** Map the raw `scope_entry_ids` tool argument to an explicit scope request. */
export function scopeRequestFromToolInput(scopeEntryIds: unknown): ScopeRequest {
  if (scopeEntryIds == null) return { kind: 'unspecified' };
  if (!Array.isArray(scopeEntryIds)) throw new Error('scope_entry_ids 必须是条目 ID 数组。');
  const entryIds = uniqueNonEmpty(scopeEntryIds);
  return {
    kind: entryIds.length === 1 ? 'explicit_single' : 'explicit_multi',
    entry_ids: entryIds
  };
}
