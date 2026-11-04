import { describe, expect, it, vi } from 'vitest';
import type { ConversationSourceLink } from '@/shared/ipc/assistantApi';
import { fallbackDirectPlan, type PlanResolution, type PlanningOutput } from './retrievalPlanning';
import { runSearchLibrary, type TaskEvidence, type TaskSearchResult } from './searchLibrary';

const planned = (plan: PlanningOutput): PlanResolution => ({ status: 'planned', plan, attempts: 1 });

function source(entryId: string, segmentUid: string): ConversationSourceLink {
  return { entry_id: entryId, entry_title: `title-${entryId}`, segment_uid: segmentUid, page_idx: 0, quote: `quote-${segmentUid}` };
}

function evidence(entryId: string, segmentUid: string): TaskEvidence {
  return { entry_id: entryId, entry_title: `title-${entryId}`, segment_uid: segmentUid, page: 1, score: 1, snippet: `snip-${segmentUid}`, marker: `[S1]` };
}

describe('runSearchLibrary', () => {
  it('searches the full library with the original question when planning is off', async () => {
    const search = vi.fn(async (): Promise<TaskSearchResult> => ({ evidence: [evidence('a', 'seg1')], sources: [source('a', 'seg1')] }));
    const result = await runSearchLibrary({
      question: '论文 A 使用什么数据集？',
      scopeEntryIds: null,
      usePlanning: false,
      topK: 8,
      frozenEntryIds: [],
      plan: async () => fallbackDirectPlan('论文 A 使用什么数据集？', 'no_model_configured'),
      search
    });
    expect(result.modelOutput.scope_kind).toBe('full_library');
    expect(search).toHaveBeenCalledWith(expect.objectContaining({ id: 'direct', question: '论文 A 使用什么数据集？' }), [], 8);
    expect(result.modelOutput.coverage.covered_tasks).toEqual(['direct']);
    expect(result.modelOutput.planning.status).toBe('disabled');
    expect(result.modelOutput.planning.reason).toBeUndefined();
    expect(result.modelOutput.warnings).toEqual([]);
  });

  it('resolves an explicit single scope and passes its ids through', async () => {
    const search = vi.fn(async (_task: unknown, scopeIds: string[], _k: number): Promise<TaskSearchResult> => ({ evidence: [evidence('a', 'seg1')], sources: [source('a', 'seg1')] }));
    await runSearchLibrary({
      question: '论文 A 使用什么数据集？',
      scopeEntryIds: ['a'],
      usePlanning: false,
      topK: 8,
      frozenEntryIds: [],
      plan: async () => fallbackDirectPlan('论文 A 使用什么数据集？', 'no_model_configured'),
      search
    });
    expect(search).toHaveBeenCalledWith(expect.anything(), ['a'], 8);
    expect(search.mock.calls[0][1]).toEqual(['a']);
  });

  it('returns no-results without searching when the scope is explicitly empty', async () => {
    const search = vi.fn();
    const result = await runSearchLibrary({
      question: '论文 A 使用什么数据集？',
      scopeEntryIds: [],
      usePlanning: false,
      topK: 8,
      frozenEntryIds: [],
      plan: async () => fallbackDirectPlan('论文 A 使用什么数据集？', 'no_model_configured'),
      search
    });
    expect(search).not.toHaveBeenCalled();
    expect(result.modelOutput.planning.status).toBe('disabled');
    expect(result.modelOutput.coverage.missing_tasks).toEqual([]);
    expect(result.modelOutput.warnings).toContain('范围内没有可检索条目。');
  });

  it('executes each planned task and reports coverage', async () => {
    const search = vi.fn(async (task: { id: string }): Promise<TaskSearchResult> => {
      if (task.id === 'q2') return { evidence: [], sources: [] };
      return { evidence: [evidence('a', 'seg1')], sources: [source('a', 'seg1')] };
    });
    const result = await runSearchLibrary({
      question: '论文 A 的方法和数据集是什么？',
      scopeEntryIds: ['a'],
      usePlanning: true,
      topK: 8,
      frozenEntryIds: ['a'],
      plan: async () => planned({
        question_types: ['multi_target'],
        scope_kind: 'explicit_single',
        original_question: '论文 A 的方法和数据集是什么？',
        retrieval_tasks: [
          { id: 'q1', question: '论文 A 的方法？', constraints: [], depends_on: [] },
          { id: 'q2', question: '论文 A 的数据集？', constraints: [], depends_on: [] }
        ]
      }),
      search
    });
    expect(search).toHaveBeenCalledTimes(2);
    expect(result.modelOutput.coverage.covered_tasks).toEqual(['q1']);
    expect(result.modelOutput.coverage.missing_tasks).toEqual(['q2']);
    expect(result.modelOutput.planning.status).toBe('planned');
  });

  it('deduplicates the same source across tasks without dropping either task coverage', async () => {
    const search = vi.fn(async (): Promise<TaskSearchResult> => ({ evidence: [evidence('a', 'seg1')], sources: [source('a', 'seg1')] }));
    const result = await runSearchLibrary({
      question: '比较论文 A 的方法和数据集',
      scopeEntryIds: ['a'],
      usePlanning: true,
      topK: 8,
      frozenEntryIds: ['a'],
      plan: async () => planned({
        question_types: ['multi_target'],
        scope_kind: 'explicit_single',
        original_question: '比较论文 A 的方法和数据集',
        retrieval_tasks: [
          { id: 'q1', question: '方法？', constraints: [], depends_on: [] },
          { id: 'q2', question: '数据集？', constraints: [], depends_on: [] }
        ]
      }),
      search
    });
    expect(result.sources).toHaveLength(1);
    expect(result.modelOutput.coverage.covered_tasks).toEqual(['q1', 'q2']);
  });

  it('marks dependent_multihop and global plans with coverage warnings', async () => {
    const search = vi.fn(async (): Promise<TaskSearchResult> => ({ evidence: [], sources: [] }));
    const result = await runSearchLibrary({
      question: '论文 A 效果最好的方法最初在哪篇论文提出？',
      scopeEntryIds: null,
      usePlanning: true,
      topK: 8,
      frozenEntryIds: [],
      plan: async () => planned({
        question_types: ['dependent_multihop'],
        scope_kind: 'unspecified',
        original_question: '论文 A 效果最好的方法最初在哪篇论文提出？',
        retrieval_tasks: []
      }),
      search
    });
    expect(result.modelOutput.warnings.some((warning) => warning.includes('依赖型多跳'))).toBe(true);
  });

  it('aborts before searching when the signal is already aborted', async () => {
    const controller = new AbortController();
    controller.abort();
    const search = vi.fn();
    await expect(runSearchLibrary({
      question: '论文 A 使用什么数据集？',
      scopeEntryIds: null,
      usePlanning: false,
      topK: 8,
      frozenEntryIds: [],
      plan: async () => fallbackDirectPlan('论文 A 使用什么数据集？', 'no_model_configured'),
      search,
      abortSignal: controller.signal
    })).rejects.toThrow();
    expect(search).not.toHaveBeenCalled();
  });

  it('stops before a later task when the signal is aborted mid-run', async () => {
    const controller = new AbortController();
    const search = vi.fn(async (_task: unknown, _scopeIds: string[], _k: number): Promise<TaskSearchResult> => {
      controller.abort();
      return { evidence: [evidence('a', 'seg1')], sources: [source('a', 'seg1')] };
    });
    await expect(runSearchLibrary({
      question: '论文 A 的方法和数据集是什么？',
      scopeEntryIds: ['a'],
      usePlanning: true,
      topK: 8,
      frozenEntryIds: ['a'],
      plan: async () => planned({
        question_types: ['multi_target'],
        scope_kind: 'explicit_single',
        original_question: '论文 A 的方法和数据集是什么？',
        retrieval_tasks: [
          { id: 'q1', question: '方法？', constraints: [], depends_on: [] },
          { id: 'q2', question: '数据集？', constraints: [], depends_on: [] }
        ]
      }),
      search,
      abortSignal: controller.signal
    })).rejects.toThrow();
    expect(search).toHaveBeenCalledTimes(1);
  });
});
