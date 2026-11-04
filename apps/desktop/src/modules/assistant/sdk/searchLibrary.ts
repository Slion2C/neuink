import type { ConversationSourceLink } from '@/shared/ipc/assistantApi';
import { conversationSourceKey } from '@/shared/ipc/assistantApi';
import {
  type DegradationReason,
  type PlanResolution,
  type PlanningOutput,
  type QuestionType,
  type RetrievalTask,
  type ScopeKind
} from './retrievalPlanning';
import {
  resolveSearchScope,
  scopeRequestFromToolInput,
  type ResolvedSearchScope
} from './searchScope';

/**
 * Unified retrieval orchestration (`search_library`). The planner decides how
 * to search and this module executes each retrieval task against the existing
 * `search_segments` backend path, associating evidence per task and reporting
 * coverage without ever widening the frozen assistant scope.
 */

export type TaskEvidence = {
  entry_id: string;
  entry_title: string;
  segment_uid: string;
  page: number;
  score?: number;
  snippet?: string;
  marker: string;
};

export type TaskSearchResult = {
  evidence: TaskEvidence[];
  sources: ConversationSourceLink[];
};

export type TaskResult = {
  task_id: string;
  question: string;
  covered: boolean;
  evidence: TaskEvidence[];
};

export type SearchLibraryOutput = {
  kind: 'search_library';
  scope_kind: ScopeKind | 'full_library';
  original_question: string;
  planning: {
    status: 'planned' | 'degraded' | 'disabled';
    reason?: DegradationReason;
    attempts?: number;
    question_types: QuestionType[];
  };
  task_results: TaskResult[];
  coverage: {
    covered_tasks: string[];
    missing_tasks: string[];
  };
  warnings: string[];
};

export type SearchLibraryResult = {
  modelOutput: SearchLibraryOutput;
  sources: ConversationSourceLink[];
  summary: string;
};

/** Planning was not requested (`use_planning` false). This is not a degradation. */
type DisabledResolution = { status: 'disabled'; plan: PlanningOutput; attempts: 0 };

function effectiveTasks(plan: PlanningOutput): RetrievalTask[] {
  if (plan.retrieval_tasks.length > 0) return plan.retrieval_tasks;
  return [{ id: 'direct', question: plan.original_question, constraints: [], depends_on: [] }];
}

function scopeHintFor(scope: ResolvedSearchScope): string {
  if (scope.status === 'full_library') return 'full library';
  if (scope.status === 'empty') return 'explicit empty scope';
  return `${scope.scope_kind}: ${scope.entry_ids.length} entr${scope.entry_ids.length === 1 ? 'y' : 'ies'}`;
}

export async function runSearchLibrary({
  question,
  scopeEntryIds,
  usePlanning,
  topK,
  frozenEntryIds,
  plan,
  search,
  abortSignal
}: {
  question: string;
  scopeEntryIds: string[] | null;
  usePlanning: boolean;
  topK: number;
  frozenEntryIds: string[];
  plan: (question: string, scopeHint: string) => Promise<PlanResolution>;
  search: (task: RetrievalTask, scopeEntryIds: string[], topK: number) => Promise<TaskSearchResult>;
  abortSignal?: AbortSignal;
}): Promise<SearchLibraryResult> {
  const trimmed = question.trim();
  const scope = resolveSearchScope(scopeRequestFromToolInput(scopeEntryIds), frozenEntryIds);
  abortSignal?.throwIfAborted();

  const basePlanning = {
    status: 'disabled' as const,
    question_types: ['single'] as QuestionType[]
  };
  const noResults = (warnings: string[]): SearchLibraryResult => ({
    modelOutput: {
      kind: 'search_library',
      scope_kind: scope.status === 'empty' ? 'explicit_multi' : 'full_library',
      original_question: trimmed,
      planning: basePlanning,
      task_results: [],
      coverage: { covered_tasks: [], missing_tasks: [] },
      warnings
    },
    sources: [],
    summary: warnings.join(' ')
  });

  if (scope.status === 'empty') {
    return noResults(['范围内没有可检索条目。']);
  }

  const scopeEntryIdsForSearch = scope.status === 'full_library' ? [] : scope.entry_ids;
  const scopeHint = scopeHintFor(scope);

  const resolution: PlanResolution | DisabledResolution = usePlanning
    ? await plan(trimmed, scopeHint)
    : {
        status: 'disabled',
        plan: {
          question_types: ['single'],
          scope_kind: 'unspecified',
          original_question: trimmed,
          retrieval_tasks: [],
          synthesis_task: undefined
        },
        attempts: 0
      };
  abortSignal?.throwIfAborted();

  const warnings: string[] = [];
  if (resolution.status === 'degraded') {
    warnings.push(`检索规划未生效，已按原问题直接检索（${resolution.reason}）。`);
  }
  const planQuestions = resolution.plan.question_types;
  if (planQuestions.includes('dependent_multihop')) {
    warnings.push('依赖型多跳问题本次未执行完整依赖链，仅按原问题检索。');
  }
  if (planQuestions.includes('global')) {
    warnings.push('全局聚合问题仅返回检索到的证据，不代表穷尽覆盖。');
  }

  const tasks = effectiveTasks(resolution.plan);
  const taskResults: TaskResult[] = [];
  const sources: ConversationSourceLink[] = [];
  const seenSources = new Set<string>();

  for (const task of tasks) {
    abortSignal?.throwIfAborted();
    let result: TaskSearchResult;
    try {
      result = await search(task, scopeEntryIdsForSearch, topK);
    } catch (error) {
      taskResults.push({ task_id: task.id, question: task.question, covered: false, evidence: [] });
      warnings.push(`任务 ${task.id} 检索失败：${error instanceof Error ? error.message : String(error)}`);
      continue;
    }
    taskResults.push({ task_id: task.id, question: task.question, covered: result.evidence.length > 0, evidence: result.evidence });
    for (const source of result.sources) {
      const key = conversationSourceKey(source);
      if (!seenSources.has(key)) {
        seenSources.add(key);
        sources.push(source);
      }
    }
  }

  const coveredTasks = taskResults.filter((task) => task.covered).map((task) => task.task_id);
  const missingTasks = taskResults.filter((task) => !task.covered).map((task) => task.task_id);

  const modelOutput: SearchLibraryOutput = {
    kind: 'search_library',
    scope_kind: scope.status === 'full_library' ? 'full_library' : scope.scope_kind,
    original_question: trimmed,
    planning: {
      status: resolution.status,
      reason: resolution.status === 'degraded' ? resolution.reason : undefined,
      attempts: resolution.attempts,
      question_types: planQuestions
    },
    task_results: taskResults,
    coverage: { covered_tasks: coveredTasks, missing_tasks: missingTasks },
    warnings
  };

  const evidenceCount = taskResults.reduce((total, task) => total + task.evidence.length, 0);
  const summary = taskResults.length === 0
    ? `Retrieved evidence for "${trimmed}".`
    : `Retrieved ${evidenceCount} evidence item${evidenceCount === 1 ? '' : 's'} across ${taskResults.length} task${taskResults.length === 1 ? '' : 's'}.`;

  return { modelOutput, sources, summary };
}
