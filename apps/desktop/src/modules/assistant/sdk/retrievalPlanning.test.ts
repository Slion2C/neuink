import { describe, expect, it, vi } from 'vitest';
import {
  buildPlannerPrompt,
  fallbackDirectPlan,
  planRetrieval,
  PLANNER_SYSTEM_PROMPT,
  validatePlanningOutput
} from './retrievalPlanning';

const multiTargetPlan = {
  question_types: ['multi_target'],
  scope_kind: 'explicit_single',
  original_question: '论文 A 用了什么方法，在什么数据集上评测？',
  retrieval_tasks: [
    { id: 'q1', question: '论文 A 使用了什么方法？' },
    { id: 'q2', question: '论文 A 使用了哪些评测数据集？' }
  ],
  synthesis_task: '整理方法和数据集，并分别给出来源'
};

describe('validatePlanningOutput', () => {
  it('accepts a valid multi-target plan', () => {
    const result = validatePlanningOutput(multiTargetPlan);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.plan.retrieval_tasks).toHaveLength(2);
      expect(result.plan.scope_kind).toBe('explicit_single');
    }
  });

  it('defaults scope_kind to unspecified when absent', () => {
    const { scope_kind: _scope, ...withoutScope } = multiTargetPlan;
    const result = validatePlanningOutput(withoutScope);
    expect(result.ok).toBe(true);
    if (result.ok) expect(result.plan.scope_kind).toBe('unspecified');
  });

  it('rejects duplicate retrieval task ids', () => {
    const result = validatePlanningOutput({
      ...multiTargetPlan,
      retrieval_tasks: [
        { id: 'q1', question: 'a' },
        { id: 'q1', question: 'b' }
      ]
    });
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.issues.join(' ')).toContain('duplicate');
  });

  it('rejects dependencies on unknown or self tasks', () => {
    const unknown = validatePlanningOutput({
      ...multiTargetPlan,
      retrieval_tasks: [{ id: 'q1', question: 'a', depends_on: ['missing'] }]
    });
    expect(unknown.ok).toBe(false);

    const self = validatePlanningOutput({
      ...multiTargetPlan,
      retrieval_tasks: [{ id: 'q1', question: 'a', depends_on: ['q1'] }]
    });
    expect(self.ok).toBe(false);
  });

  it('requires decomposition plans to carry tasks, and forbids single plans from decomposing', () => {
    const emptyDecomposition = validatePlanningOutput({ ...multiTargetPlan, retrieval_tasks: [] });
    expect(emptyDecomposition.ok).toBe(false);

    const decomposedSingle = validatePlanningOutput({
      question_types: ['single'],
      scope_kind: 'unspecified',
      original_question: '论文 A 使用什么数据集？',
      retrieval_tasks: [{ id: 'q1', question: '数据集是什么？' }]
    });
    expect(decomposedSingle.ok).toBe(false);
  });

  it('accepts a single direct plan with no tasks', () => {
    const result = validatePlanningOutput({
      question_types: ['single'],
      scope_kind: 'unspecified',
      original_question: '论文 A 使用什么数据集？',
      retrieval_tasks: []
    });
    expect(result.ok).toBe(true);
  });
});

describe('fallbackDirectPlan', () => {
  it('returns a degraded single plan preserving the question', () => {
    const resolution = fallbackDirectPlan('  论文 A 的方法是什么？  ', 'plan_invalid');
    expect(resolution.status).toBe('degraded');
    expect(resolution.reason).toBe('plan_invalid');
    expect(resolution.plan.question_types).toEqual(['single']);
    expect(resolution.plan.original_question).toBe('论文 A 的方法是什么？');
    expect(resolution.plan.retrieval_tasks).toEqual([]);
  });
});

describe('planRetrieval', () => {
  it('returns the validated plan on the first valid response', async () => {
    const runModel = async () => multiTargetPlan;
    const result = await planRetrieval({ question: multiTargetPlan.original_question, runModel });
    expect(result.status).toBe('planned');
    if (result.status === 'planned') expect(result.plan.retrieval_tasks).toHaveLength(2);
  });

  it('retries with feedback and degrades when the model never returns a valid plan', async () => {
    const runModel = vi.fn(async (_system: string, _prompt: string) => ({ question_types: ['single'] }));
    const result = await planRetrieval({ question: '论文 A 使用什么数据集？', runModel });
    expect(result.status).toBe('degraded');
    if (result.status === 'degraded') expect(result.reason).toBe('plan_invalid');
    expect(runModel).toHaveBeenCalledTimes(2);
    expect(runModel.mock.calls[1][1]).toContain('previous plan was invalid');
  });

  it('degrades when the model call throws', async () => {
    const runModel = vi.fn(async (_system: string, _prompt: string) => {
      throw new Error('model unavailable');
    });
    const result = await planRetrieval({ question: '论文 A 使用什么数据集？', runModel });
    expect(result.status).toBe('degraded');
    expect(result.plan.question_types).toEqual(['single']);
  });
});

describe('prompts', () => {
  it('keeps the frozen question and only the compact scope hint, not paper titles', () => {
    const prompt = buildPlannerPrompt({
      question: '论文 A 的方法是什么？',
      scopeHint: 'single entry'
    });
    expect(prompt).toContain('论文 A 的方法是什么？');
    expect(prompt).toContain('single entry');
    expect(PLANNER_SYSTEM_PROMPT).toContain('single');
  });
});
