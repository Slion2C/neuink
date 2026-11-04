import { z } from 'zod';

/**
 * Retrieval planning for the unified `search_library` tool.
 *
 * This module implements the planning-output contract from
 * docs/development/retrieval-tool-completion-plan.md, section 4. The planner
 * only decides *how* to search (classify + decompose the frozen question); the
 * host resolves scope separately and remains authoritative over it. Nothing
 * here changes the production retrieval path, which stays Legacy / alpha = 0.
 */

export const QUESTION_TYPES = [
  'single',
  'multi_target',
  'cross_paper',
  'dependent_multihop',
  'global'
] as const;
export type QuestionType = (typeof QUESTION_TYPES)[number];

export const SCOPE_KINDS = [
  'explicit_single',
  'explicit_multi',
  'explicit_tag',
  'unspecified'
] as const;
export type ScopeKind = (typeof SCOPE_KINDS)[number];

export const DEGRADATION_REASONS = [
  'plan_invalid',
  'model_unavailable',
  'scope_irresolvable',
  'no_model_configured'
] as const;
export type DegradationReason = (typeof DEGRADATION_REASONS)[number];

/** Provisional cap until the decomposition limit is measured (plan section 5.2). */
export const MAX_RETRIEVAL_TASKS = 12;

const retrievalTaskSchema = z.object({
  id: z.string().trim().min(1),
  question: z.string().trim().min(1),
  // Entities, times, and other restrictions the original question implies and
  // every retrieval against this task must preserve.
  constraints: z.array(z.string()).default([]),
  // First version does not execute dependency chains; recorded for downgrade.
  depends_on: z.array(z.string()).default([])
});

export type RetrievalTask = z.infer<typeof retrievalTaskSchema>;

export const planningOutputSchema = z.object({
  question_types: z.array(z.enum(QUESTION_TYPES)).min(1),
  // The model's best-effort guess. The merge stage overwrites this with the
  // host-resolved scope; a model must never be able to widen the frozen scope.
  scope_kind: z.enum(SCOPE_KINDS).default('unspecified'),
  original_question: z.string().trim().min(1),
  retrieval_tasks: z.array(retrievalTaskSchema).max(MAX_RETRIEVAL_TASKS).default([]),
  synthesis_task: z.string().trim().optional()
});

export type PlanningOutput = z.infer<typeof planningOutputSchema>;

export type PlanningValidation =
  | { ok: true; plan: PlanningOutput }
  | { ok: false; issues: string[] };

export type PlanResolution =
  | { status: 'planned'; plan: PlanningOutput; attempts: number }
  | { status: 'degraded'; plan: PlanningOutput; reason: DegradationReason; attempts: number };

export type DegradedResolution = Extract<PlanResolution, { status: 'degraded' }>;

const DECOMPOSITION_TYPES: ReadonlySet<QuestionType> = new Set([
  'multi_target',
  'cross_paper',
  'global'
]);

/** Schema plus business rules that a generic JSON schema cannot express. */
export function validatePlanningOutput(value: unknown): PlanningValidation {
  const parsed = planningOutputSchema.safeParse(value);
  if (!parsed.success) {
    return {
      ok: false,
      issues: parsed.error.issues.map((issue) => `${issue.path.join('.') || 'plan'}: ${issue.message}`)
    };
  }
  const plan = parsed.data;
  const issues: string[] = [];

  const ids = new Set<string>();
  for (const task of plan.retrieval_tasks) {
    if (ids.has(task.id)) issues.push(`duplicate retrieval task id: ${task.id}`);
    ids.add(task.id);
  }
  for (const task of plan.retrieval_tasks) {
    for (const dependency of task.depends_on) {
      if (dependency === task.id) issues.push(`task ${task.id} depends on itself`);
      else if (!ids.has(dependency)) issues.push(`task ${task.id} depends on unknown task ${dependency}`);
    }
  }

  const types = new Set(plan.question_types);
  const decomposes = [...types].some((type) => DECOMPOSITION_TYPES.has(type));
  if (types.size === 1 && types.has('single') && plan.retrieval_tasks.length > 0) {
    issues.push('single-question plans must reuse the original question without decomposition');
  }
  if (decomposes && !types.has('single') && plan.retrieval_tasks.length === 0) {
    issues.push('decomposition plans require at least one retrieval task');
  }
  if (plan.original_question !== plan.original_question.trim()) {
    issues.push('original_question must be trimmed');
  }

  return issues.length > 0 ? { ok: false, issues } : { ok: true, plan };
}

/** Direct-retrieval plan that preserves the frozen question (plan section 4). */
export function fallbackDirectPlan(question: string, reason: DegradationReason): DegradedResolution {
  return {
    status: 'degraded',
    plan: {
      question_types: ['single'],
      scope_kind: 'unspecified',
      original_question: question.trim(),
      retrieval_tasks: [],
      synthesis_task: undefined
    },
    reason,
    attempts: 0
  };
}

export const PLANNER_TASK_NAME = 'retrieval_plan';

export const PLANNER_TASK_DESCRIPTION =
  'A validated retrieval plan: question types, scope kind, original question, retrieval tasks, and optional synthesis.';

export const PLANNER_SYSTEM_PROMPT = [
  'You plan library retrieval for a research assistant that searches parsed academic papers.',
  'Classify the frozen question and, only when needed, decompose it into independently retrievable sub-questions.',
  'Question types:',
  '- single: one direct question; do not decompose.',
  '- multi_target: several independent information needs in one question; one task per need.',
  '- cross_paper: comparison or synthesis across papers; keep one task per paper-and-dimension and bind evidence to its paper.',
  '- dependent_multihop: the next query needs an earlier answer; mark depends_on but do not pretend the chain is executed.',
  '- global: exhaustive aggregation over a whole set; only claim the coverage you can actually retrieve.',
  'Preserve entities, times, and other restrictions from the original question in each task.',
  'Return JSON only: question_types, scope_kind, original_question, retrieval_tasks (id, question, constraints, depends_on), and optional synthesis_task.'
].join('\n');

export function buildPlannerPrompt(input: { question: string; scopeHint?: string }) {
  const scopeLine = input.scopeHint
    ? `Resolved scope hint (authoritative): ${input.scopeHint}`
    : 'No explicit scope was provided; scope is resolved by the host separately.';
  return [
    scopeLine,
    `Frozen user question:\n${input.question}`,
    'Produce a retrieval plan for this question. Do not widen the scope hint.'
  ].join('\n\n');
}

/**
 * Run the planner with bounded retries and a deterministic fallback. The model
 * is injected so this stays pure and testable; the caller wires `runJsonModelTask`.
 */
export async function planRetrieval({
  question,
  scopeHint,
  runModel,
  maxAttempts = 2
}: {
  question: string;
  scopeHint?: string;
  runModel: (system: string, prompt: string) => Promise<unknown>;
  maxAttempts?: number;
}): Promise<PlanResolution> {
  const prompt = buildPlannerPrompt({ question, scopeHint });
  let feedback = '';
  let lastIssue = 'unknown validation error';
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    try {
      const output = await runModel(
        PLANNER_SYSTEM_PROMPT,
        feedback ? `${prompt}\n\nThe previous plan was invalid: ${feedback}\nReturn corrected JSON only.` : prompt
      );
      const result = validatePlanningOutput(output);
      if (result.ok) return { status: 'planned', plan: result.plan, attempts: attempt + 1 };
      lastIssue = result.issues.join('; ');
      feedback = lastIssue.slice(0, 1_000);
    } catch (error) {
      lastIssue = error instanceof Error ? error.message : String(error);
      feedback = lastIssue.slice(0, 1_000);
    }
  }
  return { status: 'degraded', plan: fallbackDirectPlan(question, 'plan_invalid').plan, reason: 'plan_invalid', attempts: maxAttempts };
}
