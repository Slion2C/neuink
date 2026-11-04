import { describe, expect, it } from 'vitest';
import { resolveSearchScope, scopeRequestFromToolInput } from './searchScope';

describe('resolveSearchScope', () => {
  it('defaults to full library when nothing is frozen and no scope is requested', () => {
    expect(resolveSearchScope({ kind: 'unspecified' }, [])).toEqual({ status: 'full_library' });
  });

  it('uses the frozen scope when nothing is requested', () => {
    expect(resolveSearchScope({ kind: 'unspecified' }, ['a'])).toEqual({
      status: 'resolved',
      scope_kind: 'explicit_single',
      entry_ids: ['a']
    });
    expect(resolveSearchScope({ kind: 'unspecified' }, ['a', 'b'])).toEqual({
      status: 'resolved',
      scope_kind: 'explicit_multi',
      entry_ids: ['a', 'b']
    });
  });

  it('resolves an explicit request against no frozen scope', () => {
    expect(resolveSearchScope({ kind: 'explicit_single', entry_ids: ['a'] }, [])).toEqual({
      status: 'resolved',
      scope_kind: 'explicit_single',
      entry_ids: ['a']
    });
  });

  it('intersects an explicit request with the frozen scope and rejects out-of-scope ids', () => {
    expect(resolveSearchScope({ kind: 'explicit_multi', entry_ids: ['a', 'b'] }, ['a', 'b', 'c'])).toEqual({
      status: 'resolved',
      scope_kind: 'explicit_multi',
      entry_ids: ['a', 'b']
    });
    expect(() => resolveSearchScope({ kind: 'explicit_multi', entry_ids: ['a', 'z'] }, ['a', 'b'])).toThrow(
      '超出冻结范围'
    );
  });

  it('treats an explicitly empty request as no-results, not full library', () => {
    expect(resolveSearchScope({ kind: 'explicit_multi', entry_ids: [] }, [])).toEqual({
      status: 'empty',
      scope_kind: 'explicit_multi'
    });
    expect(resolveSearchScope({ kind: 'explicit_single', entry_ids: [] }, ['a'])).toEqual({
      status: 'empty',
      scope_kind: 'explicit_single'
    });
  });

  it('deduplicates and trims ids', () => {
    expect(resolveSearchScope({ kind: 'explicit_multi', entry_ids: [' a ', 'a', ''] }, [])).toEqual({
      status: 'resolved',
      scope_kind: 'explicit_single',
      entry_ids: ['a']
    });
  });
});

describe('scopeRequestFromToolInput', () => {
  it('maps absent input to unspecified', () => {
    expect(scopeRequestFromToolInput(undefined)).toEqual({ kind: 'unspecified' });
    expect(scopeRequestFromToolInput(null)).toEqual({ kind: 'unspecified' });
  });

  it('maps a single id to explicit_single and many ids to explicit_multi', () => {
    expect(scopeRequestFromToolInput(['a'])).toEqual({ kind: 'explicit_single', entry_ids: ['a'] });
    expect(scopeRequestFromToolInput(['a', 'b'])).toEqual({ kind: 'explicit_multi', entry_ids: ['a', 'b'] });
  });

  it('maps an empty array to an explicit empty scope', () => {
    expect(scopeRequestFromToolInput([])).toEqual({ kind: 'explicit_multi', entry_ids: [] });
  });

  it('rejects non-array input', () => {
    expect(() => scopeRequestFromToolInput('a')).toThrow('必须是条目 ID 数组');
  });
});
