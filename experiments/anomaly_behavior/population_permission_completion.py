"""Complete the two missing N50 permission arms using the frozen report protocol."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config
from run_experiments import (
    parse_args, run_case, select_cases, temporary_config,
    write_experiment_metadata, write_summary,
)
from tools.analyze_experiment_pair import compare_frames, wilson_interval
from low_level_controller import rule_mode_for_alarms

OLD = ROOT / 'logs/experiments/paired_100_20261008/3_3_population/20261008_193408_239008'
OUT = ROOT / 'logs/experiments/population_permissions_50_6101_6150_20261013'
SEEDS = list(range(6101, 6151))
VARIANTS = {'mode_only': (True, False), 'target_only': (False, True)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compatible_sources(protocol):
    audit = {}
    for name in ['dynamics.py', 'environment.py', 'detector.py',
                 'low_level_controller.py', 'mode_forecast.py']:
        assert digest(ROOT / name) == protocol['source_sha256'][name], name
        audit[name] = 'byte-identical'
    # Undo only the new provider dispatch and equivalent method extractions in
    # memory, then require the exact original hashes. The checkout is untouched.
    text = (ROOT / 'high_level_controller.py').read_text()
    text = text.replace('("openai", "deepseek", "vllm")', '("openai", "deepseek")')
    text = text.replace('        from vllm_backend import VLLMBackend\n', '')
    text = text.replace(
        '        backend_class = {"openai": OpenAIBackend, "deepseek": DeepSeekBackend, "vllm": VLLMBackend}[kind]',
        '        backend_class = OpenAIBackend if kind == "openai" else DeepSeekBackend')
    assert hashlib.sha256(text.replace('\n', '\r\n').encode()).hexdigest() == protocol['source_sha256']['high_level_controller.py']
    audit['high_level_controller.py'] = 'only optional vllm dispatch added; original hash reconstructed'
    text = (ROOT / 'responses_api_backend.py').read_text()
    text = text.replace('response = self._create_response(request_options)',
                        'response = self.client.responses.create(**request_options)')
    text = text.replace('"backend_error": self._format_error(exc),',
                        '"backend_error": f"{type(exc).__name__}: {exc}"[:500],')
    text = text.replace('    def _create_response(self, request_options):\n        return self.client.responses.create(**request_options)\n\n    def _format_error(self, exc):\n        return f"{type(exc).__name__}: {exc}"[:500]\n\n', '')
    assert hashlib.sha256(text.replace('\n', '\r\n').encode()).hexdigest() == protocol['source_sha256']['responses_api_backend.py']
    audit['responses_api_backend.py'] = 'equivalent method extractions only; original hash reconstructed'
    audit['run_experiments.py'] = 'optional vllm CLI/config/dispatch additions; DeepSeek frame loop unchanged'
    return audit


def initial_state(path):
    df = pd.read_parquet(path, columns=['Frame', 'Agent_Type', 'Agent_ID', 'X', 'Y', 'Status'])
    return df.loc[df.Frame.eq(df.Frame.min())].sort_values(['Agent_Type', 'Agent_ID']).reset_index(drop=True)


def old_subset(label, arm):
    df = pd.read_csv(OLD / label / arm / 'summary.csv')
    df = df[df.Seed.isin(SEEDS)].sort_values('Seed').reset_index(drop=True)
    assert df.Seed.tolist() == SEEDS
    return df


def summarize(df):
    return {'runs': len(df), 'successes': int(df.Success.sum()),
            'success_rate': float(df.Success.mean()),
            'success_95_interval': wilson_interval(int(df.Success.sum()), len(df)),
            'mean_frames': float(df.Frames.mean()),
            'final_inside_mean': float(df.FinalInsideGoal.mean()),
            'requests': int(df.HighLevelRequests.sum()),
            'responses': int(df.HighLevelResponses.sum()),
            'override_frames': int(df.HighLevelOverrideFrames.sum())}


def audit_api(df, arm):
    records = []
    for row in df.itertuples():
        control = pd.read_parquet(row.Control)
        if arm == 'mode_only':
            assert not control.llm_target_override.fillna(False).any()
            assert not control.llm_selected_target_sheep_id.notna().any()
        if arm == 'target_only':
            active = control.loc[control.decision_source.ne('rule')]
            for event in active.itertuples():
                alarms = () if event.alarms == 'none' else tuple(event.alarms.split(';'))
                assert event.current_mode == rule_mode_for_alarms(alarms, True)
        returned = control.loc[control.llm_request_id.notna() & control.llm_latency_seconds.notna()]
        returned = returned.drop_duplicates('llm_request_id')
        records.append(returned)
    returned = pd.concat(records, ignore_index=True)
    assert int(df.HighLevelCacheHits.sum()) == 0
    return {'runs': len(df), 'requests_recorded': int(df.HighLevelRequests.sum()),
            'responses_recorded': int(df.HighLevelResponses.sum()),
            'return_events_audited': len(returned),
            'backend_errors': returned.llm_backend_error.dropna().value_counts().to_dict(),
            'models': returned.llm_model.dropna().value_counts().to_dict(),
            'prompts': returned.llm_prompt_version.dropna().value_counts().to_dict(),
            'mean_response_seconds': float(returned.llm_latency_seconds.mean()),
            'cache_hits': 0, 'execution_permissions_verified': True}


def finish_statistics():
    report = {'protocol': json.loads((OUT / 'protocol.json').read_text()), 'population': {}}
    for label in ['ratio_30', 'ratio_60']:
        frames = {'rule': old_subset(label, 'rule'), 'mode_target': old_subset(label, 'deepseek')}
        for arm in VARIANTS:
            df = pd.read_csv(OUT / label / arm / 'summary.csv').sort_values('Seed').reset_index(drop=True)
            assert df.Seed.tolist() == SEEDS
            frames[arm] = df
        group = {'arms': {}, 'comparisons': {}}
        for arm in ['rule', 'mode_only', 'target_only', 'mode_target']:
            df = frames[arm]
            group['arms'][arm] = summarize(df)
            if arm != 'rule':
                group['arms'][arm]['api_audit'] = audit_api(df, arm)
            df.to_csv(OUT / label / f'{arm}_50.csv', index=False)
            if arm != 'rule':
                pairs, stats = compare_frames(frames['rule'], df)
                pairs.to_csv(OUT / label / f'{arm}_pairs.csv', index=False)
                group['comparisons'][arm] = stats
        report['population'][label] = group
    (OUT / 'report_statistics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--analyze-only', action='store_true')
    args = parser.parse_args()
    if args.analyze_only:
        print(json.dumps(finish_statistics()['population'], indent=2))
        return
    protocol = json.loads((OLD / 'population_api_protocol.json').read_text())
    audit = compatible_sources(protocol)
    frozen = protocol['base_config']
    overrides = {k: v for k, v in frozen.items() if hasattr(config, k)}
    os.environ['DEEPSEEK_MODEL'] = frozen['DEEPSEEK_MODEL']
    os.environ['DEEPSEEK_BASE_URL'] = frozen['DEEPSEEK_BASE_URL']
    OUT.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: digest(ROOT / name) for name in protocol['source_sha256']}
    new_protocol = {'seeds': SEEDS, 'num_sheep': 50, 'max_steps': 2000,
                    'levels': {'ratio_30': [5, 5, 5, 35], 'ratio_60': [10, 10, 10, 20]},
                    'new_arms': list(VARIANTS), 'new_runs_planned': 200,
                    'reused_arms': ['rule', 'mode_target'], 'base_config': frozen,
                    'source_sha256': source_hashes, 'compatibility_audit': audit,
                    'reuse_source': str(OLD), 'llm_cache_enabled': False,
                    'note': 'One fresh API realization per seed and missing arm; providers run on different dates.'}
    if (OUT / 'protocol.json').exists():
        assert json.loads((OUT / 'protocol.json').read_text()) == new_protocol
    else:
        (OUT / 'protocol.json').write_text(json.dumps(new_protocol, indent=2), encoding='utf-8')
    print('Source compatibility verified; frozen original config; 200 new runs.', flush=True)
    if args.audit_only:
        return
    api_audit = json.loads((OUT / 'api_audit.json').read_text()) if (OUT / 'api_audit.json').exists() else {}
    with temporary_config(**overrides):
        # A free deterministic replay checks the production runner before paid calls.
        probe_arm = OUT / 'rule_replay'
        with temporary_config(DATA_LOG_DIR=str(probe_arm / 'data'), CONTROL_LOG_DIR=str(probe_arm / 'control')):
            case = select_cases(parse_args(['--scenario', 'custom', '--abnormal-a', '5',
                                           '--abnormal-b', '5', '--abnormal-c', '5', '--repair']))[0]
            replay = run_case(case, 6101, 2000, 100000, controller_kind='rule', llm_cache_enabled=False)
            ref = old_subset('ratio_30', 'rule').iloc[0]
            for field in ['Success', 'Frames', 'FinalInsideGoal', 'MaxDisp', 'DangerFrames', 'GoalExits']:
                assert replay[field] == ref[field], (field, replay[field], ref[field])
            pd.testing.assert_frame_equal(initial_state(replay['Data']), initial_state(ref.Data))
        print('Rule replay matches original seed6101 exactly.', flush=True)
        for label, count in [('ratio_30', 5), ('ratio_60', 10)]:
            reference = old_subset(label, 'rule').set_index('Seed')
            rows = {}
            for name in VARIANTS:
                path = OUT / label / name / 'summary.csv'
                rows[name] = pd.read_csv(path).to_dict('records') if path.exists() else []
            for index, seed in enumerate(SEEDS):
                names = tuple(VARIANTS) if index % 2 == 0 else tuple(reversed(VARIANTS))
                for name in names:
                    if seed in {row['Seed'] for row in rows[name]}:
                        continue
                    assert all(digest(ROOT / f) == h for f, h in source_hashes.items()), 'Source changed during run'
                    mode, target = VARIANTS[name]
                    arm = OUT / label / name
                    with temporary_config(HIGH_LEVEL_MODE_SELECTION_ENABLED=mode,
                                          HIGH_LEVEL_TARGET_SELECTION_ENABLED=target,
                                          DATA_LOG_DIR=str(arm / 'data'), CONTROL_LOG_DIR=str(arm / 'control')):
                        options = ['--scenario', 'custom', '--abnormal-a', str(count),
                                   '--abnormal-b', str(count), '--abnormal-c', str(count),
                                   '--repair', '--controller', 'deepseek', '--no-llm-cache',
                                   '--max-steps', '2000', '--runs', '50', '--seed-mode', 'sequence',
                                   '--batch-seed', '6101', '--llm-mode-only' if mode else '--llm-target-only']
                        batch_args = parse_args(options)
                        case = select_cases(batch_args)[0]
                        arm.mkdir(parents=True, exist_ok=True)
                        if not rows[name]:
                            write_experiment_metadata(arm / 'summary.csv', batch_args, SEEDS, [case])
                        row = run_case(case, seed, 2000, 100000, controller_kind='deepseek', llm_cache_enabled=False)
                    pd.testing.assert_frame_equal(initial_state(row['Data']), initial_state(reference.loc[seed, 'Data']))
                    control = pd.read_parquet(row['Control'])
                    events = control.loc[control.llm_request_id.notna()].drop_duplicates('llm_request_id', keep='last')
                    errors = events.llm_backend_error.dropna().astype(str)
                    assert row['HighLevelCacheHits'] == 0
                    fatal = errors.str.contains(r'Error code: (?:401|402|403|404)|insufficient.*(?:balance|quota)', case=False, regex=True).any()
                    if fatal:
                        (OUT / 'fatal_failure.json').write_text(json.dumps({'arm': name, 'ratio': label, 'seed': seed, 'control': row['Control']}))
                        raise RuntimeError('Fatal provider error; partial raw logs saved, affected run not counted')
                    rows[name].append(row)
                    write_summary(rows[name], arm / 'summary.csv')
                    api_audit[f'{label}/{name}/{seed}'] = {
                        'requests': row['HighLevelRequests'], 'responses': row['HighLevelResponses'],
                        'cache_hits': row['HighLevelCacheHits'], 'backend_errors': errors.tolist(),
                        'models': control.llm_model.dropna().value_counts().to_dict(),
                        'prompts': control.llm_prompt_version.dropna().value_counts().to_dict(),
                        'initial_state_matches_original': True}
                    (OUT / 'api_audit.json').write_text(json.dumps(api_audit, indent=2), encoding='utf-8')
                    done = sum(len(v) for v in rows.values()) + (100 if label == 'ratio_60' else 0)
                    print(f'{done}/200 {label} {name} seed={seed}: success={row["Success"]} frames={row["Frames"]} inside={row["FinalInsideGoal"]} requests={row["HighLevelRequests"]} responses={row["HighLevelResponses"]}', flush=True)
    report = finish_statistics()
    print('COMPLETE: 200 new simulations; four arms x two ratios x 50 matched seeds.', flush=True)
    for label, group in report['population'].items():
        print(label, json.dumps(group['arms']), flush=True)


if __name__ == '__main__':
    main()
