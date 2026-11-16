"""Offline reproduction from saved observations; no provider calls or API key."""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json

import numpy as np

from auditstudy.common import digest
from diagnostics.item_confidence_evaluation_analysis import analyze_evaluation

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / 'results' / name).read_text())


def save(name, value):
    out = ROOT / 'reproduced'
    out.mkdir(exist_ok=True)
    (out / name).write_text(json.dumps(value, indent=2) + '\n')


def verify_release():
    for name, expected in json.loads((ROOT / 'release_manifest.json').read_text()).items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Release file changed: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refit', action='store_true', help='Also rerun all 2,000 post-hoc refit bootstrap replicates (CPU only).')
    parser.add_argument('--groups', action='store_true', help='Also regenerate descriptive subject and calibration-group results.')
    args = parser.parse_args()
    verify_release()
    rows, pilot = read('evaluation_records.json'), read('pilot_records.json')
    models, expected = read('frozen_models.json'), read('evaluation_results.json')
    protocol = read('evaluation_protocol.json')
    if {r['id'] for r in rows} & {r['id'] for r in pilot}:
        raise ValueError('Evaluation overlaps calibration/development')
    if set(protocol['evaluation_ids']) != {r['id'] for r in rows}:
        raise ValueError('Evaluation membership differs from protocol')
    if digest(models) != protocol['frozen_models_hash'] or protocol['fingerprint'] != expected['protocol_fingerprint']:
        raise ValueError('Frozen-model or protocol provenance mismatch')
    for file, expected_hash in protocol['code_hashes'].items():
        if hashlib.sha256((ROOT / file).read_bytes()).hexdigest() != expected_hash:
            raise ValueError('Original source hash mismatch: ' + file)
    result = analyze_evaluation(rows, models)
    for name, score in result['scores'].items():
        for metric in ('brier', 'log_loss'):
            np.testing.assert_allclose(score[metric], expected['scores'][name][metric], rtol=0, atol=1e-12)
        print(f"{name:12s} Brier {score['brier']:.8f}")
    for name, contrast in result['contrasts'].items():
        for metric in ('brier', 'log_loss'):
            for field in ('mean_difference', 'ci95'):
                np.testing.assert_allclose(contrast[metric][field], expected['contrasts'][name][metric][field], rtol=0, atol=1e-12)
    save('evaluation_results.json', result)
    print('Verified scores, contrasts, bootstrap intervals and source provenance. No API calls.')
    if args.groups:
        from diagnostics.item_confidence_group_checks import analyze_groups
        save('group_results.json', analyze_groups(rows, result['predictions']))
        print('Saved descriptive group analysis; no new fits or API calls.')
    if args.refit:
        from diagnostics.item_confidence_refit_sensitivity import (
            REPETITIONS, WORKERS, MODELS, fit, validate_inputs, worker_initialize, worker, summarize,
        )
        train = sorted([r for r in pilot if r['split'] == 'calibration'], key=lambda r: r['id'])
        evaluation = sorted(rows, key=lambda r: r['id'])
        validate_inputs(train, evaluation)
        for name, signals in MODELS.items():
            fitted = fit(train, signals)
            np.testing.assert_allclose(fitted['coefficients'], models[name]['coefficients'], rtol=0, atol=1e-10)
        print(f'Running {REPETITIONS} paired calibration/evaluation refits...', flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS, initializer=worker_initialize, initargs=(train,evaluation)) as pool:
            replicates = list(pool.map(worker, range(REPETITIONS), chunksize=10))
        sensitivity = summarize(replicates, expected)
        saved = read('refit_results.json')
        if sensitivity['failed_replicates'] != saved['failed_replicates']:
            raise ValueError('Refit failure count differs from original')
        for name, contrast in sensitivity['contrasts'].items():
            for metric in ('brier', 'log_loss'):
                np.testing.assert_allclose(contrast[metric]['refit_bootstrap_ci95'], saved['contrasts'][name][metric]['refit_bootstrap_ci95'], rtol=0, atol=1e-8)
        save('refit_results.json', sensitivity)
        print('Verified all post-hoc refit intervals against saved results. No API calls.')


if __name__ == '__main__':
    main()
