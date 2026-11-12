from __future__ import annotations

import json

from simulate_v8_1_f02_2024q1 import load
from evaluate_v9_0_q1q2q3 import load_entries
from evaluate_v9_2_q1q2q3 import evaluate_v9_2, strip_rows


def main():
    q1 = evaluate_v9_2(
        '2024Q1_V9_2_F28_SIMULATION',
        load(),
        load_entries('2024_q1'),
    )
    result = {
        'scheme': 'v9.2-F28',
        'dataset': '2024Q1',
        'mode': 'SIMULATION_REQUESTED_BY_USER',
        'scheme_changed_for_run': False,
        'result': strip_rows(q1),
    }
    print('V9_2_F28_Q1_ONLY_BEGIN')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('V9_2_F28_Q1_ONLY_END')


if __name__ == '__main__':
    main()
