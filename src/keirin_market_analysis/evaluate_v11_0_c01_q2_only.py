from __future__ import annotations

import json

from evaluate_v8_17_q2_oos import load_q2
from evaluate_v9_0_q1q2q3 import load_entries
from evaluate_v11_0_c01_q1_only import evaluate_v11_context


def main():
    result = evaluate_v11_context(
        '2024Q2_V11_0_C01_SIMULATION',
        load_q2(),
        load_entries('2024_q2'),
    )
    out = {
        'scheme': 'v11.0-C01',
        'base_scheme': 'v8.25-F26',
        'dataset': '2024Q2',
        'mode': 'SIMULATION_REQUESTED_BY_USER',
        'context_definition_changed_from_q1': False,
        'bet_logic_changed_vs_v8_25': False,
        **result,
    }
    print('V11_0_C01_Q2_ONLY_BEGIN')
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print('V11_0_C01_Q2_ONLY_END')


if __name__ == '__main__':
    main()
