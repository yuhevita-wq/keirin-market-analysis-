from __future__ import annotations

from datetime import date

from . import g1_g2_all_2021 as base2021


# KEIRIN Grand Prix 2021 series started on 2021-12-28.
# Young Grand Prix 2021 was held on day 2 (2021-12-29), so KDreams race IDs
# use 20211228 as the meeting base date and day_index=02.
_corrected = list(base2021.G1_G2_MEETINGS_2021)
_corrected[-1] = base2021.GradeMeeting(
    "静岡",
    "38",
    "shizuoka",
    "G2",
    date(2021, 12, 28),
    date(2021, 12, 29),
    12,
    "ヤング",
)
base2021.G1_G2_MEETINGS_2021 = tuple(_corrected)


def main() -> int:
    return base2021.main()


if __name__ == "__main__":
    raise SystemExit(main())
