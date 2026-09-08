"""Macquart Table 1 observable fixture, pipeline smoke only; host uses ln.

Primary source https://arxiv.org/html/2005.13161 (Table 1), checked 2026-09-08.
Fluences are reported E_nu in Jy ms. No SNR is supplied by this table; the
existing encoder does not use SNR or DM errors. We build its seven observable
features directly rather than inventing an observed SNR. 190611 is tentative.
"""
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

TABLE = (
    ("180924",361.42,.3214,16.,"21:44:25.255","-40:54:00.10"),
    ("181112",589.27,.4755,26.,"21:49:23.63","-52:58:15.4"),
    ("190102",363.6,.291,14.,"21:29:39.76","-79:28:32.5"),
    ("190608",338.7,.1178,26.,"22:16:04.75","-07:53:53.6"),
    ("190611",321.4,.378,10.,"21:22:58.91","-79:23:51.3"),
    ("190711",593.1,.522,34.,"21:57:40.68","-80:21:28.8"),
)


def smoke_features():
    sky = SkyCoord([r[4] for r in TABLE],[r[5] for r in TABLE],unit=(u.hourangle,u.deg)).galactic.cartesian
    return np.column_stack((np.log10([r[1] for r in TABLE]),sky.x.value,sky.y.value,sky.z.value,
                            np.log10(np.array([r[3] for r in TABLE])+1.),[r[2] for r in TABLE],
                            np.zeros(len(TABLE)))).astype(np.float32)
