from enum import Enum

class BenchmarkType(Enum):
    NYUV2 = 'NYUV2'
    RGBDD = 'RGBDD'
    TOFDSRD = "TOFDSRD"
    LU = 'LU'
    MIDDLE = "MIDDLE"