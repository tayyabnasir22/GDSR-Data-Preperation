from enum import Enum

class BenchmarkType(Enum):
    NYUV2 = 'NYUV2'
    RGBDD = 'RGBDD'
    TOFDSRD = "TOFDSRD"
    LU = 'LU'
    MIDDLE = "MIDDLE"
    HYPERSIM = "HYPERSIM"
    SUNRGBD = "SUNRGBD"
    DIML = "DIML"
    DIODE = "DIODE"
    SINTEL = "SINTEL"