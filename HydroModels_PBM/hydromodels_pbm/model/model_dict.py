from .GR4J import gr4j
from .GR4J_CemaNeige import gr4j_cemaneige
from .GR5J import gr5j
from .GR5J_CemaNeige import gr5j_cemaneige
from .GR6J import gr6j
from .GR6J_CemaNeige import gr6j_cemaneige
from .XAJ import xaj
from .XAJ_mz import xaj_mz
from .XAJ_mz_CemaNeige import xaj_mz_cemaneige

MODEL_DICT = {
    'GR4J': gr4j,
    'GR4J-CemaNeige': gr4j_cemaneige,
    'GR5J': gr5j,
    'GR5J-CemaNeige': gr5j_cemaneige,
    'GR6J': gr6j,
    'GR6J-CemaNeige': gr6j_cemaneige,
    'XAJ': xaj,
    'XAJ-mz': xaj_mz,
    'XAJ-mz-CemaNeige': xaj_mz_cemaneige,
}
