from src.constants.base import ABC_START
from src.models.enums.distance import MetricDistance
from src.models.enums.notation import Notation


class Application:
    """Singleton de configuración global de KQNodes."""

    def __init__(self) -> None:
        self.semilla_numpy = 73
        self.pagina_red_muestra: str = ABC_START
        self.distancia_metrica: str = MetricDistance.HAMMING.value
        self.indexado_llegada: str = Notation.LIL_ENDIAN.value
        self.notacion_indexado: str = Notation.LIL_ENDIAN.value
        self.profiler_habilitado: bool = False

    def activar_profiling(self) -> None:
        self.profiler_habilitado = True

    def desactivar_profiling(self) -> None:
        self.profiler_habilitado = False


aplicacion = Application()
