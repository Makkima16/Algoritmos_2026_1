from src.constants.base import ACTIVOS, INACTIVOS
from src.models.enums.notation import Notation

class Application:
    """
    La clase aplicación es un singleton utilizado para la obtención y configuración de parámetros.
    """

    def __init__(self) -> None:
        self.pagina_sample_network: str = "A"
        self.semilla_numpy = 73
        self.notacion: str = Notation.LIL_ENDIAN.value
        self.modo_estados = ACTIVOS
        self.distancia_metrica = "emd-effect"
        self.profiler_habilitado = False

    def set_notacion(self, tipo: Notation):
        self.notacion = tipo

    def set_distancia(self, tipo: str):
        self.distancia_metrica = tipo

    def set_estados_activos(self):
        self.modo_estados = ACTIVOS

    def set_estados_inactivos(self):
        self.modo_estados = INACTIVOS


aplicacion = Application()
