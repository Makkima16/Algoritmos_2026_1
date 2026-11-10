from src.models.enums.notation import Notation


class Application:
    """Singleton de configuración global de KGeoMIP."""

    def __init__(self) -> None:
        self.pagina_sample_network: str = "A"
        self.semilla_numpy = 73
        self.notacion: str = Notation.LIL_ENDIAN.value
        self.distancia_metrica = "emd-effect"
        self.profiler_habilitado = False


aplicacion = Application()
