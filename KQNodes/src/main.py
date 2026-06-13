from src.controllers.manager import Manager
from src.strategies.q_nodes import DynamicPartition


def iniciar():
    """Punto de entrada — ejecuta DynamicPartition sobre la red configurada."""

    # ABCD #
    estado_inicial = "1000000000"
    condiciones =    "1111111111"
    alcance =        "1111111111"
    mecanismo =      "1111111111"

    gestor_redes = Manager(estado_inicial)
    mpt = gestor_redes.cargar_red()

    analizador = DynamicPartition(mpt)

    solucion = analizador.aplicar_estrategia(
        estado_inicial,
        condiciones,
        alcance,
        mecanismo,
    )
    print(solucion)
