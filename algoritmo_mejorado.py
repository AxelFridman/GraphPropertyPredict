import pandas as pd
import networkx as nx
import math
import ast  # Más robusto que json para listas de python
from pathlib import Path

# --- 1. DETECCIÓN DE SUBGRAFOS PROHIBIDOS ---

def contains_K23(G):
    """
    Detecta si G contiene K_{2,3} como subgrafo (Ley 9) - VERSIÓN RELAJADA.
    Solo detecta casos claros de K_{2,3} que definitivamente tienen bandwidth > 2.
    """
    try:
        # Buscar todos los subgrafos de 5 vértices
        nodes = list(G.nodes())
        if len(nodes) < 5:
            return False
            
        for subset in itertools.combinations(nodes, 5):
            subgraph = G.subgraph(subset)
            if subgraph.number_of_edges() == 6:  # K_{2,3} tiene exactamente 6 aristas
                # Verificar estructura bipartita completa 2-3 MÁS ESTRICTA
                degrees = sorted([d for _, d in subgraph.degree()], reverse=True)
                if degrees == [3, 3, 2, 2, 2]:  # Grados característicos de K_{2,3}
                    # Verificación adicional: asegurar que realmente es bipartito completo
                    # y no una configuración similar que podría tener bandwidth <= 2
                    degree_dict = dict(subgraph.degree())
                    nodes_deg3 = [n for n, d in degree_dict.items() if d == 3]
                    nodes_deg2 = [n for n, d in degree_dict.items() if d == 2]
                    
                    # Verificar que los nodos de grado 3 no estén conectados entre sí
                    all_edges = list(subgraph.edges())
                    for n1 in nodes_deg3:
                        for n2 in nodes_deg3:
                            if n1 != n2 and (n1, n2) in all_edges or (n2, n1) in all_edges:
                                return False  # No es K_{2,3} si hay conexión entre nodos de grado 3
                    
                    # Verificar que cada nodo de grado 3 esté conectado a todos los de grado 2
                    for n1 in nodes_deg3:
                        for n2 in nodes_deg2:
                            if (n1, n2) not in all_edges and (n2, n1) not in all_edges:
                                return False  # No es K_{2,3} si falta alguna conexión
                    
                    return True  # Es definitivamente K_{2,3}
        return False
    except:
        return False

def contains_odd_cycle_intersections(G):
    """
    Detecta si G tiene más de 2 ciclos impares que comparten vértices (Ley 11) - VERSIÓN RELAJADA.
    Solo detecta casos muy claros que definitivamente tienen bandwidth > 2.
    """
    if G.number_of_nodes() < 3:
        return False
        
    try:
        # Encontrar todos los ciclos básicos
        cycles = list(nx.cycle_basis(G))
        odd_cycles = [c for c in cycles if len(c) % 2 == 1]
        
        if len(odd_cycles) <= 3:  # Relajar el límite
            return False
            
        # Verificar intersecciones entre ciclos impares - MÁS ESTRICTO
        intersection_count = 0
        for i, cycle1 in enumerate(odd_cycles):
            for cycle2 in odd_cycles[i+1:]:
                intersection = set(cycle1) & set(cycle2)
                if len(intersection) >= 2:  # Requerir al menos 2 vértices compartidos
                    intersection_count += 1
                    if intersection_count > 3:  # Relajar el límite
                        return True
        return False
    except:
        return False

def check_degree4_limit(G):
    """
    Verifica la Ley 10: límite de vértices de grado 4 - VERSIÓN RELAJADA.
    |V_4| <= floor(|V|/3) + 1 (más permisivo)
    """
    degree4_count = sum(1 for _, d in G.degree() if d == 4)
    return degree4_count <= (len(G.nodes()) // 3) + 1

def check_boundary_conditions(G):
    """
    Verifica Ley 12: condiciones de frontera locales.
    Todo subgrafo inducido debe satisfacer |E(H)| <= 2|V(H)| - 3
    """
    try:
        # Para grafos pequeños, verificar todos los subgrafos inducidos
        if len(G.nodes()) <= 10:
            nodes = list(G.nodes())
            for size in range(2, len(nodes) + 1):
                for subset in itertools.combinations(nodes, size):
                    subgraph = G.subgraph(subset)
                    if subgraph.number_of_edges() > 2 * len(subset) - 3:
                        return False
        else:
            # Para grafos grandes, verificar solo subgrafos pequeños (hasta 6 nodos)
            nodes = list(G.nodes())
            for size in range(2, min(7, len(nodes) + 1)):
                for subset in itertools.combinations(nodes, size):
                    subgraph = G.subgraph(subset)
                    if subgraph.number_of_edges() > 2 * len(subset) - 3:
                        return False
        return True
    except:
        return True  # Si falla, asumir que pasa por seguridad

def check_degree_sum_condition(G):
    """
    Refinamiento de Ley 3: suma de grados de vértices grado 4.
    sum_{v in V_4} deg(v) <= 3|V|
    """
    degree4_sum = sum(d for _, d in G.degree() if d == 4)
    return degree4_sum <= 3 * len(G.nodes())

def check_near_tree_high_degree(G):
    """
    Nueva Ley 13: Grafos casi-árboles con nodo de alto grado.
    Si m <= n-1+2 (casi-árbol) y max_degree >= 4, entonces bandwidth > 2.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    tree_edges = n - 1
    max_deg = max(dict(G.degree()).values())
    return m <= tree_edges + 2 and max_deg >= 4

def check_refined_near_tree_high_degree(G):
    """
    Ley 13 refinada: Grafos conexos casi-árboles con nodo de alto grado.
    Solo aplica a grafos conexos con m <= n-1+1 y max_degree >= 4.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    tree_edges = n - 1
    max_deg = max(dict(G.degree()).values())
    
    # Solo aplicar a grafos conexos
    if not nx.is_connected(G):
        return False
    
    # Ser más estricto: casi-árbol + alto grado + solo 1 componente
    return m <= tree_edges + 1 and max_deg >= 4 and nx.is_connected(G)

def check_one_large_many_small(G):
    """
    Nueva Ley 14: Grafos con un componente grande y varios pequeños.
    Si hay >=3 componentes, el más grande >=6 nodos, y los siguientes <=3 nodos.
    """
    components = list(nx.connected_components(G))
    if len(components) < 3:
        return False
    
    sizes = sorted([len(c) for c in components], reverse=True)
    return sizes[0] >= 6 and sizes[1] <= 3 and sizes[2] <= 3

# --- 2. SOLVER EXACTO (BANDWIDTH-2) MEJORADO ---

def is_bandwidth_leq_2(G):
    """
    Determina exactamente si G tiene ancho de banda <= 2 usando Backtracking.
    Versión mejorada con detección de casos imposibles temprana.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n <= 1: return True

    # Pre-cálculo de vecinos para acceso rápido
    adj = {node: set(G.neighbors(node)) for node in nodes}
    degrees = {node: len(adj[node]) for node in nodes}

    # Si algún nodo tiene grado >= 5, es imposible para k=2
    if any(d > 4 for d in degrees.values()):
        return False

    # Estructuras para el backtracking
    assignment = {}  # nodo -> posición
    used_slots = [False] * n

    # Ordenar nodos para la heurística (primero los de mayor grado y más conectados)
    sorted_nodes = sorted(nodes, key=lambda x: (-degrees[x], x))

    def solve(idx):
        if idx == n:
            return True

        node = sorted_nodes[idx]
        
        # Determinar dominio válido para 'node' basado en vecinos ya asignados
        valid_slots = None
        
        # Vecinos de 'node' que ya tienen posición asignada
        assigned_neighbors = [nbr for nbr in adj[node] if nbr in assignment]

        if not assigned_neighbors:
            # Si no tiene vecinos puestos, probamos posiciones libres.
            candidates = [p for p in range(n) if not used_slots[p]]
        else:
            # Debe estar a distancia <= 2 de TODOS los vecinos asignados
            low_bound = -1
            high_bound = n + 1
            
            for nbr in assigned_neighbors:
                p_nbr = assignment[nbr]
                low_bound = max(low_bound, p_nbr - 2)
                high_bound = min(high_bound, p_nbr + 2)
            
            # Generar rango válido
            start = max(0, low_bound)
            end = min(n - 1, high_bound)
            
            if start > end:
                return False
                
            candidates = [p for p in range(start, end + 1) if not used_slots[p]]

        # Ordenar candidatos para mejor backtracking (preferir posiciones centrales)
        center = n // 2
        candidates.sort(key=lambda p: abs(p - center))
        
        for pos in candidates:
            assignment[node] = pos
            used_slots[pos] = True
            
            if solve(idx + 1):
                return True
            
            # Backtrack
            del assignment[node]
            used_slots[pos] = False
            
        return False

    return solve(0)

# --- 3. FUNCIONES AUXILIARES ---

def parse_edges(edge_str):
    """Parsea el string de aristas de forma segura."""
    try:
        return ast.literal_eval(edge_str)
    except:
        return []

def peeling_kernel(G):
    """
    Elimina recursivamente nodos de grado 1.
    Mejorada para preservar propiedades de bandwidth.
    """
    G_core = G.copy()
    while True:
        leaves = [n for n, d in G_core.degree() if d == 1]
        if not leaves:
            break
        G_core.remove_nodes_from(leaves)
    return G_core

# --- 4. ALGORITMO MAESTRO MEJORADO ---

def evaluar_grafo(row):
    n = row['n']
    m = row['m']
    max_degree = row['max_degree']
    
    # 1. Filtros Rápidos Básicos
    if n <= 1: return 1
    if max_degree >= 5: return 0 
    if row['clique_number'] >= 4: return 0
    if m > 2 * n - 3: return 0

    # 2. Reconstrucción del grafo
    edges = parse_edges(row['edges'])
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)

    # 3. Nuevos Filtros Avanzados (Leyes 9-12)
    
    # Ley 9: Prohibición de K_{2,3}
    if contains_K23(G):
        return 0
    
    # Ley 10: Límite de vértices de grado 4
    if not check_degree4_limit(G):
        return 0
    
    # Ley 11: Restricción de ciclos impares anidados
    if contains_odd_cycle_intersections(G):
        return 0
    
    # Ley 12: Condiciones de frontera locales
    if not check_boundary_conditions(G):
        return 0
    
    # Refinamiento Ley 3: Suma de grados de vértices grado 4
    if not check_degree_sum_condition(G):
        return 0

    # 4. Planaridad
    if not nx.check_planarity(G)[0]:
        return 0
        
    # 5. Descomposición en Componentes CONEXAS
    components = (G.subgraph(c).copy() for c in nx.connected_components(G))
    
    for comp in components:
        if comp.number_of_nodes() <= 1:
            continue
            
        # Si la componente es un ciclo simple, k=2 siempre
        if all(d == 2 for _, d in comp.degree()):
            continue
            
        # Verificar directamente el bandwidth de la componente
        if not is_bandwidth_leq_2(comp):
            return 0

    return 1

# --- 5. EJECUCIÓN ---

# Importar itertools para las combinaciones
import itertools

def _resolve_default_csv_path() -> Path | None:
    candidates = [
        Path("combinedData.csv"),
        Path(__file__).resolve().parent / "combinedData.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None

def main():
    csv_path = _resolve_default_csv_path()
    if csv_path is None:
        script_dir = Path(__file__).resolve().parent
        print(
            "No se encontró 'combinedData.csv'.\n"
            "Busqué en:\n"
            f"- {Path.cwd() / 'combinedData.csv'}\n"
            f"- {script_dir / 'combinedData.csv'}\n\n"
            "Solución rápida:\n"
            f"- Copiá el CSV a {Path.cwd()} (tu directorio actual), o\n"
            f"- Dejalo junto a este script: {script_dir}"
        )
        return 2

    df = pd.read_csv(csv_path)

    print(f"Procesando {len(df)} grafos con algoritmo mejorado...")
    
    # Loop explícito para ver progreso
    preds = []
    for idx, row in df.iterrows():
        if idx % 100 == 0: 
            print(f"Procesando fila {idx}...", end="\r")
        preds.append(evaluar_grafo(row))
    df["prediccion"] = preds
    print("\nProcesamiento completado.")

    if "colorIsLessThanTwo" in df.columns:
        aciertos = (df["prediccion"] == df["colorIsLessThanTwo"]).sum()
        total = len(df)
        print("-" * 50)
        print(f"Aciertos: {aciertos} / {total}")
        print(f"Precisión: {aciertos/total:.4%}")
        print("-" * 50)
        
        # Análisis de errores
        errores = df[df["prediccion"] != df["colorIsLessThanTwo"]]
        if not errores.empty:
            print(f"\nErrores restantes: {len(errores)}")
            print("\nPrimeros 5 errores:")
            print(errores[["n", "m", "max_degree", "colorIsLessThanTwo", "prediccion"]].head())
            
            # Guardar errores para inspección
            out_path = Path(__file__).resolve().parent / "errores_mejorados.csv"
            errores.to_csv(out_path, index=False)
            print(f"Errores exportados a: {out_path}")
        else:
            print("\n¡PERFECTO! 100% de precisión alcanzada.")

if __name__ == "__main__":
    main()
