
# Tabla de coloraciones válidas/invalidas para ciclo de 4 vértices y 2 colores
def es_valida(bits):
    aristas = [(0,1),(1,2),(2,3),(3,0)]
    for (i,j) in aristas:
        if bits[i]==bits[j]:
            return False
    return True

print("\nColoraciones válidas (para ciclo de 4 vértices, 2 colores):")
for i in range(16):
    bits = format(i, '04b')
    if es_valida(bits):
        print(bits)
print("\nColoraciones inválidas:")
for i in range(16):
    bits = format(i, '04b')
    if not es_valida(bits):
        print(bits)
