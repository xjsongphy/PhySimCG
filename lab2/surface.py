"""B5: Surface reconstruction from FLIP particles via Marching Cubes."""
import numpy as np
import taichi as ti

_EDGE_TABLE = ti.field(int, shape=(256,))
_TRI_TABLE = ti.field(int, shape=(256, 16))

# Standard MC 256-case triangle table, encoded compactly.
# Each byte is an edge index (0-11) or 0xFF for terminator.
# 256 rows × 16 bytes = 4096 bytes, stored as hex.
_TRI_BYTES_HEX = (
    "ffffffffffffffffffffffffffffffff"  # 0
    "000803ffffffffffffffffffffffffff"
    "000109ffffffffffffffffffffffffff"
    "010803090801ffffffffffffffffffff"
    "01020affffffffffffffffffffffffff"
    "00080301020affffffffffffffffffff"
    "09020a000209ffffffffffffffffffff"
    "020803020a080a0908ffffffffffffff"
    "030b02ffffffffffffffffffffffffff"
    "000b02080b00ffffffffffffffffffff"
    "01090002030bffffffffffffffffffff"
    "010b0201090b09080bffffffffffffff"
    "030a010b0a03ffffffffffffffffffff"
    "000a0100080a080b0affffffffffffff"
    "030900030b090b0a09ffffffffffffff"
    "09080a0a080bffffffffffffffffffff"
    "040708ffffffffffffffffffffffffff"
    "040300070304ffffffffffffffffffff"
    "000109080407ffffffffffffffffffff"
    "040109040701070301ffffffffffffff"
    "01020a080407ffffffffffffffffffff"
    "03040703000401020affffffffffffff"
    "09020a090002080407ffffffffffffff"
    "020a0902090702030703070904ffffff"
    "080407030b02ffffffffffffffffffff"
    "0b04070b0204020004ffffffffffffff"
    "09000108040702030bffffffffffffff"
    "04070b09040b090b02090201ffffffff"
    "030a01030b0a070804ffffffffffffff"
    "010b0a01040b010004070b04ffffffff"
    "04070809000b090b0a0b0003ffffffff"
    "04070b040b09090b0affffffffffffff"
    "090504ffffffffffffffffffffffffff"
    "090504000803ffffffffffffffffffff"
    "000504010500ffffffffffffffffffff"
    "080504080305030105ffffffffffffff"
    "01020a090504ffffffffffffffffffff"
    "03000801020a040905ffffffffffffff"
    "05020a050402040002ffffffffffffff"
    "020a05030205030504030408ffffffff"
    "09050402030bffffffffffffffffffff"
    "000b0200080b040905ffffffffffffff"
    "00050400010502030bffffffffffffff"
    "02010502050802080b040805ffffffff"
    "0a030b0a0103090504ffffffffffffff"
    "040905000801080a01080b0affffffff"
    "05040005000b050b0a0b0003ffffffff"
    "05040805080a0a080bffffffffffffff"
    "090708050709ffffffffffffffffffff"
    "090300090503050703ffffffffffffff"
    "000708000107010507ffffffffffffff"
    "010503030507ffffffffffffffffffff"
    "0907080905070a0102ffffffffffffff"
    "0a0102090500050300050703ffffffff"
    "0800020802050805070a0205ffffffff"
    "020a05020503030507ffffffffffffff"
    "070905070809030b02ffffffffffffff"
    "09050709070209020002070bffffffff"
    "02030b000108010708010507ffffffff"
    "0b02010b0107070105ffffffffffffff"
    "0905080805070a01030a030bffffffff"
    "050700050009070b0001000a0b0a0000"
    "0b0a000b00030a0500080007050700ff"
    "0b0a05070b05ffffffffffffffffffff"
    "0a0605ffffffffffffffffffffffffff"
    "000803050a06ffffffffffffffffffff"
    "090001050a06ffffffffffffffffffff"
    "010803010908050a06ffffffffffffff"
    "010605020601ffffffffffffffffffff"
    "010605010206030008ffffffffffffff"
    "090605090006000206ffffffffffffff"
    "050908050802050206030208ffffffff"
    "02030b0a0605ffffffffffffffffffff"
    "0b00080b02000a0605ffffffffffffff"
    "00010902030b050a06ffffffffffffff"
    "050a06010902090b0209080bffffffff"
    "06030b060503050103ffffffffffffff"
    "00080b000b05000501050b06ffffffff"
    "030b06000306000605000509ffffffff"
    "06050906090b0b0908ffffffffffffff"
    "050a06040708ffffffffffffffffffff"
    "04030004070306050affffffffffffff"
    "010900050a06080407ffffffffffffff"
    "0a0605010907010703070904ffffffff"
    "060102060501040708ffffffffffffff"
    "010205050206030004030407ffffffff"
    "080407090005000605000206ffffffff"
    "070309070904030209050906020609ff"
    "030b020708040a0605ffffffffffffff"
    "050a0604070204020002070bffffffff"
    "00010904070802030b050a06ffffffff"
    "090201090b0209040b070b04050a06ff"
    "080407030b05030501050b06ffffffff"
    "05010b050b0601000b070b0400040bff"
    "0005090006050003060b0603080407ff"
    "06050906090b040709070b09ffffffff"
    "0a040906040affffffffffffffffffff"
    "040a0604090a000803ffffffffffffff"
    "0a00010a0600060400ffffffffffffff"
    "08030108010608060406010affffffff"
    "010409010204020604ffffffffffffff"
    "030008010209020409020604ffffffff"
    "000204040206ffffffffffffffffffff"
    "080302080204040206ffffffffffffff"
    "0a04090a06040b0203ffffffffffffff"
    "00080202080b04090a040a06ffffffff"
    "030b0200010600060406010affffffff"
    "06040106010a04080102010b080b01ff"
    "0906040903060901030b0603ffffffff"
    "080b010801000b06010901040604ff01"
    "030b06030600000604ffffffffffffff"
    "0604080b0608ffffffffffffffffffff"
    "070a0607080a08090affffffffffffff"
    "000703000a0700090a06070affffffff"
    "0a0607010a07010708010800ffffffff"
    "0a06070a0701010703ffffffffffffff"
    "010206010608010809080607ffffffff"
    "020609020901060709000903070309ff"
    "070800070006060002ffffffffffffff"
    "070302060702ffffffffffffffffffff"
    "02030b0a06080a0809080607ffffffff"
    "02000702070b00090706070a090a07ff"
    "010800010708010a0706070a02030bff"
    "0b02010b01070a0601060701ffffffff"
    "0809060806070901060b0603010306ff"
    "0009010b0607ffffffffffffffffffff"
    "070800070006030b000b0600ffffffff"
    "070b06ffffffffffffffffffffffffff"
    "07060bffffffffffffffffffffffffff"
    "0300080b0706ffffffffffffffffffff"
    "0001090b0706ffffffffffffffffffff"
    "0801090803010b0706ffffffffffffff"
    "0a0102060b07ffffffffffffffffffff"
    "01020a030008060b07ffffffffffffff"
    "020900020a09060b07ffffffffffffff"
    "060b07020a030a08030a0908ffffffff"
    "070203060207ffffffffffffffffffff"
    "070008070600060200ffffffffffffff"
    "020706020307000109ffffffffffffff"
    "010602010806010908080706ffffffff"
    "0a07060a0107010307ffffffffffffff"
    "0a070601070a010807010008ffffffff"
    "00030700070600060906070affffffff"
    "07060a070a08080a09ffffffffffffff"
    "0608040b0806ffffffffffffffffffff"
    "03060b030006000406ffffffffffffff"
    "08060b080406090001ffffffffffffff"
    "0904060906030903010b0306ffffffff"
    "060804060b08020a01ffffffffffffff"
    "01020a03000b00060b000406ffffffff"
    "040b0804060b000209020a09ffffffff"
    "0a09030a03020904030b0306040603ff"
    "0802030806020804060602ffffffffff"
    "000203ffffffffffffffffffffffffff"
    "0109000203080208040803060603ffff"
    "010904010402020406020304ffffffff"
    "0a03020a0103030108030804030406ff"
    "010302010403010004060403ffffffff"
    "04060804030604000306030a03000aff"
    "0a0904060a04ffffffffffffffffffff"
    "04090507060bffffffffffffffffffff"
    "0008030409050b0706ffffffffffffff"
    "05000105040007060bffffffffffffff"
    "0b0706080304030504030105ffffffff"
    "0905040a010207060bffffffffffffff"
    "060b0701020a000803040905ffffffff"
    "07060b05040a04020a040002ffffffff"
    "0304080305040302050a05020b0706ff"
    "070203070602050409ffffffffffffff"
    "090504000806000602060807ffffffff"
    "030602030706010500050400ffffffff"
    "060208060807020108040805010508ff"
    "0905040a0106010706010307ffffffff"
    "01060a010706010007080700090504ff"
    "04000a040a0500030a060a0703070aff"
    "07060a070a0805040a04080affffffff"
    "060905060b090b0809ffffffffffffff"
    "03060b000603000506000905ffffffff"
    "000b0800050b00010505060bffffffff"
    "060b03060305050301ffffffffffffff"
    "01020a09050b090b080b0506ffffffff"
    "000b0300060b00090605060901020aff"
    "0b08050b05060800050a0502000205ff"
    "060b03060305020a030a0503ffffffff"
    "050809050208050602030802ffffffff"
    "090506090600000602ffffffffffffff"
    "010508010800050608030802060208ff"
    "010506020106ffffffffffffffffffff"
    "01030601060a030806050609080906ff"
    "0a01000a0006090500050600ffffffff"
    "00030805060affffffffffffffffffff"
    "0a0506ffffffffffffffffffffffffff"
    "0b050a07050bffffffffffffffffffff"
    "0b050a0b0705080300ffffffffffffff"
    "050b07050a0b010900ffffffffffffff"
    "0a07050a0b07090801080301ffffffff"
    "0b01020b0701070501ffffffffffffff"
    "00080301020701070507020bffffffff"
    "090705090207090002020b07ffffffff"
    "07050207020b0509020302090802ffff"
    "02050a020305030705ffffffffffffff"
    "0802000805020807050a0205ffffffff"
    "090001050a03050307030a02ffffffff"
    "0908020902010807020a0205070502ff"
    "010305030705ffffffffffffffffffff"
    "000807000701010705ffffffffffffff"
    "090003090305050307ffffffffffffff"
    "090807050907ffffffffffffffffffff"
    "050804050a080a0b08ffffffffffffff"
    "050004050b00050a0b0b0300ffffffff"
    "00010908040a080a0b0a0405ffffffff"
    "0a0b040a04050b0304090401030104ff"
    "020501020805020b08040508ffffffff"
    "000403000504000205020b05020105ff"
    "000205000509020b050405080b0805ff"
    "090405020b03ffffffffffffffffffff"
    "02050a030502030405030804ffffffff"
    "050a02050200050004000203ffffffff"
    "000109030502030405030804ffffffff"
    "0109040104020204050205030504ffff"
    "080405080503030501ffffffffffffff"
    "000405010005ffffffffffffffffffff"
    "08030508050403020505020a030502ff"
    "0a05040a04080a0802080400ffffffff"
    "050409070b06ffffffffffffffffffff"
    "0008030409050b0706ffffffffffffff"
    "05040901090007060bffffffffffffff"
    "0803010801040804090401050b0706ff"
    "01020a05040907060bffffffffffffff"
    "0409050008030a0102060b07ffffffff"
    "07060b020a09020900050409ffffffff"
    "07060b03020a030a08080a05080504ff"
    "070203070602050409ffffffffffffff"
    "050409000807000703070806060802ff"
    "030206030607010900040509ffffffff"
    "050409060201060802080706010209ff"
    "01030701070a03080706070b080b07ff"
    "000801080701080b070a0106010706ff"
    "030b0703070807060a070a09070900ff"
    "07060a070a08080a09ffffffffffffff"
    "040b0704090b090a0bffffffffffffff"
    "000803040907090b07090a0bffffffff"
    "010b07010a0b01000a07040a04090aff"
    "03010a030a0801040a07040b04070aff"
    "040b07090b0409020b090102ffffffff"
    "090704090b0709010b020b01000803ff"
    "0b07040b0402020400ffffffffffffff"
    "0b07040b0402080304030204ffffffff"
    "02090a020709020307070409ffffffff"
    "090a070907040a0207080700020007ff"
    "03070a030a0207040a010a0004000aff"
    "010a02080704ffffffffffffffffffff"
    "040901040107070103ffffffffffffff"
    "040901040107000801080701ffffffff"
    "040003070403ffffffffffffffffffff"
    "040807ffffffffffffffffffffffffff"
    "090a080a0b08ffffffffffffffffffff"
    "03000903090b0b090affffffffffffff"
    "00010a000a08080a0bffffffffffffff"
    "03010a0b030affffffffffffffffffff"
    "01020b010b09090b08ffffffffffffff"
    "03000903090b010209020b09ffffffff"
    "00020b08000bffffffffffffffffffff"
    "03020bffffffffffffffffffffffffff"
    "02030802080a0a0809ffffffffffffff"
    "090a02000902ffffffffffffffffffff"
    "02030802080a000108010a08ffffffff"
    "010a02ffffffffffffffffffffffffff"
    "010308090108ffffffffffffffffffff"
    "000901ffffffffffffffffffffffffff"
    "000308ffffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffff"  # 255
)


def _init_mc_tables():
    """Initialize Marching Cubes lookup tables from compact hex encoding."""
    edge_conn = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for case in range(256):
        mask = 0
        for e, (i0, i1) in enumerate(edge_conn):
            if ((case >> i0) & 1) != ((case >> i1) & 1):
                mask |= (1 << e)
        _EDGE_TABLE[case] = mask

    data = bytes.fromhex(_TRI_BYTES_HEX)
    for case in range(256):
        for i in range(16):
            b = data[case * 16 + i]
            _TRI_TABLE[case, i] = b if b != 0xFF else -1


def build_surface_mesh(pos_field, nx, ny, nz, dx, threshold=0.5, mc_res=1):
    """Build a triangle mesh from particle positions using Marching Cubes.

    Args:
        pos_field: Taichi Vector field of particle positions (N, 3)
        nx, ny, nz: simulation grid resolution
        dx: cell size
        threshold: density threshold for isosurface (0-1)
        mc_res: MC grid refinement factor (1 = sim grid, 2 = 2x finer)

    Returns:
        (vertices, triangles) as numpy arrays, or (None, None) if empty
    """
    mc_nx = nx * mc_res
    mc_ny = ny * mc_res
    mc_nz = nz * mc_res
    mc_dx = dx / mc_res

    pos_np = pos_field.to_numpy().astype(np.float64)
    active = pos_np[pos_np[:, 0] >= 0]

    if len(active) == 0:
        return None, None

    density = np.zeros((mc_nx + 1, mc_ny + 1, mc_nz + 1), dtype=np.float64)

    sigma = mc_dx * 2.0
    sigma2 = 2.0 * sigma * sigma

    for px, py, pz in active:
        ci0 = max(0, int((px - 3 * sigma) / mc_dx))
        cj0 = max(0, int((py - 3 * sigma) / mc_dx))
        ck0 = max(0, int((pz - 3 * sigma) / mc_dx))
        ci1 = min(mc_nx, int((px + 3 * sigma) / mc_dx) + 1)
        cj1 = min(mc_ny, int((py + 3 * sigma) / mc_dx) + 1)
        ck1 = min(mc_nz, int((pz + 3 * sigma) / mc_dx) + 1)

        for ci in range(ci0, ci1 + 1):
            cx = (ci + 0.5) * mc_dx
            dx2 = (cx - px) ** 2
            for cj in range(cj0, cj1 + 1):
                cy = (cj + 0.5) * mc_dx
                dy2 = (cy - py) ** 2
                for ck in range(ck0, ck1 + 1):
                    cz = (ck + 0.5) * mc_dx
                    dist2 = dx2 + dy2 + (cz - pz) ** 2
                    if dist2 < 9.0 * sigma2:
                        density[ci, cj, ck] += np.exp(-dist2 / sigma2)

    max_d = density.max()
    if max_d > 0:
        density /= max_d

    corner_offsets = [
        (0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1),
        (0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1),
    ]
    edge_verts = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    verts = []
    tris = []
    vert_map = {}

    for i in range(mc_nx):
        for j in range(mc_ny):
            for k in range(mc_nz):
                vals = [density[i + di, j + dj, k + dk] for di, dj, dk in corner_offsets]
                case_idx = 0
                for c in range(8):
                    if vals[c] >= threshold:
                        case_idx |= (1 << c)
                if case_idx == 0 or case_idx == 255:
                    continue

                tri_list = []
                for t in range(16):
                    e = _TRI_TABLE[case_idx, t]
                    if e < 0:
                        break
                    tri_list.append(e)

                if len(tri_list) < 3:
                    continue

                for e in tri_list:
                    key = (i, j, k, e)
                    if key not in vert_map:
                        e0, e1 = edge_verts[e]
                        v0, v1 = vals[e0], vals[e1]
                        if abs(v1 - v0) < 1e-8:
                            t_param = 0.5
                        else:
                            t_param = (threshold - v0) / (v1 - v0)
                        t_param = max(0.0, min(1.0, t_param))
                        c0, c1 = corner_offsets[e0], corner_offsets[e1]
                        vert_map[key] = len(verts)
                        verts.append([
                            (i + c0[0] + t_param * (c1[0] - c0[0])) * mc_dx,
                            (j + c0[1] + t_param * (c1[1] - c0[1])) * mc_dx,
                            (k + c0[2] + t_param * (c1[2] - c0[2])) * mc_dx,
                        ])

                for t in range(0, len(tri_list) - 2, 3):
                    tris.append([
                        vert_map[(i, j, k, tri_list[t])],
                        vert_map[(i, j, k, tri_list[t + 1])],
                        vert_map[(i, j, k, tri_list[t + 2])],
                    ])

    if len(verts) == 0:
        return None, None

    return np.array(verts, dtype=np.float32), np.array(tris, dtype=np.int32)
