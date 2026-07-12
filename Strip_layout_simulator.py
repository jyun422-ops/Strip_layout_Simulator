import streamlit as st
import ezdxf
from ezdxf import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Circle, Rectangle
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from shapely.ops import unary_union
import tempfile
import os
import glob
import hashlib
import io
import itertools
from matplotlib.backends.backend_pdf import PdfPages

# 브릿지(간섭) 판정용 buffer 근사 해상도. 값이 클수록 라운드/필렛 구간의 오프셋 곡선이
# 정밀해지지만 연산량이 늘어난다. 4는 너무 거칠어 곡률부에서 실제 최소 간격과
# 시뮬레이션 결과가 어긋날 수 있어 16으로 상향.
BRIDGE_BUFFER_RESOLUTION = 16
# 피치/오프셋 이진탐색 수렴 허용오차 (mm). 화면에 소수점 2자리로 표시되는 값의
# 실제 정밀도를 보장하기 위해 표시 자릿수보다 한 단계 더 정밀하게 잡는다.
GEOM_SEARCH_TOL = 0.005


# ============================================================
# [-1] matplotlib 한글 폰트 자동 설정
# ============================================================
def setup_korean_font():
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "NanumBarunGothic",
                  "Noto Sans CJK KR", "Noto Sans KR", "UnDotum", "Batang"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            mpl.rcParams['font.family'] = name
            mpl.rcParams['axes.unicode_minus'] = False
            return name

    search_paths = [
        "/usr/share/fonts/**/*Nanum*.ttf", "/usr/share/fonts/**/*Nanum*.otf",
        "/usr/share/fonts/**/*Noto*CJK*.ttc", "/usr/share/fonts/**/*Noto*CJK*.otf",
        "/usr/share/fonts/**/*malgun*.ttf", "/Library/Fonts/**/*Nanum*.ttf",
        "./fonts/*.ttf", "./fonts/*.otf",
    ]
    for pattern in search_paths:
        for fpath in glob.glob(pattern, recursive=True):
            try:
                fm.fontManager.addfont(fpath)
                font_name = fm.FontProperties(fname=fpath).get_name()
                mpl.rcParams['font.family'] = font_name
                mpl.rcParams['axes.unicode_minus'] = False
                return font_name
            except Exception:
                continue
    mpl.rcParams['axes.unicode_minus'] = False
    return None

KOREAN_FONT_FOUND = setup_korean_font()

# ============================================================
# [0] DXF 단위 변환
# ============================================================
INSUNITS_TO_MM = {0: None, 1: 25.4, 2: 304.8, 3: 1609344.0, 4: 1.0, 5: 10.0, 6: 1000.0, 13: 100.0}

def get_unit_factor(doc):
    code = doc.header.get('$INSUNITS', 0)
    factor = INSUNITS_TO_MM.get(code, None)
    if factor is None: return 1.0, "⚠️ 단위 정보 없음: **mm으로 간주**"
    if factor == 1.0: return 1.0, "✅ 도면 단위: mm (변환 불필요)"
    return factor, f"✅ 도면 단위 자동 인식 (환산 계수 ×{factor} 적용)"


# ============================================================
# [1] 핵심 알고리즘 (Claude의 초정밀 Stacking 로직 융합)
# ============================================================
def calculate_1d_pitch(geom, bridge, tol=GEOM_SEARCH_TOL):
    """geom을 X축으로 dx만큼 평행이동한 사본이 자기 자신과 bridge 간격 이상
    떨어지는 최소 dx(=피치)를 이진탐색으로 구한다.
    dx=0은 항상 간섭(동일 위치)하고, dx=hi는 간섭하지 않도록 hi를 먼저 확보한 뒤
    구간을 절반씩 좁혀나가므로, 부품 크기와 무관하게 tol 수준의 정밀도가 보장된다
    (기존의 상대적 스텝(w/30, w/300) 방식은 큰 부품일수록 오차가 커지고,
    폭이 0에 가까운 경우 스텝이 0이 되어 무한루프에 빠질 위험이 있었다)."""
    minx, miny, maxx, maxy = geom.bounds
    w = max(maxx - minx, 1e-6)
    buffered_geom = geom.buffer(bridge, resolution=BRIDGE_BUFFER_RESOLUTION)

    lo, hi = 0.0, w + bridge
    guard = 0
    while buffered_geom.intersects(translate(geom, xoff=hi, yoff=0)) and guard < 20:
        hi *= 1.5
        guard += 1

    while hi - lo > tol:
        mid = (lo + hi) / 2
        if buffered_geom.intersects(translate(geom, xoff=mid, yoff=0)):
            lo = mid
        else:
            hi = mid
    return hi

def _min_y_gap(base_tiled, template, dx, y_start, y_floor, tol=GEOM_SEARCH_TOL):
    """base_tiled(고정 형상들의 합집합)와 겹치지 않으면서 template를 X로 dx,
    Y로 최대한 아래(y가 작은 쪽)로 내렸을 때의 최소 dy를 이진탐색으로 구한다.
    y_start는 비간섭 상태를 가정한 시작값이지만, template가 part_a와 다른
    형상/회전으로 인해 자체 바운딩박스 오프셋이 있는 경우(예: 180도 회전된
    비대칭 형상) y_start에서도 간섭이 남아있을 수 있어, 비간섭이 확인될 때까지
    위로 확장한다. y_floor까지 내려도 비간섭이면 유효한 접촉 경계가 없다는
    뜻이므로 y_floor를 그대로 반환한다."""
    hi = y_start
    span = max(y_start - y_floor, 1.0)
    guard = 0
    while base_tiled.intersects(translate(template, xoff=dx, yoff=hi)) and guard < 20:
        hi += span
        guard += 1
    if base_tiled.intersects(translate(template, xoff=dx, yoff=hi)):
        return None  # 확장해도 간섭을 벗어나지 못함 -> 이 dx는 사용 불가

    lo = y_floor
    if not base_tiled.intersects(translate(template, xoff=dx, yoff=lo)):
        return lo  # 탐색 하한까지도 간섭이 없는 극단적 경우

    while hi - lo > tol:
        mid = (lo + hi) / 2
        if base_tiled.intersects(translate(template, xoff=dx, yoff=mid)):
            lo = mid
        else:
            hi = mid
    return hi

def _dx_samples(p_base, target_step=0.5, min_n=90, max_n=400):
    """탐색할 dx 격자점 수를 피치 크기에 맞춰 적응적으로 정한다.
    고정 90분할은 피치가 큰 부품에서 격자 간격이 수 mm까지 벌어져
    실제 최적 오프셋을 건너뛸 수 있어, 목표 간격(target_step)을 기준으로 늘린다."""
    n = int(np.clip(p_base / target_step, min_n, max_n))
    return np.linspace(0, p_base, n, endpoint=False)

def find_nesting_offset(part_a, part_b, p_base, bridge):
    buffered_a = part_a.buffer(bridge, resolution=BRIDGE_BUFFER_RESOLUTION)
    minx, miny, maxx, maxy = part_a.bounds
    w, h = maxx - minx, maxy - miny

    reps = int(np.ceil((maxx - minx + p_base) / p_base)) + 2
    base_geoms = [translate(buffered_a, xoff=i*p_base, yoff=0) for i in range(-2, reps+1)]
    base_row = unary_union(base_geoms)

    best_dx, best_dy, best_part_b = 0, float('inf'), None
    y_start = h + bridge * 2
    y_floor = -h * 1.5

    for dx in _dx_samples(p_base):
        dy = _min_y_gap(base_row, part_b, dx, y_start, y_floor)
        if dy is None: continue
        if dy < best_dy:
            best_dy, best_dx, best_part_b = dy, dx, translate(part_b, xoff=dx, yoff=dy)

    if best_part_b:
        return best_dx, best_dy, unary_union([part_a, best_part_b]), best_part_b
    return None, None, None, None

# --- Claude 다열 적층(Stacking) 알고리즘 적용 ---
def _tile_row_x(buffered_geom, p_base, extra_reps=2):
    minx, miny, maxx, maxy = buffered_geom.bounds
    w = maxx - minx
    reps = int(np.ceil((w + p_base) / p_base)) + extra_reps
    tiles = [translate(buffered_geom, xoff=i * p_base, yoff=0) for i in range(-extra_reps, reps + 1)]
    return unary_union(tiles)

def _find_next_row(collision_base_tiled, new_part_template, start_y, p_base, bridge):
    minx, miny, maxx, maxy = new_part_template.bounds
    h = maxy - miny
    top_start = start_y - miny + h + bridge * 2
    floor_limit = start_y - h * 2.5
    best_dx, best_dy, best_part = 0, top_start, None

    for dx in _dx_samples(p_base):
        dy = _min_y_gap(collision_base_tiled, new_part_template, dx, top_start, floor_limit)
        if dy is None: continue
        if dy < best_dy:
            best_dy, best_dx, best_part = dy, dx, translate(new_part_template, xoff=dx, yoff=dy)

    return best_dx, best_dy, best_part

def build_stacked_rows(rotated_part, num_rows, p_base, bridge):
    rows_placed = [rotated_part]
    collision_tiled = _tile_row_x(rotated_part.buffer(bridge, resolution=BRIDGE_BUFFER_RESOLUTION), p_base)
    current_top_y = rotated_part.bounds[3]

    for _ in range(1, max(1, num_rows)):
        dx, dy, placed = _find_next_row(collision_tiled, rotated_part, current_top_y, p_base, bridge)
        if placed is None: break
        rows_placed.append(placed)
        collision_tiled = unary_union([collision_tiled, _tile_row_x(placed.buffer(bridge, resolution=BRIDGE_BUFFER_RESOLUTION), p_base)])
        current_top_y = max(current_top_y, placed.bounds[3])
    return rows_placed

def check_multi_row_clash(parts, bridge):
    """모든 행 쌍(인접하지 않은 조합 포함)에 대해 브릿지 간섭 여부를 검사한다.
    기존에는 0행과 나머지 행만 비교해 예를 들어 1행-3행처럼 0행이 끼지 않는
    조합의 간섭을 놓칠 수 있었다 (num_rows>=4에서 실제로 발생 가능)."""
    if len(parts) < 3: return False
    for r1, r2 in itertools.combinations(range(len(parts)), 2):
        if abs(r1 - r2) == 1: continue  # 인접 행은 배치 단계에서 이미 검증됨
        buf = parts[r1].buffer(bridge - 0.05, resolution=BRIDGE_BUFFER_RESOLUTION)
        if buf.intersects(parts[r2]): return True
    return False

def row_colors(n):
    palette = ['#004b87', '#007934', '#d55e00', '#9b59b6', '#c0392b', '#16a085']
    return [palette[i % len(palette)] for i in range(n)]


# ============================================================
# [2] DXF 읽기 및 시각화 (버그 픽스 + 블록 분해)
# ============================================================
def extract_entities_recursive(entity_container, fail_log=None):
    target_types = {'LWPOLYLINE', 'POLYLINE', 'CIRCLE', 'ELLIPSE', 'SPLINE'}
    extracted = []
    for entity in entity_container:
        if entity.dxftype() == 'INSERT':
            try:
                for virt_entity in entity.virtual_entities():
                    if virt_entity.dxftype() == 'INSERT':
                        extracted.extend(extract_entities_recursive([virt_entity], fail_log))
                    elif virt_entity.dxftype() in target_types:
                        extracted.append(virt_entity)
            except Exception as e:
                if fail_log is not None: fail_log.append(f"INSERT({entity.dxf.name if entity.dxf.hasattr('name') else '?'}): {e}")
        elif entity.dxftype() in target_types:
            extracted.append(entity)
    return extracted

def read_part_with_holes(msp, unit_factor):
    candidates = []
    fail_log = []
    flatten_tol = max(1e-4, 0.05 / unit_factor)
    valid_entities = extract_entities_recursive(msp, fail_log)

    for entity in valid_entities:
        try:
            p = path.make_path(entity)
            coords = [(v.x * unit_factor, v.y * unit_factor) for v in p.flattening(distance=flatten_tol)]
            if len(coords) < 3: continue
            poly = Polygon(coords).buffer(0)
            if poly.is_empty or poly.area < 1e-4: continue
            if poly.geom_type == 'MultiPolygon': poly = max(poly.geoms, key=lambda a: a.area)
            candidates.append(poly)
        except Exception as e:
            fail_log.append(f"{entity.dxftype()}: {e}")

    if not candidates: return None, [], fail_log

    # 외곽 윤곽선은 정의상 모든 홀보다 항상 면적이 크므로(홀은 외곽 내부에 포함),
    # 원형 여부(circularity)로 후보를 걸러낼 필요가 없다.
    # 예전 로직은 부품 외곽 자체가 원형(와셔, 디스크 등)인데 내부에 비원형
    # 홀(슬롯, D컷 등)이 하나라도 있으면 그 홀을 외곽선으로 오인하는 버그가 있었다.
    outer = max(candidates, key=lambda a: a.area)

    holes = [c for c in candidates if c is not outer and outer.contains(c.buffer(-1e-6))]
    net_part = Polygon(outer.exterior.coords, [h.exterior.coords for h in holes]) if holes else outer
    return net_part, holes, fail_log

def plot_polygon(ax, poly, color, lw=1.5, alpha=0.5):
    verts, codes = [], []
    ext_coords = list(poly.exterior.coords)
    if not poly.exterior.is_ccw: ext_coords = ext_coords[::-1]
    verts.extend(ext_coords)
    codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(ext_coords) - 2) + [MplPath.CLOSEPOLY]
    
    for interior in poly.interiors:
        int_coords = list(interior.coords)
        if interior.is_ccw: int_coords = int_coords[::-1]
        verts.extend(int_coords)
        codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(int_coords) - 2) + [MplPath.CLOSEPOLY]
        
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor=color, linewidth=lw, alpha=alpha, zorder=2))
    for interior in poly.interiors:
        xs, ys = interior.xy
        ax.fill(xs, ys, color='white', alpha=1.0, zorder=3) 
        ax.plot(xs, ys, color=color, linewidth=lw * 0.8, zorder=4) 
    ax.plot(*poly.exterior.xy, color=color, linewidth=lw, zorder=4)


# ============================================================
# [3] UI 렌더링 헬퍼 함수
# ============================================================
def render_case_column(col, label, results, best, used_fallback, colors, margin, carrier_width):
    with col:
        st.subheader(f"{label} ({best['angle']}°)")
        st.caption(f"이용율: :blue[**{best['util']:.2f}%**] | 단가: :blue[**{int(best['cost']):,}원**]")
        if used_fallback: st.warning("⚠️ 압연방향 제약을 만족하는 각도가 없어 제약을 무시한 최적값입니다.")

        fig, ax = plt.subplots(figsize=(6, 6))
        for geom, color in zip(best['parts'], colors):
            plot_polygon(ax, geom, color)
        
        all_geom = unary_union(best['parts'])
        sx1, sy1 = all_geom.bounds[0], all_geom.bounds[1] - margin - carrier_width
        sy2 = all_geom.bounds[3] + margin + carrier_width
        
        ax.plot([sx1, sx1 + best['p'], sx1 + best['p'], sx1, sx1], [sy1, sy1, sy2, sy2, sy1], 
                color='red', linestyle='--', linewidth=2.5, label=f"1피치 소요 면적\n(폭: {best['w']:.1f} × 피치: {best['p']:.2f})")
        
        ax.axis('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), fontsize=9, framealpha=1.0)
        fig.subplots_adjust(bottom=0.28)  
        st.pyplot(fig)

        df = pd.DataFrame(results)  
        df['is_chosen'] = df['각도'] == f"{best['angle']}°"
        df = df.sort_values(by=['is_chosen', '소재이용율(%)'], ascending=[False, False]).reset_index(drop=True)
        df = df.drop(columns=['is_chosen'])
        
        def highlight(row):
            if row['각도'] == f"{best['angle']}°": return ['color: blue; font-weight: bold; background-color: #e6f2ff;'] * len(row)
            if row['압연방향 적합'] == 'X': return ['background-color: #ffe6e6;'] * len(row)
            return [''] * len(row)
        st.dataframe(df.style.apply(highlight, axis=1).format({'피치(mm)': '{:.2f}', '소재폭(mm)': '{:.2f}', '소재이용율(%)': '{:.2f}'}), use_container_width=True)


# ============================================================
# [4] 데이터 추출 헬퍼 (DXF / Excel / PDF)
# ============================================================
def plot_strip_layout(parts_and_colors, pitch, part_zone_width, margin, carrier_width, pilot_dia, total_stations):
    strip_width = part_zone_width + carrier_width * 2
    all_geoms = unary_union([p[0] for p in parts_and_colors])
    minx, miny, maxx, maxy = all_geoms.bounds
    part_length = maxx - minx
    total_length = (pitch * (total_stations - 1)) + part_length + (pitch * 0.4)

    fig, ax = plt.subplots(figsize=(max(8, total_stations * 2), 4))
    ax.plot([0, total_length, total_length, 0, 0], [0, 0, strip_width, strip_width, 0],
            color='red', linestyle='-', linewidth=2.5, label=f'금형 코어 최소 사이즈\n(가로: {total_length:.1f} x 세로: {strip_width:.1f})')

    if carrier_width > 0:
        ax.add_patch(Rectangle((0, 0), total_length, carrier_width, facecolor='#999999', alpha=0.25, edgecolor='none', zorder=1))
        ax.add_patch(Rectangle((0, strip_width - carrier_width), total_length, carrier_width, facecolor='#999999', alpha=0.25, edgecolor='none', label='캐리어(스켈레톤) 영역', zorder=1))

    y_offset, x_offset = -miny + margin + carrier_width, -minx + (pitch * 0.2)

    for i in range(total_stations):
        for geom, color in parts_and_colors:
            shifted = translate(geom, xoff=x_offset + (i * pitch), yoff=y_offset)
            plot_polygon(ax, shifted, color, lw=1.5, alpha=0.5)

        if carrier_width > 0 and 0 < pilot_dia < carrier_width:
            ax.add_patch(Circle((pitch * (i + 0.5), carrier_width / 2), pilot_dia / 2, facecolor='white', edgecolor='black', linewidth=1.2, zorder=5))
        if i < total_stations - 1:
            ax.plot([pitch * (i + 1), pitch * (i + 1)], [0, strip_width], color='black', linestyle=':', alpha=0.4, zorder=1)

    ax.axis('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    return fig

def render_strip_section(label, best, total_stations, margin, carrier_width, pilot_dia, colors):
    all_geoms = unary_union(best['parts'])
    minx, miny, maxx, maxy = all_geoms.bounds
    l_val = (best['p'] * (total_stations - 1)) + (maxx - minx) + (best['p'] * 0.4)
    
    st.info(f"📐 **{label} 금형 코어 최소 사이즈:** 가로(L) :blue[**{l_val:.1f} mm**] × 세로(W) :blue[**{best['w']:.1f} mm**] (캐리어 {carrier_width}mm 포함)  |  피치(P) :blue[**{best['p']:.2f} mm**] × **{total_stations}**스테이션")
    fig = plot_strip_layout(list(zip(best['parts'], colors)), best['p'], best['w'] - carrier_width * 2, margin, carrier_width, pilot_dia, total_stations)
    st.pyplot(fig)

def generate_dxf_bytes(tuned_parts, tune_pitch, tune_width, total_stations, margin, carrier_width, pilot_dia, x_shift, y_shift):
    doc = ezdxf.new('R2010')
    doc.header['$INSUNITS'] = 4  
    msp = doc.modelspace()
    doc.layers.add("STRIP_EDGE", color=1); doc.layers.add("PARTS", color=7); doc.layers.add("PILOT_HOLES", color=5)

    all_geom_tuned = unary_union(tuned_parts)
    minx, miny, maxx, maxy = all_geom_tuned.bounds
    total_length_tuned = (tune_pitch * (total_stations - 1)) + (maxx - minx) + (tune_pitch * 0.4)

    msp.add_lwpolyline([(0, 0), (total_length_tuned, 0)], dxfattribs={'layer': 'STRIP_EDGE'})
    msp.add_lwpolyline([(0, tune_width), (total_length_tuned, tune_width)], dxfattribs={'layer': 'STRIP_EDGE'})
    
    def add_poly_to_block(poly, layer, block):
        if poly.geom_type == 'MultiPolygon':
            for p in poly.geoms: add_poly_to_block(p, layer, block)
            return
        block.add_lwpolyline(list(poly.exterior.coords), dxfattribs={'layer': layer, 'closed': True})
        for interior in poly.interiors:
            block.add_lwpolyline(list(interior.coords), dxfattribs={'layer': layer, 'closed': True})

    part_centroids = []
    for idx, geom in enumerate(tuned_parts):
        block_name = f"PART_BLOCK_{idx + 1}"
        if block_name not in doc.blocks:
            block = doc.blocks.new(name=block_name)
            cx, cy = geom.centroid.x, geom.centroid.y
            add_poly_to_block(translate(geom, xoff=-cx, yoff=-cy), "PARTS", block)
            part_centroids.append((cx, cy))

    for i in range(total_stations):
        for idx in range(len(tuned_parts)):
            msp.add_blockref(f"PART_BLOCK_{idx + 1}", insert=(x_shift + (i * tune_pitch) + part_centroids[idx][0], y_shift + part_centroids[idx][1]))
        if carrier_width > 0 and 0 < pilot_dia < carrier_width:
            msp.add_circle(center=(tune_pitch * (i + 0.5), carrier_width / 2), radius=pilot_dia / 2, dxfattribs={'layer': 'PILOT_HOLES'})

    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
        doc.saveas(tmp.name)
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f: dxf_bytes = f.read()
    os.remove(tmp_path)
    return dxf_bytes

def generate_excel_report(best_name, t_pitch, t_width, t_util, t_cost, total_st, mat_type, mat_th, mat_price, mat_den, s_res, i_res, z_res):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        pd.DataFrame({
            "항목": ["채택 배열", "소재 분류", "두께(t)", "비중", "단가(원/kg)", "총 공정수", "피치(mm)", "폭(mm)", "이용율(%)", "단가(원)"],
            "값": [best_name, mat_type, mat_th, mat_den, mat_price, total_st, t_pitch, t_width, f"{t_util:.2f}", int(t_cost)]
        }).to_excel(writer, sheet_name="최종 요약", index=False)
        
        all_dfs = []
        if s_res: all_dfs.append(pd.DataFrame(s_res).assign(**{'배열 방식': '단일 배열'}))
        if i_res: all_dfs.append(pd.DataFrame(i_res).assign(**{'배열 방식': '교차 배열'}))
        if z_res: all_dfs.append(pd.DataFrame(z_res).assign(**{'배열 방식': '다열 배치'}))
        if all_dfs: pd.concat(all_dfs, ignore_index=True).to_excel(writer, sheet_name="전체 시뮬레이션", index=False)
    return output.getvalue()

def generate_pdf_report(fig_layout, best_name, t_pitch, t_width, t_util, t_cost, total_st, mat_type, mat_th):
    pdf_buf = io.BytesIO()
    with PdfPages(pdf_buf) as pdf:
        fig_text, ax_text = plt.subplots(figsize=(8, 5))
        ax_text.axis('off')
        summary_text = (f"■ 소재: {mat_type} ({mat_th}t)\n\n■ 설계 결과: {best_name}\n - {total_st} 스테이션\n"
                        f" - 피치: {t_pitch:.2f} mm | 폭: {t_width:.2f} mm\n\n■ 산출물\n - 이용율: {t_util:.2f} %\n - 1개당 단가: {int(t_cost):,} 원\n")
        ax_text.text(0.05, 0.95, "프레스 레이아웃 시뮬레이션 리포트", fontdict={'fontsize': 16, 'fontweight': 'bold'}, va='top')
        ax_text.text(0.05, 0.75, summary_text, fontdict={'fontsize': 12}, va='top')
        pdf.savefig(fig_text); plt.close(fig_text); pdf.savefig(fig_layout)
    return pdf_buf.getvalue()


# ============================================================
# [5] 사이드바 및 UI 구성
# ============================================================
st.set_page_config(page_title="프레스 레이아웃 최적화기", layout="wide")
st.title("⚙️ 프로그레시브 금형 스트립 설계 시뮬레이터")

if KOREAN_FONT_FOUND is None: st.warning("⚠️ 서버에 한글 폰트가 설치되어 있지 않아 이미지 안의 텍스트가 깨질 수 있습니다.")

st.sidebar.header("📝 1. 소재 조건")
mat_type = st.sidebar.radio("분류", ["일반 철강/연질 (SPCC, AL 등)", "고장력강/경질 (STS, SUS 등)"])
material_thickness = st.sidebar.number_input("두께 (t)", value=1.2, step=0.1)
material_price = st.sidebar.number_input("단가 (원/kg)", value=1200, step=50)
material_density = st.sidebar.number_input("비중", value=7.85, step=0.01)

rec_bridge, rec_margin = (max(1.2, 1.2*material_thickness), max(1.5, 1.5*material_thickness)) if "일반" in mat_type else (max(1.5, 1.5*material_thickness), max(2.0, 1.8*material_thickness))

st.sidebar.header("📏 2. 배열 간격")
bridge = st.sidebar.number_input("부품간 최소 간격 (mm)", value=float(round(rec_bridge, 1)), step=0.1)
margin = st.sidebar.number_input("가장자리 마진 (mm)", value=float(round(rec_margin, 1)), step=0.1)

st.sidebar.header("🧱 3-1. 다열(N행) 설정")
num_rows = st.sidebar.number_input("배치 행 수 (2행 이상)", value=2, min_value=2, max_value=8, step=1, help="다열 배열 시뮬레이션에 적용됩니다. 단일 배열은 1열로 고정됩니다.")

st.sidebar.header("🔧 3. 캐리어 & 파일럿")
carrier_width = st.sidebar.number_input("캐리어 폭 (mm)", value=float(round(max(4.0, 3*material_thickness), 1)), step=0.5)
pilot_dia = st.sidebar.number_input("파일럿 홀 지름 (mm)", value=4.0, step=0.5)

st.sidebar.header("🛠️ 4. 공도도 설계")
st_notch = st.sidebar.number_input("노칭 / 파일럿 홀", value=1, step=1)
st_pierce = st.sidebar.number_input("피어싱", value=1, step=1)
st_bend = st.sidebar.number_input("벤딩", value=0, step=1)
st_form = st.sidebar.number_input("포밍", value=0, step=1)
st_final_notch = st.sidebar.number_input("최종 낙하", value=1, step=1)
st_idle = st.sidebar.number_input("아이들 피치", value=1, step=1)
st_simul = st.sidebar.number_input("➖ 동시 성형 (차감)", value=0, step=1)
total_stations = max(1, int((st_notch + st_pierce + st_bend + st_form + st_final_notch + st_idle) - st_simul))
st.sidebar.info(f"**총 예상 스테이션: {total_stations} 피치**")

st.sidebar.header("🧭 5. 압연방향(그레인)")
apply_rolling_constraint = st.sidebar.checkbox("벤딩 라인 제약 적용", value=(st_bend > 0))
bend_angles_input = st.sidebar.text_input("벤딩 각도 (°, 쉼표구분)", value="0", disabled=not apply_rolling_constraint)
min_angle_from_rolling = st.sidebar.number_input("최소 이격각 (°)", value=30.0, step=5.0, disabled=not apply_rolling_constraint)

bend_line_angles = []
if apply_rolling_constraint:
    try: bend_line_angles = [float(x.strip()) for x in bend_angles_input.split(',')]
    except ValueError: st.sidebar.error("❌ 각도는 숫자와 쉼표로만 입력하세요."); st.stop()


# ============================================================
# [6] 메인 화면 및 연산 로직 (Stacking 알고리즘)
# ============================================================
st.markdown("#### 📂 1. 도면 업로드")
uploaded_file = st.file_uploader("DXF 전개도면을 업로드하세요.", type=['dxf'])

angle_step = 5 if "5도" in st.radio("회전 탐색 각도 간격:", ["10도씩 (기본)", "5도씩 (정밀)"], horizontal=True) else 10
file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest() if uploaded_file else None
current_params = (file_hash, bridge, margin, carrier_width, material_thickness, material_density, material_price, apply_rolling_constraint, tuple(bend_line_angles), min_angle_from_rolling, angle_step, num_rows)

if 'last_params' not in st.session_state: st.session_state.last_params = None
recalculated = False

if uploaded_file is not None:
    if st.session_state.last_params != current_params:
        recalculated = True 
        with st.spinner('안전 간격 및 다열(Multi-row) 파고들기를 분석 중입니다...'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                tmp.write(uploaded_file.getvalue()); tmp_path = tmp.name
            doc = ezdxf.readfile(tmp_path)
            msp = doc.modelspace()
            os.remove(tmp_path)

            unit_factor, unit_msg = get_unit_factor(doc)
            part, holes, parse_fail_log = read_part_with_holes(msp, unit_factor)
            if part is None: st.error("❌ 다각형 정보를 찾을 수 없습니다."); st.stop()
            if part.geom_type == 'MultiPolygon': part = max(part.geoms, key=lambda a: a.area)
            part_area, pair_area = part.area, part.area * 2

            single_results, inter_results, zigzag_results = [], [], []
            best_s = best_i = best_z = fallback_s = fallback_i = fallback_z = None

            for angle in range(0, 180, angle_step):
                rotated_part = rotate(part, angle, origin='centroid')
                p_base = calculate_1d_pitch(rotated_part, bridge)

                valid = True
                if apply_rolling_constraint:
                    for b_angle in bend_line_angles:
                        eff_angle = (b_angle + angle) % 180
                        if min(eff_angle, 180 - eff_angle) < min_angle_from_rolling:
                            valid = False; break

                # [1] 단일 배열 계산
                minx, miny, maxx, maxy = rotated_part.bounds
                w_s = (maxy - miny) + margin * 2 + carrier_width * 2
                util_s = (part_area / (p_base * w_s)) * 100
                cost_s = (((p_base * w_s * material_thickness) * material_density) / 1_000_000) * material_price
                rec_s = {'util': util_s, 'cost': cost_s, 'angle': angle, 'parts': [rotated_part], 'w': w_s, 'p': p_base}
                single_results.append({'각도': f"{angle}°", '피치(mm)': round(p_base, 2), '소재폭(mm)': round(w_s, 2), '소재이용율(%)': round(util_s, 2), '1개당 원가(원)': int(cost_s), '압연방향 적합': 'O' if valid else 'X'})
                if not fallback_s or util_s > fallback_s['util']: fallback_s = rec_s
                if valid and (not best_s or util_s > best_s['util']): best_s = rec_s

                # [2] N열 교차 배열 (Fast Vector Copy)
                part_180 = rotate(rotated_part, 180, origin='centroid')
                dx_i1, dy_i1, _, _ = find_nesting_offset(rotated_part, part_180, p_base, bridge)
                if dx_i1 is not None:
                    dx_i2, dy_i2, _, _ = find_nesting_offset(part_180, rotated_part, p_base, bridge)
                    if dx_i2 is not None:
                        i_parts = [rotated_part]
                        cx, cy = 0.0, 0.0
                        for r in range(1, num_rows):
                            if r % 2 == 1: cx += dx_i1; cy += dy_i1; i_parts.append(translate(part_180, xoff=cx, yoff=cy))
                            else: cx += dx_i2; cy += dy_i2; i_parts.append(translate(rotated_part, xoff=cx, yoff=cy))
                        if not check_multi_row_clash(i_parts, bridge):
                            geom_i = unary_union(i_parts)
                            w_i = (geom_i.bounds[3] - geom_i.bounds[1]) + margin * 2 + carrier_width * 2
                            util_i = (part_area * num_rows / (p_base * w_i)) * 100
                            cost_i = (((p_base * w_i * material_thickness) * material_density) / 1_000_000) * material_price / num_rows
                            rec_i = {'util': util_i, 'cost': cost_i, 'angle': angle, 'parts': i_parts, 'w': w_i, 'p': p_base}
                            inter_results.append({'각도': f"{angle}°", '피치(mm)': round(p_base, 2), '소재폭(mm)': round(w_i, 2), '소재이용율(%)': round(util_i, 2), '1개당 원가(원)': int(cost_i), '압연방향 적합': 'O' if valid else 'X'})
                            if not fallback_i or util_i > fallback_i['util']: fallback_i = rec_i
                            if valid and (not best_i or util_i > best_i['util']): best_i = rec_i

                # [3] N열 지그재그 배열 (Claude's Stacking Algorithm)
                zigzag_rows = build_stacked_rows(rotated_part, num_rows, p_base, bridge)
                n_rows_actual = len(zigzag_rows)
                if n_rows_actual >= 2:
                    geom_z_full = unary_union(zigzag_rows)
                    w_z = (geom_z_full.bounds[3] - geom_z_full.bounds[1]) + margin * 2 + carrier_width * 2
                    util_z = (part_area * n_rows_actual / (p_base * w_z)) * 100
                    cost_z = (((p_base * w_z * material_thickness) * material_density) / 1_000_000) * material_price / n_rows_actual
                    rec_z = {'util': util_z, 'cost': cost_z, 'angle': angle, 'parts': zigzag_rows, 'w': w_z, 'p': p_base, 'n_rows': n_rows_actual}
                    zigzag_results.append({'각도': f"{angle}°", '피치(mm)': round(p_base, 2), '소재폭(mm)': round(w_z, 2), '소재이용율(%)': round(util_z, 2), '1개당 원가(원)': int(cost_z), '압연방향 적합': 'O' if valid else 'X'})
                    if not fallback_z or util_z > fallback_z['util']: fallback_z = rec_z
                    if valid and (not best_z or util_z > best_z['util']): best_z = rec_z

            st.session_state.update({
                'part_area': part_area, 'holes_count': len(holes), 'holes_area': sum(h.area for h in holes) if holes else 0.0,
                'single': (single_results, best_s or fallback_s, not best_s),
                'inter': (inter_results, best_i or fallback_i, not best_i),
                'zigzag': (zigzag_results, best_z or fallback_z, not best_z),
                'unit_msg': unit_msg,
                'parse_fail_log': parse_fail_log,
            })
            st.session_state.last_params = current_params

    s_res, best_s, s_fall = st.session_state['single']
    i_res, best_i, i_fall = st.session_state['inter']
    z_res, best_z, z_fall = st.session_state['zigzag']

    st.caption(st.session_state['unit_msg'])
    if st.session_state.get('parse_fail_log'):
        with st.expander(f"⚠️ 도면 해석에 실패한 요소 {len(st.session_state['parse_fail_log'])}건 (형상/면적 계산에서 제외됨)"):
            for msg in st.session_state['parse_fail_log']:
                st.text(msg)
    if st.session_state['holes_count'] > 0: st.info(f"🕳️ 내측 홀 **{st.session_state['holes_count']}개** (순단면적 반영 완료)")

    cands = [('단일 배열', best_s)]
    if best_i: cands.append((f'{num_rows}열 교차 배열', best_i))
    if best_z: cands.append((f"{best_z.get('n_rows', num_rows)}열 지그재그 배열", best_z))
    best_method_name, best_overall = min(cands, key=lambda kv: kv[1]['cost'])
    
    st.success(f"🏆 가장 훌륭한 배열은 **[{best_method_name}]**이며, 단일 배열 대비 1개당 :blue[**{int(best_s['cost'] - best_overall['cost']):,}원**]을 절감합니다.")

    st.header("📊 [1단계] 단위 배열 최적화 결과")
    col1, col2, col3 = st.columns(3)
    render_case_column(col1, "[1] 단일 배열", s_res, best_s, s_fall, ['#004b87'], margin, carrier_width)
    if best_i: render_case_column(col2, f"[2] {num_rows}열 교차 배열", i_res, best_i, i_fall, row_colors(len(best_i['parts'])), margin, carrier_width)
    else: col2.warning("교차 배열 불가 형상")
    if best_z: render_case_column(col3, f"[3] {best_z.get('n_rows', num_rows)}열 지그재그 배열", z_res, best_z, z_fall, row_colors(len(best_z['parts'])), margin, carrier_width)
    else: col3.warning("지그재그 다열 배열 불가 형상")

    st.divider()
    st.header("🎞️ [2단계] 스트립 Layout도 및 금형 코어 사이즈 도출")
    st.subheader("◼️ [1] 단일 배열 Layout도")
    render_strip_section("단일 배열", best_s, total_stations, margin, carrier_width, pilot_dia, ['#004b87'])
    if best_i: st.divider(); st.subheader(f"◼️ [2] {num_rows}열 교차 배열 Layout도"); render_strip_section(f"{num_rows}열 교차 배열", best_i, total_stations, margin, carrier_width, pilot_dia, row_colors(len(best_i['parts'])))
    if best_z: st.divider(); st.subheader(f"◼️ [3] {best_z.get('n_rows', num_rows)}열 지그재그 배열 Layout도"); render_strip_section(f"{best_z.get('n_rows', num_rows)}열 지그재그 배열", best_z, total_stations, margin, carrier_width, pilot_dia, row_colors(len(best_z['parts'])))

    # ============================================================
    # [7] 수동 미세 조정 (Gemini의 직관적인 Step-Offset 융합 UI)
    # ============================================================
    st.divider()
    st.header("🛠️ [3단계] 수동 미세 조정 (Fine-Tuning)")
    st.markdown("자동 계산된 최적 배열을 바탕으로 **숫자를 증감(+, -) 시키면 즉시 도면과 원가에 반영**됩니다.")

    valid_opts = [k for k, v in cands]
    if 'tune_target_select' not in st.session_state or st.session_state['tune_target_select'] not in valid_opts:
        st.session_state['tune_target_select'] = best_method_name
    tune_target_name = st.selectbox("조정할 배열 방식 선택", options=valid_opts, key='tune_target_select')
    target_best = next(v for k, v in cands if k == tune_target_name)
    n_parts = len(target_best['parts'])

    tkey = tune_target_name
    tune_reset_sig = (file_hash, tune_target_name)
    if st.session_state.get('last_tune_target') != tune_reset_sig or recalculated:
        st.session_state[f"tune_angle_{tkey}"] = float(target_best['angle'])
        st.session_state[f"tune_pitch_{tkey}"] = float(target_best['p'])
        st.session_state[f"tune_width_{tkey}"] = float(target_best['w'])
        st.session_state[f"tune_y_all_{tkey}"] = 0.0
        st.session_state[f"tune_x_step_{tkey}"] = 0.0
        st.session_state[f"tune_y_step_{tkey}"] = 0.0
        st.session_state['last_tune_target'] = tune_reset_sig

    fc1, fc2 = st.columns(2)
    with fc1:
        st.subheader("⚙️ 파라미터 미세조정")
        tune_angle = st.number_input("전체 회전 각도 (°)", step=1.0, key=f"tune_angle_{tkey}")
        tune_pitch = st.number_input("피치 (mm)", step=0.1, key=f"tune_pitch_{tkey}")
        tune_width = st.number_input("소재 폭 (mm)", step=0.5, key=f"tune_width_{tkey}")
    with fc2:
        st.subheader("↕️ 위치 오프셋 (일괄 조절)")
        tune_y_all = st.number_input("전체 Y축 묶음 이동 (mm)", step=0.5, key=f"tune_y_all_{tkey}")
        tune_x_step = st.number_input("행간 X축 간격 조절 (mm)", step=0.5, disabled=(n_parts<2), key=f"tune_x_step_{tkey}")
        tune_y_step = st.number_input("행간 Y축 간격 조절 (mm)", step=0.5, disabled=(n_parts<2), key=f"tune_y_step_{tkey}")

    center_geom = target_best['parts'][0] if n_parts == 1 else unary_union(target_best['parts'])
    delta_angle = tune_angle - target_best['angle']
    tuned_parts = []
    
    # Gemini의 핵심 강점: 행(Row)간 일괄 Step Offset 적용
    for idx, geom in enumerate(target_best['parts']):
        g = rotate(geom, delta_angle, origin=center_geom.centroid)
        g = translate(g, 0, tune_y_all)  
        g = translate(g, idx * tune_x_step, idx * tune_y_step) 
        tuned_parts.append(g)

    all_geom_tuned = unary_union(tuned_parts)
    minx, miny, maxx, maxy = all_geom_tuned.bounds

    interference_tol = 0.01
    is_clashing = False
    base_min_pitch = calculate_1d_pitch(tuned_parts[0], bridge)
    if tune_pitch < base_min_pitch - interference_tol: is_clashing = True
    else:
        # 모든 행 쌍(인접하지 않은 조합 포함)을 같은 스테이션 내 / ±피치 이동 상태 모두 검사한다.
        # 기존에는 (r-1, r) 인접 쌍만 검사해 0행-2행처럼 인접하지 않은 행끼리의
        # 간섭(특히 사용자가 행간 오프셋을 직접 조정할 때)을 놓칠 수 있었다.
        row_bufs = [p.buffer(bridge - interference_tol, resolution=BRIDGE_BUFFER_RESOLUTION) for p in tuned_parts]
        for r in range(n_parts):
            for step in [-1, 1]:
                if row_bufs[r].intersects(translate(tuned_parts[r], xoff=step*tune_pitch, yoff=0)):
                    is_clashing = True
        for r1, r2 in itertools.combinations(range(n_parts), 2):
            if row_bufs[r1].intersects(tuned_parts[r2]):
                is_clashing = True
            for step in [-1, 1]:
                if row_bufs[r1].intersects(translate(tuned_parts[r2], xoff=step*tune_pitch, yoff=0)):
                    is_clashing = True

    if is_clashing: st.error(f"🚫 **간섭 경고:** 부품 간격이 브릿지({bridge}mm)를 침범합니다.")
    req_w = (maxy - miny) + margin * 2 + carrier_width * 2
    if tune_width < req_w - interference_tol: st.error(f"🚫 **폭 부족 경고:** 소재 폭이 최소 필요 폭({req_w:.2f}mm)보다 작습니다.")

    tune_util = (st.session_state['part_area'] * n_parts) / (tune_pitch * tune_width) * 100
    tune_cost = (((tune_pitch * tune_width * material_thickness) * material_density) / 1_000_000) * material_price / n_parts
    st.success(f"**미세조정 결과** ➔ 변경된 소재이용율: :blue[**{tune_util:.2f}%**] | 변경된 1개당 단가: :blue[**{int(tune_cost):,}원**]")

    total_len_tuned = (tune_pitch * (total_stations - 1)) + (maxx - minx) + (tune_pitch * 0.4)
    x_shift = -minx + (tune_pitch * 0.2)
    y_shift = -miny + margin + carrier_width + max(0.0, tune_width - req_w) / 2

    fig_tune, ax_tune = plt.subplots(figsize=(max(8, total_stations * 2), 4))
    ax_tune.plot([0, total_len_tuned, total_len_tuned, 0, 0], [0, 0, tune_width, tune_width, 0], color='red', linestyle='-', linewidth=2.5)
    if carrier_width > 0:
        ax_tune.add_patch(Rectangle((0, 0), total_len_tuned, carrier_width, facecolor='#999999', alpha=0.25, edgecolor='none', zorder=1))
        ax_tune.add_patch(Rectangle((0, tune_width - carrier_width), total_len_tuned, carrier_width, facecolor='#999999', alpha=0.25, edgecolor='none', zorder=1))

    for i in range(total_stations):
        for idx, geom in enumerate(tuned_parts):
            shifted = translate(geom, xoff=x_shift + (i * tune_pitch), yoff=y_shift)
            plot_polygon(ax_tune, shifted, row_colors(n_parts)[idx], lw=1.5, alpha=0.7)
        if carrier_width > 0 and 0 < pilot_dia < carrier_width:
            ax_tune.add_patch(Circle((tune_pitch * (i + 0.5), carrier_width / 2), pilot_dia / 2, facecolor='white', edgecolor='black', linewidth=1.2, zorder=5))
        if i < total_stations - 1:
            ax_tune.plot([tune_pitch * (i + 1), tune_pitch * (i + 1)], [0, tune_width], color='black', linestyle=':', alpha=0.4, zorder=1)

    ax_tune.axis('equal'); ax_tune.set_xticks([]); ax_tune.set_yticks([]) # <--- [디버깅 완료: set_yticks([]) 빈 리스트 추가]
    st.pyplot(fig_tune)

    # ============================================================
    # [8] 데이터 추출 및 내보내기 (DXF / Excel / PDF)
    # ============================================================
    st.divider()
    st.header("💾 [4단계] 데이터 추출 및 내보내기")
    
    dxf_bytes = generate_dxf_bytes(tuned_parts, tune_pitch, tune_width, total_stations, margin, carrier_width, pilot_dia, x_shift, y_shift)
    excel_bytes = generate_excel_report(tune_target_name, tune_pitch, tune_width, tune_util, tune_cost, total_stations, mat_type, material_thickness, material_price, material_density, s_res, i_res, z_res)
    pdf_bytes = generate_pdf_report(fig_tune, tune_target_name, tune_pitch, tune_width, tune_util, tune_cost, total_stations, mat_type, material_thickness)
    
    col_dxf, col_xls, col_pdf = st.columns(3)
    with col_dxf: st.download_button("📥 DXF 도면 다운로드", data=dxf_bytes, file_name="optimized_strip.dxf", mime="application/dxf", type="primary", use_container_width=True)
    with col_xls: st.download_button("📥 Excel 데이터 다운로드", data=excel_bytes, file_name="layout_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary", use_container_width=True)
    with col_pdf: st.download_button("📥 PDF 리포트 다운로드", data=pdf_bytes, file_name="layout_report.pdf", mime="application/pdf", type="primary", use_container_width=True)
