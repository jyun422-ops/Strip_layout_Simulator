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
# [0] DXF 단위 변환 테이블
# ============================================================
INSUNITS_TO_MM = {
    0: None, 1: 25.4, 2: 304.8, 3: 1609344.0, 4: 1.0, 
    5: 10.0, 6: 1000.0, 13: 100.0,
}

def get_unit_factor(doc):
    try:
        code = doc.header.get('$INSUNITS', 0)
    except Exception:
        code = 0
    factor = INSUNITS_TO_MM.get(code, None)
    if factor is None:
        return 1.0, "⚠️ 도면에 단위 정보가 없거나 인식할 수 없어 **mm으로 간주**하고 계산합니다."
    if factor == 1.0:
        return 1.0, "✅ 도면 단위: mm (변환 불필요)"
    return factor, f"✅ 도면 단위 자동 인식: 환산 계수 ×{factor} 적용하여 mm로 변환했습니다."


# ============================================================
# [1] 핵심 알고리즘 (초정밀 파고들기 스캔 적용)
# ============================================================
def calculate_1d_pitch(geom, bridge):
    minx, miny, maxx, maxy = geom.bounds
    w = maxx - minx
    buffered_geom = geom.buffer(bridge, resolution=4)
    dx = w + bridge
    step = w / 30

    while dx > 0:
        test_geom = translate(geom, xoff=dx, yoff=0)
        if buffered_geom.intersects(test_geom):
            dx += step
            break
        dx -= step

    if dx <= 0: dx = step
    fine_step = step / 10
    while dx > 0:
        test_geom = translate(geom, xoff=dx, yoff=0)
        if buffered_geom.intersects(test_geom):
            dx += fine_step
            break
        dx -= fine_step
    return dx

def find_nesting_offset(part_a, part_b, p_base, bridge):
    buffered_a = part_a.buffer(bridge, resolution=4)
    minx, miny, maxx, maxy = part_a.bounds
    w, h = maxx - minx, maxy - miny

    reps = int(np.ceil((maxx - minx + p_base) / p_base)) + 2
    base_geoms = [translate(buffered_a, xoff=i*p_base, yoff=0) for i in range(-2, reps+1)]
    base_row = unary_union(base_geoms)

    best_dx, best_dy = 0, float('inf')
    best_part_b = None

    for dx in np.linspace(0, p_base, 90):
        dy = h + bridge * 2  
        step = max(0.5, min(h / 20, bridge / 2)) 
        test_b = translate(part_b, xoff=dx, yoff=dy)

        while not base_row.intersects(test_b):
            dy -= step
            if dy < -h * 1.5: break 
            test_b = translate(part_b, xoff=dx, yoff=dy)

        dy += step 
        fine_step = step / 10
        test_b = translate(part_b, xoff=dx, yoff=dy)

        while not base_row.intersects(test_b):
            dy -= fine_step
            if dy < -h * 1.5: break
            test_b = translate(part_b, xoff=dx, yoff=dy)
        dy += fine_step 

        if dy < best_dy:
            best_dy = dy
            best_dx = dx
            best_part_b = translate(part_b, xoff=dx, yoff=dy)

    if best_part_b:
        return best_dx, best_dy, unary_union([part_a, best_part_b]), best_part_b
    return None, None, None, None


# ============================================================
# [2] DXF 읽기 및 시각화(Rendering)
# ============================================================
def read_part_with_holes(msp, unit_factor):
    candidates = []
    flatten_tol = 0.1 if unit_factor == 0 else max(1e-4, 0.05 / unit_factor)
    for entity in msp.query('LWPOLYLINE POLYLINE CIRCLE ELLIPSE SPLINE'):
        try:
            p = path.make_path(entity)
            coords = [(v.x * unit_factor, v.y * unit_factor) for v in p.flattening(distance=flatten_tol)]
            if len(coords) < 3: continue

            poly = Polygon(coords).buffer(0)
            
            # ⭐ 버그수정: 끝점 일치 검사를 폐기하고, 면적이 거의 없는 선(Line) 찌꺼기만 필터링
            if poly.is_empty or poly.area < 1e-4: 
                continue
                
            if poly.geom_type == 'MultiPolygon':
                poly = max(poly.geoms, key=lambda a: a.area)
            candidates.append(poly)
        except Exception:
            continue

    if not candidates: return None, []

    outer = max(candidates, key=lambda a: a.area)
    holes = [c for c in candidates if c is not outer and outer.contains(c.buffer(-1e-6))]
    
    net_part = Polygon(outer.exterior.coords, [h.exterior.coords for h in holes]) if holes else outer
    return net_part, holes


def plot_polygon(ax, poly, color, lw=1.5, alpha=0.5):
    verts = []
    codes = []
    
    ext_coords = list(poly.exterior.coords)
    if not poly.exterior.is_ccw:
        ext_coords = ext_coords[::-1]
        
    verts.extend(ext_coords)
    codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(ext_coords) - 2) + [MplPath.CLOSEPOLY]
    
    for interior in poly.interiors:
        int_coords = list(interior.coords)
        if interior.is_ccw:
            int_coords = int_coords[::-1]
        verts.extend(int_coords)
        codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(int_coords) - 2) + [MplPath.CLOSEPOLY]
        
    patch = PathPatch(MplPath(verts, codes), facecolor=color, edgecolor=color, linewidth=lw, alpha=alpha, zorder=2)
    ax.add_patch(patch)
    
    for interior in poly.interiors:
        xs, ys = interior.xy
        ax.fill(xs, ys, color='white', alpha=1.0, zorder=3) 
        ax.plot(xs, ys, color=color, linewidth=lw * 0.8, zorder=4) 
        
    ax.plot(*poly.exterior.xy, color=color, linewidth=lw, zorder=4)


# ============================================================
# [3] 배열 각도 스캔 공통 로직
# ============================================================
def analyze_case(base_parts, origin, area_for_util, cost_divisor, bridge, margin, carrier_width,
                  material_thickness, material_density, material_price, check_rolling, bend_line_angles, min_angle_from_rolling, angle_step):
    results, best, fallback_best = [], None, None

    for angle in range(0, 180, angle_step):
        rotated = [rotate(g, angle, origin=origin) for g in base_parts]
        unioned = rotated[0] if len(rotated) == 1 else unary_union(rotated)

        p_val = calculate_1d_pitch(unioned, bridge)
        minx, miny, maxx, maxy = unioned.bounds
        w_val = (maxy - miny) + margin * 2 + carrier_width * 2
        util = (area_for_util / (p_val * w_val)) * 100
        cost = (((p_val * w_val * material_thickness) * material_density) / 1_000_000) * material_price / cost_divisor

        valid = True
        if check_rolling:
            for b_angle in bend_line_angles:
                eff_angle = (b_angle + angle) % 180
                dist_from_parallel = min(eff_angle, 180 - eff_angle)
                if dist_from_parallel < min_angle_from_rolling:
                    valid = False
                    break 

        results.append({
            '각도': f"{angle}°", '피치(mm)': round(p_val, 2), '소재폭(mm)': round(w_val, 2),
            '소재이용율(%)': round(util, 2), '1개당 원가(원)': int(cost), '압연방향 적합': 'O' if valid else 'X',
        })

        record = {'util': util, 'cost': cost, 'angle': angle, 'parts': rotated, 'w': w_val, 'p': p_val}
        if fallback_best is None or util > fallback_best['util']: fallback_best = record
        if valid and (best is None or util > best['util']): best = record

    used_fallback = best is None
    if best is None: best = fallback_best
    return results, best, used_fallback


def render_case_column(col, label, results, best, used_fallback, colors, margin, carrier_width):
    with col:
        st.subheader(f"{label} ({best['angle']}°)")
        st.caption(f"이용율: :blue[**{best['util']:.2f}%**] | 단가: :blue[**{int(best['cost']):,}원**]")
        if used_fallback:
            st.warning("⚠️ 압연방향 제약을 만족하는 각도가 없어 제약을 무시한 최적값입니다.")

        fig, ax = plt.subplots(figsize=(6, 6))
        for geom, color in zip(best['parts'], colors):
            plot_polygon(ax, geom, color)
        all_geom = best['parts'][0] if len(best['parts']) == 1 else unary_union(best['parts'])
        sx1, sy1 = all_geom.bounds[0], all_geom.bounds[1] - margin - carrier_width
        sy2 = all_geom.bounds[3] + margin + carrier_width
        
        ax.plot([sx1, sx1 + best['p'], sx1 + best['p'], sx1, sx1], [sy1, sy1, sy2, sy2, sy1], 
                color='red', linestyle='--', linewidth=2.5,
                label=f"1피치 소요 면적\n(폭: {best['w']:.1f} × 피치: {best['p']:.2f})")
        
        ax.axis('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), fontsize=9, framealpha=1.0)
        fig.subplots_adjust(bottom=0.28)  
        st.pyplot(fig)

        df = pd.DataFrame(results)  
        df['is_chosen'] = df['각도'] == f"{best['angle']}°"
        df = df.sort_values(by=['is_chosen', '소재이용율(%)'], ascending=[False, False]).reset_index(drop=True)
        df = df.drop(columns=['is_chosen'])
        
        def highlight(row):
            is_chosen = (row['각도'] == f"{best['angle']}°")
            if is_chosen: 
                return ['color: blue; font-weight: bold; background-color: #e6f2ff;'] * len(row)
            if row['압연방향 적합'] == 'X': 
                return ['background-color: #ffe6e6;'] * len(row)
            return [''] * len(row)
            
        st.dataframe(df.style.apply(highlight, axis=1).format({'피치(mm)': '{:.2f}', '소재폭(mm)': '{:.2f}', '소재이용율(%)': '{:.2f}'}), use_container_width=True)


# ============================================================
# [4] 스트립 Layout도 렌더링 및 DXF Export
# ============================================================
def plot_strip_layout(parts_and_colors, pitch, part_zone_width, margin, carrier_width, pilot_dia, total_stations):
    strip_width = part_zone_width + carrier_width * 2
    all_geoms = unary_union([p[0] for p in parts_and_colors])
    minx, miny, maxx, maxy = all_geoms.bounds
    part_length = maxx - minx
    total_length = (pitch * (total_stations - 1)) + part_length + (pitch * 0.4)

    fig, ax = plt.subplots(figsize=(max(8, total_stations * 2), 4))
    ax.plot([0, total_length, total_length, 0, 0], [0, 0, strip_width, strip_width, 0],
            color='red', linestyle='-', linewidth=2.5,
            label=f'금형 코어 최소 사이즈\n(가로: {total_length:.1f} x 세로: {strip_width:.1f})')

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

    ax.axis('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    return fig

def render_strip_section(label, best, total_stations, margin, carrier_width, pilot_dia, colors):
    all_geoms = best['parts'][0] if len(best['parts']) == 1 else unary_union(best['parts'])
    minx, miny, maxx, maxy = all_geoms.bounds
    part_length = maxx - minx
    l_val = (best['p'] * (total_stations - 1)) + part_length + (best['p'] * 0.4)
    
    st.info(f"📐 **{label} 금형 코어 최소 사이즈:** 가로(L) :blue[**{l_val:.1f} mm**] × 세로(W) :blue[**{best['w']:.1f} mm**] (캐리어 {carrier_width}mm 포함)  |  피치(P) :blue[**{best['p']:.2f} mm**] × **{total_stations}**스테이션")
    part_zone = best['w'] - carrier_width * 2
    fig = plot_strip_layout(list(zip(best['parts'], colors)), best['p'], part_zone, margin, carrier_width, pilot_dia, total_stations)
    st.pyplot(fig)

def generate_dxf_bytes(tuned_parts, tune_pitch, tune_width, total_stations, margin, carrier_width, pilot_dia, x_shift, y_shift):
    doc = ezdxf.new('R2010')
    doc.header['$INSUNITS'] = 4
    msp = doc.modelspace()
    doc.layers.add("STRIP_EDGE", color=1)
    doc.layers.add("PARTS", color=7)
    doc.layers.add("PILOT_HOLES", color=5)

    all_geom_tuned = tuned_parts[0] if len(tuned_parts) == 1 else unary_union(tuned_parts)
    minx, miny, maxx, maxy = all_geom_tuned.bounds
    part_length_tuned = maxx - minx
    total_length_tuned = (tune_pitch * (total_stations - 1)) + part_length_tuned + (tune_pitch * 0.4)

    msp.add_lwpolyline([(0, 0), (total_length_tuned, 0)], dxfattribs={'layer': 'STRIP_EDGE'})
    msp.add_lwpolyline([(0, tune_width), (total_length_tuned, tune_width)], dxfattribs={'layer': 'STRIP_EDGE'})
    
    def add_poly_to_msp(poly, layer_name):
        if poly.geom_type == 'MultiPolygon':
            for p in poly.geoms: add_poly_to_msp(p, layer_name)
            return
        msp.add_lwpolyline(list(poly.exterior.coords), dxfattribs={'layer': layer_name, 'closed': True})
        for interior in poly.interiors:
            msp.add_lwpolyline(list(interior.coords), dxfattribs={'layer': layer_name, 'closed': True})

    for i in range(total_stations):
        for geom in tuned_parts:
            shifted = translate(geom, xoff=x_shift + (i * tune_pitch), yoff=y_shift)
            add_poly_to_msp(shifted, "PARTS")
        if carrier_width > 0 and 0 < pilot_dia < carrier_width:
            cx, cy = tune_pitch * (i + 0.5), carrier_width / 2
            msp.add_circle(center=(cx, cy), radius=pilot_dia / 2, dxfattribs={'layer': 'PILOT_HOLES'})

    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
        doc.saveas(tmp.name)
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f: dxf_bytes = f.read()
    os.remove(tmp_path)
    return dxf_bytes


# ============================================================
# [5] 웹사이트 화면 및 사이드바
# ============================================================
st.set_page_config(page_title="프레스 레이아웃 최적화기", layout="wide")
st.title("⚙️ 프로그레시브 금형 스트립 설계 시뮬레이터")

st.sidebar.header("📝 1. 소재 조건 입력")
mat_type = st.sidebar.radio("소재 특성 분류", ["일반 철강/연질 (SPCC, AL 등)", "고장력강/경질 (STS, SUS 등)"])
material_thickness = st.sidebar.number_input("소재 두께 (t)", value=1.2
