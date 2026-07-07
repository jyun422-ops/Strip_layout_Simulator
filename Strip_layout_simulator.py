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
            dxftype = entity.dxftype()
            p = path.make_path(entity)
            coords = [(v.x * unit_factor, v.y * unit_factor) for v in p.flattening(distance=flatten_tol)]
            if len(coords) < 3: continue

            if dxftype not in ('CIRCLE', 'ELLIPSE'):
                sx, sy = coords[0]
                ex, ey = coords[-1]
                if ((sx - ex) ** 2 + (sy - ey) ** 2) ** 0.5 > max(flatten_tol * 5, 1e-2):
                    continue

            poly = Polygon(coords).buffer(0)
            if poly.is_empty: continue
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
material_thickness = st.sidebar.number_input("소재 두께 (t)", value=1.2, step=0.1)
material_price = st.sidebar.number_input("단가 (원/kg)", value=1200, step=50)
material_density = st.sidebar.number_input("비중", value=7.85, step=0.01)

if "일반" in mat_type:
    rec_bridge, rec_margin = max(1.2, 1.2 * material_thickness), max(1.5, 1.5 * material_thickness)
else:
    rec_bridge, rec_margin = max(1.5, 1.5 * material_thickness), max(2.0, 1.8 * material_thickness)

st.sidebar.header("📏 2. 배열 간격 (다이 강도 고려)")
bridge = st.sidebar.number_input("부품간 최소 간격 (mm)", value=float(round(rec_bridge, 1)), step=0.1)
margin = st.sidebar.number_input("가장자리 마진 (mm)", value=float(round(rec_margin, 1)), step=0.1)

st.sidebar.header("🔧 3. 캐리어 & 파일럿 설계")
carrier_width = st.sidebar.number_input("캐리어(스켈레톤) 폭 (mm)", value=float(round(max(4.0, 3 * material_thickness), 1)), step=0.5, help="부품을 다음 스테이션으로 이송시키는 스켈레톤 밴드 폭. 스트립 상/하단에 배치됩니다.")
pilot_dia = st.sidebar.number_input("파일럿 홀 지름 (mm)", value=4.0, step=0.5)
if not (carrier_width > 0 and 0 < pilot_dia < carrier_width):
    st.sidebar.caption("⚠️ 파일럿 홀 지름이 캐리어 폭보다 크거나 같아 Layout도에 표시되지 않습니다.")

st.sidebar.header("🛠️ 4. Layout도 설계")
st_notch = st.sidebar.number_input("노칭 / 파이롯트 홀", value=1, step=1)
st_pierce = st.sidebar.number_input("피어싱 (내측 홀 타발)", value=1, step=1)
st_bend = st.sidebar.number_input("벤딩", value=0, step=1)
st_form = st.sidebar.number_input("포밍", value=0, step=1)
st_final_notch = st.sidebar.number_input("노칭 (최종 낙하)", value=1, step=1)
st_idle = st.sidebar.number_input("아이들 피치 (빈 구간)", value=1, step=1)
st_simul = st.sidebar.number_input("➖ 동시 성형 (중복 차감)", value=0, step=1)

total_stations = max(1, int((st_notch + st_pierce + st_bend + st_form + st_final_notch + st_idle) - st_simul))
st.sidebar.info(f"**총 예상 스테이션: {total_stations} 피치**")

st.sidebar.header("🧭 5. 압연방향(그레인) 제약")
apply_rolling_constraint = st.sidebar.checkbox("벤딩 라인 - 압연방향 최소 이격각 적용", value=(st_bend > 0), help="프레스 피드 방향(=압연방향, 도면 X축과 평행)과 벤딩 라인이 너무 나란하면 성형 시 크랙 위험이 커집니다.")
bend_angles_input = st.sidebar.text_input("벤딩 라인 각도 (°, 쉼표로 다중 입력)", value="0", disabled=not apply_rolling_constraint)
min_angle_from_rolling = st.sidebar.number_input("최소 이격각 (°, 통상 30~45° 권장)", value=30.0, step=5.0, disabled=not apply_rolling_constraint)

bend_line_angles = []
if apply_rolling_constraint:
    try:
        bend_line_angles = [float(x.strip()) for x in bend_angles_input.split(',')]
    except ValueError:
        st.sidebar.error("❌ 벤딩 라인 각도는 숫자와 쉼표(,)로만 입력해주세요. (예: 0, 90)")
        st.stop()


# ============================================================
# [6] 메인 화면 동작 및 결과 캐싱 로직
# ============================================================
st.markdown("#### 📂 1. 도면 업로드")
st.info("💡 **DXF 업로드 시 주의사항:** 정확한 계산을 위해 제품의 외곽선과 내부 피어싱 홀들은 반드시 각각 하나의 **'닫힌 폴리라인(Closed Polyline)'**으로 연결되어 있어야 합니다.")
uploaded_file = st.file_uploader("DXF 전개도면을 업로드하세요.", type=['dxf'])

st.markdown("#### 📐 2. 계산 각도 옵션 선택")
angle_step_option = st.radio("배열 회전 탐색 각도 간격을 선택하세요:", ["10도씩 (기본, 빠른 계산)", "5도씩 (정밀 계산)"], horizontal=True)
angle_step = 5 if "5도" in angle_step_option else 10

file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest() if uploaded_file else None

current_params = (
    file_hash, bridge, margin, carrier_width,
    material_thickness, material_density, material_price, 
    apply_rolling_constraint, tuple(bend_line_angles), min_angle_from_rolling, angle_step
)

if 'last_params' not in st.session_state:
    st.session_state.last_params = None

recalculated = False

if uploaded_file is not None:
    if st.session_state.last_params != current_params:
        recalculated = True 
        with st.spinner('안전 간격 적용 및 초정밀 형상 파고들기(Staggering)를 분석 중입니다...'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            doc = ezdxf.readfile(tmp_path)
            msp = doc.modelspace()
            os.remove(tmp_path)

            unit_factor, unit_msg = get_unit_factor(doc)
            part, holes = read_part_with_holes(msp, unit_factor)
            
            if part is None:
                st.error("❌ 도면에서 다각형 폴리라인을 찾을 수 없습니다.")
                st.stop()
                
            if part.geom_type == 'MultiPolygon':
                part = max(part.geoms, key=lambda a: a.area)
                
            part_area, part.area, part.area * 2

            single_results, inter_results, zigzag_results = [], [], []
            best_s = best_i = best_z = None
            fallback_s = fallback_i = fallback_z = None

            for angle in range(0, 180, angle_step):
                rotated_part = rotate(part, angle, origin='centroid')
                p_base = calculate_1d_pitch(rotated_part, bridge)

                valid = True
                if apply_rolling_constraint:
                    for b_angle in bend_line_angles:
                        eff_angle = (b_angle + angle) % 180
                        dist_from_parallel = min(eff_angle, 180 - eff_angle)
                        if dist_from_parallel < min_angle_from_rolling:
                            valid = False; break

                # [1] 단일 배열 계산
                minx, miny, maxx, maxy = rotated_part.bounds
                w_s = (maxy - miny) + margin * 2 + carrier_width * 2
                util_s = (part_area / (p_base * w_s)) * 100
                cost_s = (((p_base * w_s * material_thickness) * material_density) / 1_000_000) * material_price
                record_s = {'util': util_s, 'cost': cost_s, 'angle': angle, 'parts': [rotated_part], 'w': w_s, 'p': p_base}
                single_results.append({
                    '각도': f"{angle}°", '피치(mm)': round(p_base, 2), '소재폭(mm)': round(w_s, 2),
                    '소재이용율(%)': round(util_s, 2), '1개당 원가(원)': int(cost_s), '압연방향 적합': 'O' if valid else 'X'
                })
                if fallback_s is None or util_s > fallback_s['util']: fallback_s = record_s
                if valid and (best_s is None or util_s > best_s['util']): best_s = record_s

                # [2] 180도 교차 배열 계산
                part_180 = rotate(rotated_part, 180, origin='centroid')
                dx_i, dy_i, geom_i, part_b_i = find_nesting_offset(rotated_part, part_180, p_base, bridge)
                if part_b_i:
                    w_i = (geom_i.bounds[3] - geom_i.bounds[1]) + margin * 2 + carrier_width * 2
                    util_i = (pair_area / (p_base * w_i)) * 100
                    cost_i = (((p_base * w_i * material_thickness) * material_density) / 1_000_000) * material_price / 2
                    record_i = {'util': util_i, 'cost': cost_i, 'angle': angle, 'parts': [rotated_part, part_b_i], 'w': w_i, 'p': p_base}
                    inter_results.append({
                        '각도': f"{angle}°", '피치(mm)': round(p_base, 2), '소재폭(mm)': round(w_i, 2),
                        '소재이용율(%)': round(util_i, 2), '1개당 원가(원)': int(cost_i), '압연방향 적합': 'O' if valid else 'X'
                    })
                    if fallback_i is None or util_i > fallback_i['util']: fallback_i = record_i
                    if valid and (best_i is None or util_i > best_i['util']): best_i = record_i

                # [3] 지그재그 배열 계산
                part_same = rotated_part
                dx_z, dy_z, geom_z, part_b_z = find_nesting_offset(rotated_part, part_same, p_base, bridge)
                if part_b_z:
                    w_z = (geom_z.bounds[3] - geom_z.bounds[1]) + margin * 2 + carrier_width * 2
                    util_z = (pair_area / (p_base * w_z)) * 100
                    cost_z = (((p_base * w_z * material_thickness) * material_density) / 1_000_000) * material_price / 2
                    record_z = {'util': util_z, 'cost': cost_z, 'angle': angle, 'parts': [rotated_part, part_b_z], 'w': w_z, 'p': p_base}
                    zigzag_results.append({
                        '각도': f"{angle}°", '피치(mm)': round(p_base, 2), '소재폭(mm)': round(w_z, 2),
                        '소재이용율(%)': round(util_z, 2), '1개당 원가(원)': int(cost_z), '압연방향 적합': 'O' if valid else 'X'
                    })
                    if fallback_z is None or util_z > fallback_z['util']: fallback_z = record_z
                    if valid and (best_z is None or util_z > best_z['util']): best_z = record_z

            s_fallback_used = best_s is None
            if best_s is None: best_s = fallback_s
            i_fallback_used = best_i is None
            if best_i is None: best_i = fallback_i
            z_fallback_used = best_z is None
            if best_z is None: best_z = fallback_z

            st.session_state.update({
                'part_area': part_area, 'pair_area': pair_area, 'holes_count': len(holes),
                'holes_area': sum(h.area for h in holes) if holes else 0.0,
                'single': (single_results, best_s, s_fallback_used),
                'inter': (inter_results, best_i, i_fallback_used),
                'zigzag': (zigzag_results, best_z, z_fallback_used),
                'unit_msg': unit_msg
            })
            st.session_state.last_params = current_params

    s_res, best_s, s_fall = st.session_state['single']
    i_res, best_i, i_fall = st.session_state['inter']
    z_res, best_z, z_fall = st.session_state['zigzag']
    
    st.caption(st.session_state['unit_msg'])
    if st.session_state['holes_count'] > 0:
        st.info(f"🕳️ 내측 홀 **{st.session_state['holes_count']}개** 인식됨 (홀 면적 합계: {st.session_state['holes_area']:.1f} mm²) → 소재이용율·원가 계산에 순단면적이 반영되었습니다.")

    candidates = [('단일 배열', best_s)]
    if best_i: candidates.append(('180도 교차 배열', best_i))
    if best_z: candidates.append(('지그재그 배열', best_z))
    best_method_name, best_overall = min(candidates, key=lambda kv: kv[1]['cost'])
    
    saving_cost = int(best_s['cost'] - best_overall['cost'])
    st.success(f"🏆 분석 완료! 가장 훌륭한 배열은 **[{best_method_name}]**이며, 단일 배열 대비 1개당 :blue[**{saving_cost:,}원**]을 절감합니다. (캐리어 폭 {carrier_width}mm 포함 기준)")

    st.header("📊 [1단계] 단위 배열 최적화 결과")
    if apply_rolling_constraint:
        st.markdown("💡 **안내:** 표 내부의 <span style='background-color:#ffe6e6; padding:2px 6px; border-radius:4px;'>붉은색 행</span>은 압연방향(그레인) 이격각 조건을 불만족하여 **크랙(터짐) 불량 위험이 높은 기각(사용 불가) 배열**을 의미합니다.", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    render_case_column(col1, "[1] 단일 배열", s_res, best_s, s_fall, ['#004b87'], margin, carrier_width)
    
    if best_i: render_case_column(col2, "[2] 180도 교차 배열", i_res, best_i, i_fall, ['#004b87', '#007934'], margin, carrier_width)
    else: 
        with col2:
            st.subheader("[2] 180도 교차 배열")
            st.warning("교차 배열 불가 형상")
    
    if best_z: render_case_column(col3, "[3] 지그재그 배열", z_res, best_z, z_fall, ['#004b87', '#d55e00'], margin, carrier_width)
    else: 
        with col3:
            st.subheader("[3] 지그재그 배열")
            st.warning("지그재그 배열 불가 형상")

    st.divider()
    st.header("🎞️ [2단계] 스트립 Layout도 및 금형 코어 사이즈 도출")
    st.markdown(f"좌측에서 입력하신 **총 {total_stations} 피치**, **캐리어 폭 {carrier_width}mm**, **파일럿홀 ⌀{pilot_dia}mm** 조건을 반영한 실제 금형 내부 작업 구간 설계 도면입니다.")
    
    st.subheader("◼️ [1] 단일 배열 Layout도")
    render_strip_section("단일 배열", best_s, total_stations, margin, carrier_width, pilot_dia, ['#004b87'])
    
    st.divider()
    st.subheader("◼️ [2] 180도 교차 배열 Layout도")
    if best_i: 
        render_strip_section("180도 교차 배열", best_i, total_stations, margin, carrier_width, pilot_dia, ['#004b87', '#007934'])
    else:
        st.warning("이 부품은 180도 교차 배열이 불가능합니다.")
        
    st.divider()
    st.subheader("◼️ [3] 다열 지그재그 배열 Layout도")
    if best_z: 
        render_strip_section("지그재그 배열", best_z, total_stations, margin, carrier_width, pilot_dia, ['#004b87', '#d55e00'])
    else:
        st.warning("이 부품은 지그재그 배열이 불가능합니다.")


    # ============================================================
    # [7] 수동 미세 조정 (Fine-Tuning)
    # ============================================================
    st.divider()
    st.header("🛠️ [3단계] 수동 미세 조정 (Fine-Tuning)")
    st.markdown("자동으로 계산된 최적 배열을 바탕으로 실무 설계자의 감각에 맞춰 **숫자를 증감(+, -) 시키면 즉시 도면과 원가에 반영**됩니다.")

    if 'last_tune_target' not in st.session_state:
        st.session_state.last_tune_target = None

    valid_option_names = [k for k, v in candidates]

    if ('tune_target_select' not in st.session_state
            or st.session_state['tune_target_select'] not in valid_option_names):
        st.session_state['tune_target_select'] = best_method_name

    tune_target_name = st.selectbox("조정할 배열 방식 선택", options=valid_option_names, key='tune_target_select')
    target_best = next(v for k, v in candidates if k == tune_target_name)
    is_pair = tune_target_name != '단일 배열'

    tkey = tune_target_name

    tune_reset_signature = (file_hash, tune_target_name)
    if st.session_state.last_tune_target != tune_reset_signature or recalculated:
        st.session_state[f"tune_angle_{tkey}"] = float(target_best['angle'])
        st.session_state[f"tune_pitch_{tkey}"] = float(target_best['p'])
        st.session_state[f"tune_width_{tkey}"] = float(target_best['w'])
        st.session_state[f"tune_y1_{tkey}"] = 0.0
        st.session_state[f"tune_y2_{tkey}"] = 0.0
        st.session_state[f"tune_x2_{tkey}"] = 0.0
        st.session_state.last_tune_target = tune_reset_signature

    fc1, fc2 = st.columns(2)
    with fc1:
        st.subheader("⚙️ 파라미터 미세조정")
        tune_angle = st.number_input("전체 회전 각도 (°, 원본 기준)", step=1.0, key=f"tune_angle_{tkey}")
        tune_pitch = st.number_input("피치 (Pitch, mm)", step=0.1, min_value=0.1, key=f"tune_pitch_{tkey}")
        tune_width = st.number_input("소재 폭 (Width, mm)", step=0.5, min_value=0.1, key=f"tune_width_{tkey}")

    with fc2:
        st.subheader("↕️ 위치 오프셋 (Offset)")
        tune_y1 = st.number_input("파트 1 Y축 위치 이동 (mm)", step=0.5, key=f"tune_y1_{tkey}")
        tune_y2 = st.number_input("파트 2 Y축 위치 이동 (mm)", step=0.5, disabled=not is_pair, key=f"tune_y2_{tkey}")
        tune_x2 = st.number_input("파트 2 X축 위치 이동 (mm)", step=0.5, disabled=not is_pair, key=f"tune_x2_{tkey}")

    # 미세 조정된 형상 재계산
    if len(target_best['parts']) == 1:
        center_geom = target_best['parts'][0]
    else:
        center_geom = unary_union(target_best['parts'])

    delta_angle = tune_angle - target_best['angle']
    tuned_parts = []
    
    for idx, geom in enumerate(target_best['parts']):
        g = rotate(geom, delta_angle, origin=center_geom.centroid)
        if idx == 0:
            g = translate(g, 0, tune_y1)
        elif idx == 1:
            g = translate(g, tune_x2, tune_y2)
        tuned_parts.append(g)

    all_geom_tuned = tuned_parts[0] if len(tuned_parts) == 1 else unary_union(tuned_parts)
    minx, miny, maxx, maxy = all_geom_tuned.bounds

    # ⭐ 버그수정: 뭉툭한 단일 블록 간섭 계산 폐기, 개별 부품 단위 초정밀 센서 롤백 적용
    interference_tol = 0.01  
    is_clashing = False
    
    if is_pair:
        base_min_pitch = calculate_1d_pitch(tuned_parts[0], bridge)
        if tune_pitch < base_min_pitch - interference_tol:
            is_clashing = True
        else:
            buf_0 = tuned_parts[0].buffer(bridge - interference_tol, resolution=4)
            buf_1 = tuned_parts[1].buffer(bridge - interference_tol, resolution=4)
            
            # 파트1 vs 파트2 (동일 스테이션)
            if buf_0.intersects(tuned_parts[1]): is_clashing = True
            
            # 인접 스테이션 (-2, -1, 1, 2 피치) 검사
            for step in [-2, -1, 1, 2]:
                shift_x = step * tune_pitch
                if buf_0.intersects(translate(tuned_parts[0], xoff=shift_x, yoff=0)): is_clashing = True
                if buf_1.intersects(translate(tuned_parts[1], xoff=shift_x, yoff=0)): is_clashing = True
                if buf_0.intersects(translate(tuned_parts[1], xoff=shift_x, yoff=0)): is_clashing = True
    else:
        # 단일 배열
        base_min_pitch = calculate_1d_pitch(tuned_parts[0], bridge)
        if tune_pitch < base_min_pitch - interference_tol:
            is_clashing = True

    if is_clashing:
        st.error(f"🚫 **간섭 경고:** 현재 설정된 피치({tune_pitch:.2f}mm) 또는 오프셋 위치에서는 인접한 부품끼리 겹치거나 최소 브릿지 간격({bridge}mm)을 침범합니다. 값을 넉넉하게 조정하세요.")
        
    min_required_width = (maxy - miny) + margin * 2 + carrier_width * 2
    if tune_width < min_required_width - interference_tol:
        st.error(f"🚫 **폭 부족 경고:** 현재 소재 폭({tune_width:.2f}mm)이 이 형상을 담기 위한 "
                 f"최소 폭({min_required_width:.2f}mm)보다 작습니다. 부품이 마진/캐리어 영역을 벗어날 수 있습니다.")

    # 원가 재계산
    tune_util = (st.session_state['pair_area'] if is_pair else st.session_state['part_area']) / (tune_pitch * tune_width) * 100
    cost_divisor = 2 if is_pair else 1
    tune_cost = (((tune_pitch * tune_width * material_thickness) * material_density) / 1_000_000) * material_price / cost_divisor

    st.success(f"**미세조정 결과** ➔ 변경된 소재이용율: :blue[**{tune_util:.2f}%**]  |  변경된 1개당 단가: :blue[**{int(tune_cost):,}원**]")

    # 화면 렌더링용 기하학 계산
    part_length_tuned = maxx - minx
    total_length_tuned = (tune_pitch * (total_stations - 1)) + part_length_tuned + (tune_pitch * 0.4)
    x_shift = -minx + (tune_pitch * 0.2)

    extra_width = max(0.0, tune_width - min_required_width)
    y_shift = -
