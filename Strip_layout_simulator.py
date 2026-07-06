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
# [1] 핵심 알고리즘
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

def find_best_interlock(part, bridge):
    part_b_rotated = rotate(part, 180, origin='centroid')
    buffered_a = part.buffer(bridge, resolution=4)
    minx, miny, maxx, maxy = part.bounds
    w, h = maxx - minx, maxy - miny

    best_pair_geom, best_part_a, best_part_b = None, part, None
    min_box_area = float('inf')

    for dy in np.linspace(-h * 0.8, h * 0.8, 20):
        dx = w + bridge
        step = w / 30
        while dx > -w:
            test_b = translate(part_b_rotated, xoff=dx, yoff=dy)
            if buffered_a.intersects(test_b):
                dx += step
                break
            dx -= step
        fine_step = step / 10
        while dx > -w:
            test_b = translate(part_b_rotated, xoff=dx, yoff=dy)
            if buffered_a.intersects(test_b):
                dx += fine_step
                break
            dx -= fine_step

        test_b = translate(part_b_rotated, xoff=dx, yoff=dy)
        try:
            pair = unary_union([part, test_b])
            p_minx, p_miny, p_maxx, p_maxy = pair.bounds
            box_area = (p_maxx - p_minx) * (p_maxy - p_miny)
            if box_area < min_box_area:
                min_box_area, best_pair_geom, best_part_b = box_area, pair, test_b
        except Exception:
            continue
    return best_part_a, best_part_b, best_pair_geom

def find_best_zigzag(part, bridge):
    part_b_same = part
    p_base = calculate_1d_pitch(part, bridge)
    buffered_a1 = part.buffer(bridge, resolution=4)
    buffered_a2 = translate(buffered_a1, xoff=p_base, yoff=0)
    minx, miny, maxx, maxy = part.bounds
    h = maxy - miny
    best_pair_geom, best_part_a, best_part_b = None, part, None
    best_dy = float('inf')

    for dx in np.linspace(0, p_base, 20):
        dy = h
        step = h / 20
        test_b = translate(part_b_same, xoff=dx, yoff=dy)
        while not (buffered_a1.intersects(test_b) or buffered_a2.intersects(test_b)):
            dy -= step
            if dy < -h * 1.5: break
            test_b = translate(part_b_same, xoff=dx, yoff=dy)

        dy += step
        fine_step = step / 10
        test_b = translate(part_b_same, xoff=dx, yoff=dy)
        while not (buffered_a1.intersects(test_b) or buffered_a2.intersects(test_b)):
            dy -= fine_step
            if dy < -h * 1.5: break
            test_b = translate(part_b_same, xoff=dx, yoff=dy)
        dy += fine_step

        if dy < best_dy:
            best_dy = dy
            best_part_b = translate(part_b_same, xoff=dx, yoff=dy)

    if best_part_b:
        try: best_pair_geom = unary_union([best_part_a, best_part_b])
        except Exception: pass
    return best_part_a, best_part_b, best_pair_geom


# ============================================================
# [2] DXF 읽기
# ============================================================
def read_part_with_holes(msp, unit_factor):
    candidates = []
    flatten_tol = 0.1 if unit_factor == 0 else max(1e-4, 0.05 / unit_factor)
    for entity in msp.query('LWPOLYLINE POLYLINE'):
        try:
            p = path.make_path(entity)
            coords = [(v.x * unit_factor, v.y * unit_factor) for v in p.flattening(distance=flatten_tol)]
            if len(coords) < 3: continue
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
    ext = np.array(poly.exterior.coords)
    verts = ext.tolist()
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(ext) - 2) + [MplPath.CLOSEPOLY]
    for interior in poly.interiors:
        intc = np.array(interior.coords)
        verts += intc.tolist()
        codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(intc) - 2) + [MplPath.CLOSEPOLY]
    patch = PathPatch(MplPath(verts, codes), facecolor=color, edgecolor=color, linewidth=lw, alpha=alpha)
    ax.add_patch(patch)
    ax.plot(*poly.exterior.xy, color=color, linewidth=lw)
    for interior in poly.interiors:
        xs, ys = interior.xy
        ax.plot(xs, ys, color=color, linewidth=lw * 0.8)


# ============================================================
# [3] 배열 각도 스캔 공통 로직
# ============================================================
def analyze_case(base_parts, origin, area_for_util, cost_divisor, bridge, margin, carrier_width,
                  material_thickness, material_density, material_price, check_rolling, bend_line_angle, min_angle_from_rolling):
    results, best, fallback_best = [], None, None

    for angle in range(0, 180, 10):
        rotated = [rotate(g, angle, origin=origin) for g in base_parts]
        unioned = rotated[0] if len(rotated) == 1 else unary_union(rotated)

        p_val = calculate_1d_pitch(unioned, bridge)
        minx, miny, maxx, maxy = unioned.bounds
        w_val = (maxy - miny) + margin * 2 + carrier_width * 2
        util = (area_for_util / (p_val * w_val)) * 100
        cost = (((p_val * w_val * material_thickness) * material_density) / 1_000_000) * material_price / cost_divisor

        valid = True
        if check_rolling:
            eff_angle = (bend_line_angle + angle) % 180
            dist_from_parallel = min(eff_angle, 180 - eff_angle)
            valid = dist_from_parallel >= min_angle_from_rolling

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
            st.warning("⚠️ 압연방향 제약을 만족하는 각도가 없어 제약을 무시한 최적값입니다. 벤딩 라인 방향이나 최소 이격각을 재검토하세요.")

        fig, ax = plt.subplots(figsize=(6, 6))
        for geom, color in zip(best['parts'], colors):
            plot_polygon(ax, geom, color)
        all_geom = best['parts'][0] if len(best['parts']) == 1 else unary_union(best['parts'])
        sx1, sy1 = all_geom.bounds[0], all_geom.bounds[1] - margin - carrier_width
        sy2 = all_geom.bounds[3] + margin + carrier_width
        ax.plot([sx1, sx1 + best['p'], sx1 + best['p'], sx1, sx1], [sy1, sy1, sy2, sy2, sy1], color='red', linestyle='--', linewidth=2.5)
        ax.axis('equal'); ax.set_xticks([]); ax.set_yticks([])
        st.pyplot(fig)

        df = pd.DataFrame(results)
        max_util = df['소재이용율(%)'].max()
        def highlight(row):
            if row['압연방향 적합'] == 'X': return ['background-color: #ffe6e6;'] * len(row)
            if row['소재이용율(%)'] == max_util: return ['color: blue; font-weight: bold; background-color: #e6f2ff;'] * len(row)
            return [''] * len(row)
        st.dataframe(df.style.apply(highlight, axis=1).format({'피치(mm)': '{:.2f}', '소재폭(mm)': '{:.2f}', '소재이용율(%)': '{:.2f}'}), use_container_width=True)


# ============================================================
# [4] 스트립 Layout도 렌더링 및 DXF Export (우선순위 2)
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
        ax.add_patch(Rectangle((0, 0), total_length, carrier_width, facecolor='#999999', alpha=0.25, edgecolor='none'))
        ax.add_patch(Rectangle((0, strip_width - carrier_width), total_length, carrier_width, facecolor='#999999', alpha=0.25, edgecolor='none', label='캐리어(스켈레톤) 영역'))

    y_offset, x_offset = -miny + margin + carrier_width, -minx + (pitch * 0.2)

    for i in range(total_stations):
        for geom, color in parts_and_colors:
            shifted = translate(geom, xoff=x_offset + (i * pitch), yoff=y_offset)
            plot_polygon(ax, shifted, color, lw=1.5, alpha=0.5)

        if carrier_width > 0 and 0 < pilot_dia < carrier_width:
            ax.add_patch(Circle((pitch * (i + 0.5), carrier_width / 2), pilot_dia / 2, facecolor='white', edgecolor='black', linewidth=1.2, zorder=5))
        if i < total_stations - 1:
            ax.plot([pitch * (i + 1), pitch * (i + 1)], [0, strip_width], color='black', linestyle=':', alpha=0.4)

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

# ⭐ DXF Export를 위한 Shapely Polygon -> DXF 변환 로직
def generate_dxf_bytes(tuned_parts, tune_pitch, tune_width, total_stations, margin, carrier_width, pilot_dia, y_shift):
    """미세 조정이 끝난 최종 배열 데이터를 CAD DXF 파일 포맷으로 변환하여 바이트 스트림으로 반환합니다."""
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    # 레이어 및 색상 설정
    doc.layers.add("STRIP_EDGE", color=1)      # 빨간색 (외곽 가이드라인)
    doc.layers.add("PARTS", color=7)           # 흰색/검정색 (제품 형상)
    doc.layers.add("PILOT_HOLES", color=5)     # 파란색 (파일럿 홀)

    # 1. 스트립 외곽 가이드라인 렌더링
    all_geom_tuned = tuned_parts[0] if len(tuned_parts) == 1 else unary_union(tuned_parts)
    minx, miny, maxx, maxy = all_geom_tuned.bounds
    part_length_tuned = maxx - minx
    total_length_tuned = (tune_pitch * (total_stations - 1)) + part_length_tuned + (tune_pitch * 0.4)

    msp.add_lwpolyline([(0, 0), (total_length_tuned, 0)], dxfattribs={'layer': 'STRIP_EDGE'})
    msp.add_lwpolyline([(0, tune_width), (total_length_tuned, tune_width)], dxfattribs={'layer': 'STRIP_EDGE'})
    
    # Shapely Polygon을 DXF 객체로 추가하는 헬퍼 함수
    def add_poly_to_msp(poly, layer_name):
        if poly.geom_type == 'MultiPolygon':
            for p in poly.geoms:
                add_poly_to_msp(p, layer_name)
            return
        
        # 제품 외곽선 (Exterior)
        ext_coords = list(poly.exterior.coords)
        msp.add_lwpolyline(ext_coords, dxfattribs={'layer': layer_name, 'closed': True})
        
        # 제품 내측 홀 (Interiors)
        for interior in poly.interiors:
            int_coords = list(interior.coords)
            msp.add_lwpolyline(int_coords, dxfattribs={'layer': layer_name, 'closed': True})

    # 2. 각 스테이션별 부품 및 파일럿 홀 배치
    for i in range(total_stations):
        # 부품 형상 평행이동 후 DXF에 삽입
        for geom in tuned_parts:
            shifted = translate(geom, xoff=(i * tune_pitch), yoff=y_shift)
            add_poly_to_msp(shifted, "PARTS")

        # 파일럿 홀 삽입
        if carrier_width > 0 and 0 < pilot_dia < carrier_width:
            cx = tune_pitch * (i + 0.5)
            cy = carrier_width / 2
            msp.add_circle(center=(cx, cy), radius=pilot_dia / 2, dxfattribs={'layer': 'PILOT_HOLES'})

    # 임시 파일로 저장 후 바이트 데이터 반환
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
        doc.saveas(tmp.name)
        tmp_path = tmp.name
        
    with open(tmp_path, "rb") as f:
        dxf_bytes = f.read()
        
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
bridge = st.sidebar.number_input("최소 브릿지 (mm)", value=float(round(rec_bridge, 1)), step=0.1)
margin = st.sidebar.number_input("가장자리 마진 (mm)", value=float(round(rec_margin, 1)), step=0.1)

st.sidebar.header("🔧 3. 캐리어 & 파일럿 설계")
carrier_width = st.sidebar.number_input("캐리어(스켈레톤) 폭 (mm)", value=float(round(max(4.0, 3 * material_thickness), 1)), step=0.5, help="부품을 다음 스테이션으로 이송시키는 스켈레톤 밴드 폭. 스트립 상/하단에 배치됩니다.")
pilot_dia = st.sidebar.number_input("파일럿 홀 지름 (mm)", value=4.0, step=0.5)

st.sidebar.header("🛠️ 4. Layout도 설계")
st_notch = st.sidebar.number_input("노칭 / 파이롯트 홀", value=1, step=1)
st_pierce = st.sidebar.number_input("피어싱 (내측 홀 타발)", value=1, step=1)
st_form = st.sidebar.number_input("벤딩 / 포밍", value=0, step=1)
st_blank = st.sidebar.number_input("블랭킹 (최종 낙하)", value=1, step=1)
st_idle = st.sidebar.number_input("아이들 피치 (빈 구간)", value=1, step=1)
st_simul = st.sidebar.number_input("➖ 동시 성형 (중복 차감)", value=0, step=1)
total_stations = max(1, int((st_notch + st_pierce + st_form + st_blank + st_idle) - st_simul))
st.sidebar.info(f"**총 예상 스테이션: {total_stations} 피치**")

st.sidebar.header("🧭 5. 압연방향(그레인) 제약")
apply_rolling_constraint = st.sidebar.checkbox("벤딩 라인 - 압연방향 최소 이격각 적용", value=(st_form > 0), help="프레스 피드 방향(=압연방향, 도면 X축과 평행)과 벤딩 라인이 너무 나란하면 성형 시 크랙 위험이 커집니다.")
bend_line_angle = st.sidebar.number_input("부품 좌표계 기준 벤딩 라인 각도 (°, 0=부품 X축과 평행)", value=0.0, step=5.0, disabled=not apply_rolling_constraint)
min_angle_from_rolling = st.sidebar.number_input("최소 이격각 (°, 통상 30~45° 권장)", value=30.0, step=5.0, disabled=not apply_rolling_constraint)


# ============================================================
# [6] 메인 화면 동작 및 결과 캐싱 로직
# ============================================================
uploaded_file = st.file_uploader("DXF 전개도면을 업로드하세요.", type=['dxf'])

current_params = (
    uploaded_file.name if uploaded_file else None, bridge, margin, carrier_width,
    material_thickness, material_density, material_price, 
    apply_rolling_constraint, bend_line_angle, min_angle_from_rolling
)

if 'last_params' not in st.session_state:
    st.session_state.last_params = None

if uploaded_file is not None:
    if st.session_state.last_params != current_params:
        with st.spinner('단위 변환, 내측 홀 인식, 안전 간격 적용 및 정밀 형상 맞춤(Nesting)을 분석 중입니다... (약 15초 소요)'):
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
                
            part_area, pair_area = part.area, part.area * 2

            common_kwargs = dict(
                bridge=bridge, margin=margin, carrier_width=carrier_width,
                material_thickness=material_thickness, material_density=material_density,
                material_price=material_price, check_rolling=apply_rolling_constraint,
                bend_line_angle=bend_line_angle, min_angle_from_rolling=min_angle_from_rolling,
            )

            single_results, best_s, s_fallback = analyze_case([part], 'center', part_area, 1, **common_kwargs)
            
            part_i_a, part_i_b, pair_i_geom = find_best_interlock(part, bridge)
            inter_results, best_i, i_fallback = (None, None, None)
            if pair_i_geom:
                inter_results, best_i, i_fallback = analyze_case([part_i_a, part_i_b], pair_i_geom.centroid, pair_area, 2, **common_kwargs)

            part_z_a, part_z_b, pair_z_geom = find_best_zigzag(part, bridge)
            zigzag_results, best_z, z_fallback = (None, None, None)
            if pair_z_geom:
                zigzag_results, best_z, z_fallback = analyze_case([part_z_a, part_z_b], pair_z_geom.centroid, pair_area, 2, **common_kwargs)

            st.session_state.update({
                'part_area': part_area, 'pair_area': pair_area, 'holes_count': len(holes),
                'holes_area': sum(h.area for h in holes) if holes else 0.0,
                'single': (single_results, best_s, s_fallback),
                'inter': (inter_results, best_i, i_fallback),
                'zigzag': (zigzag_results, best_z, z_fallback),
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

    tune_target_name = st.selectbox("조정할 배열 방식 선택", options=[k for k, v in candidates], index=[k for k, v in candidates].index(best_method_name))
    target_best = next(v for k, v in candidates if k == tune_target_name)
    is_pair = tune_target_name != '단일 배열'

    fc1, fc2 = st.columns(2)
    with fc1:
        st.subheader("⚙️ 파라미터 미세조정")
        tune_angle = st.number_input("전체 회전 각도 (°, 원본 기준)", value=float(target_best['angle']), step=1.0)
        tune_pitch = st.number_input("피치 (Pitch, mm)", value=float(target_best['p']), step=0.1)
        tune_width = st.number_input("소재 폭 (Width, mm)", value=float(target_best['w']), step=0.5)

    with fc2:
        st.subheader("↕️ 위치 오프셋 (Offset)")
        tune_y1 = st.number_input("파트 1 Y축 위치 이동 (mm)", value=0.0, step=0.5)
        tune_y2 = st.number_input("파트 2 Y축 위치 이동 (mm)", value=0.0, step=0.5, disabled=not is_pair)
        tune_x2 = st.number_input("파트 2 X축 위치 이동 (mm)", value=0.0, step=0.5, disabled=not is_pair)

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

    # 원가 재계산
    tune_util = (st.session_state['pair_area'] if is_pair else st.session_state['part_area']) / (tune_pitch * tune_width) * 100
    cost_divisor = 2 if is_pair else 1
    tune_cost = (((tune_pitch * tune_width * material_thickness) * material_density) / 1_000_000) * material_price / cost_divisor

    st.success(f"**미세조정 결과** ➔ 변경된 소재이용율: :blue[**{tune_util:.2f}%**]  |  변경된 1개당 단가: :blue[**{int(tune_cost):,}원**]")

    # 화면 렌더링용 기하학 계산
    all_geom_tuned = tuned_parts[0] if len(tuned_parts) == 1 else unary_union(tuned_parts)
    minx, miny, maxx, maxy = all_geom_tuned.bounds
    part_length_tuned = maxx - minx
    total_length_tuned = (tune_pitch * (total_stations - 1)) + part_length_tuned + (tune_pitch * 0.4)
    y_shift = -miny + margin + carrier_width

    fig_tune, ax_tune = plt.subplots(figsize=(max(8, total_stations * 2), 4))
    ax_tune.plot([0, total_length_tuned, total_length_tuned, 0, 0], 
                 [0, 0, tune_width, tune_width, 0], color='red', linestyle='-', linewidth=2.5,
                 label=f'조정 후 코어 사이즈\n(가로: {total_length_tuned:.1f} x 세로: {tune_width:.1f})')
    
    if carrier_width > 0:
        ax_tune.add_patch(Rectangle((0, 0), total_length_tuned, carrier_width, facecolor='#999999', alpha=0.25, edgecolor='none'))
        ax_tune.add_patch(Rectangle((0, tune_width - carrier_width), total_length_tuned, carrier_width, facecolor='#999999', alpha=0.25, edgecolor='none'))

    for i in range(total_stations):
        for idx, geom in enumerate(tuned_parts):
            color = ['#004b87', '#007934'][idx] if is_pair else '#004b87'
            if tune_target_name == '지그재그 배열': color = ['#004b87', '#d55e00'][idx]
            shifted = translate(geom, xoff=(i * tune_pitch), yoff=y_shift)
            plot_polygon(ax_tune, shifted, color, lw=1.5, alpha=0.7)

        if carrier_width > 0 and 0 < pilot_dia < carrier_width:
            ax_tune.add_patch(Circle((tune_pitch * (i + 0.5), carrier_width / 2), pilot_dia / 2, facecolor='white', edgecolor='black', linewidth=1.2, zorder=5))
        if i < total_stations - 1:
            ax_tune.plot([tune_pitch * (i + 1), tune_pitch * (i + 1)], [0, tune_width], color='black', linestyle=':', alpha=0.4)

    ax_tune.axis('equal'); ax_tune.set_xticks([]); ax_tune.set_yticks([])
    ax_tune.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    st.pyplot(fig_tune)

    # ============================================================
    # [8] CAD 작업용 DXF Export (우선순위 2 반영)
    # ============================================================
    st.divider()
    st.subheader("💾 [4단계] CAD 도면(DXF) 내보내기")
    st.markdown("위에서 미세조정을 마친 최종 배열을 CAD(AutoCAD 등)에서 바로 작업할 수 있도록 **DXF 파일로 다운로드**합니다. 레이어 분리(외곽선/제품/파일럿홀)가 적용되어 있어 실무 설계에 즉시 활용 가능합니다.")
    
    # Export 버튼을 누를 때마다 백그라운드에서 DXF Byte를 즉시 생성
    dxf_bytes = generate_dxf_bytes(
        tuned_parts=tuned_parts,
        tune_pitch=tune_pitch,
        tune_width=tune_width,
        total_stations=total_stations,
        margin=margin,
        carrier_width=carrier_width,
        pilot_dia=pilot_dia,
        y_shift=y_shift
    )
    
    st.download_button(
        label="📥 최종 스트립 레이아웃 DXF 다운로드",
        data=dxf_bytes,
        file_name="optimized_strip_layout.dxf",
        mime="application/dxf",
        type="primary"
    )
