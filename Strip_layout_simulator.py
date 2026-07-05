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
    0: None,      
    1: 25.4,      
    2: 304.8,     
    3: 1609344.0, 
    4: 1.0,       
    5: 10.0,      
    6: 1000.0,    
    13: 100.0,    
}


def get_unit_factor(doc):
    try:
        code = doc.header.get('$INSUNITS', 0)
    except Exception:
        code = 0
    factor = INSUNITS_TO_MM.get(code, None)
    if factor is None:
        return 1.0, f"⚠️ 도면에 단위 정보($INSUNITS={code})가 없거나 인식할 수 없어 **mm으로 간주**하고 계산합니다."
    if factor == 1.0:
        return 1.0, "✅ 도면 단위: mm (변환 불필요)"
    return factor, f"✅ 도면 단위 자동 인식: 환산 계수 ×{factor} 적용하여 mm로 변환했습니다."


# ============================================================
# [1] 핵심 알고리즘: 1D 슬라이딩 피치 계산 (형상 파고들기 + 반복 주기 검증)
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

    if dx <= 0:
        dx = step

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
            if dy < -h * 1.5:
                break
            test_b = translate(part_b_same, xoff=dx, yoff=dy)

        dy += step
        fine_step = step / 10
        test_b = translate(part_b_same, xoff=dx, yoff=dy)
        while not (buffered_a1.intersects(test_b) or buffered_a2.intersects(test_b)):
            dy -= fine_step
            if dy < -h * 1.5:
                break
            test_b = translate(part_b_same, xoff=dx, yoff=dy)
        dy += fine_step

        if dy < best_dy:
            best_dy = dy
            best_part_b = translate(part_b_same, xoff=dx, yoff=dy)

    if best_part_b:
        try:
            best_pair_geom = unary_union([best_part_a, best_part_b])
        except Exception:
            pass

    return best_part_a, best_part_b, best_pair_geom


# ============================================================
# [2] DXF 읽기: 단위 변환 + 외곽선 + 내측 홀(피어싱) 인식
# ============================================================
def read_part_with_holes(msp, unit_factor):
    candidates = []
    flatten_tol = 0.1 if unit_factor == 0 else max(1e-4, 0.05 / unit_factor)
    for entity in msp.query('LWPOLYLINE POLYLINE'):
        try:
            p = path.make_path(entity)
            coords = [(v.x * unit_factor, v.y * unit_factor) for v in p.flattening(distance=flatten_tol)]
            if len(coords) < 3:
                continue
            poly = Polygon(coords).buffer(0)
            if poly.is_empty:
                continue
            if poly.geom_type == 'MultiPolygon':
                poly = max(poly.geoms, key=lambda a: a.area)
            candidates.append(poly)
        except Exception:
            continue

    if not candidates:
        return None, []

    outer = max(candidates, key=lambda a: a.area)
    holes = []
    for c in candidates:
        if c is outer:
            continue
        if outer.contains(c.buffer(-1e-6)):
            holes.append(c)

    if holes:
        net_part = Polygon(outer.exterior.coords, [h.exterior.coords for h in holes])
    else:
        net_part = outer

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
                  material_thickness, material_density, material_price,
                  check_rolling, bend_line_angle, min_angle_from_rolling):
    results = []
    best, fallback_best = None, None

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
            '소재이용율(%)': round(util, 2), '1개당 원가(원)': int(cost),
            '압연방향 적합': 'O' if valid else 'X',
        })

        record = {'util': util, 'cost': cost, 'angle': angle, 'parts': rotated, 'w': w_val, 'p': p_val}
        if fallback_best is None or util > fallback_best['util']:
            fallback_best = record
        if valid and (best is None or util > best['util']):
            best = record

    used_fallback = best is None
    if best is None:
        best = fallback_best
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
        sx1 = all_geom.bounds[0]
        sy1 = all_geom.bounds[1] - margin - carrier_width
        sy2 = all_geom.bounds[3] + margin + carrier_width
        ax.plot([sx1, sx1 + best['p'], sx1 + best['p'], sx1, sx1], [sy1, sy1, sy2, sy2, sy1],
                 color='red', linestyle='--', linewidth=2.5)
        ax.axis('equal'); ax.set_xticks([]); ax.set_yticks([])
        st.pyplot(fig)

        df = pd.DataFrame(results)
        max_util = df['소재이용율(%)'].max()

        def highlight(row):
            if row['압연방향 적합'] == 'X':
                return ['background-color: #ffe6e6;'] * len(row)
            if row['소재이용율(%)'] == max_util:
                return ['color: blue; font-weight: bold; background-color: #e6f2ff;'] * len(row)
            return [''] * len(row)

        st.dataframe(
            df.style.apply(highlight, axis=1).format(
                {'피치(mm)': '{:.2f}', '소재폭(mm)': '{:.2f}', '소재이용율(%)': '{:.2f}'}),
            use_container_width=True)


# ============================================================
# [4] 스트립 Layout도 렌더링 (캐리어 + 파일럿홀 포함)
# ============================================================
def plot_strip_layout(parts_and_colors, pitch, part_zone_width, margin, carrier_width,
                       pilot_dia, total_stations):
    strip_width = part_zone_width + carrier_width * 2
    
    # ⭐ 수정: 부품 1개의 실제 가로(X) 길이를 추출하여 금형 코어 길이에 반영
    all_geoms = unary_union([p[0] for p in parts_and_colors])
    minx, miny, maxx, maxy = all_geoms.bounds
    part_length = maxx - minx
    
    # ⭐ 수정: 마지막 스테이션 부품 전체를 덮을 수 있도록 총 길이 계산식 보완
    total_length = (pitch * (total_stations - 1)) + part_length + (pitch * 0.4)

    fig, ax = plt.subplots(figsize=(max(8, total_stations * 2), 4))

    ax.plot([0, total_length, total_length, 0, 0], [0, 0, strip_width, strip_width, 0],
            color='red', linestyle='-', linewidth=2.5,
            label=f'금형 코어 최소 사이즈\n(가로: {total_length:.1f} x 세로: {strip_width:.1f})')

    if carrier_width > 0:
        ax.add_patch(Rectangle((0, 0), total_length, carrier_width,
                                facecolor='#999999', alpha=0.25, edgecolor='none'))
        ax.add_patch(Rectangle((0, strip_width - carrier_width), total_length, carrier_width,
                                facecolor='#999999', alpha=0.25, edgecolor='none',
                                label='캐리어(스켈레톤) 영역'))

    y_offset = -miny + margin + carrier_width
    x_offset = -minx + (pitch * 0.2)

    for i in range(total_stations):
        for geom, color in parts_and_colors:
            shifted = translate(geom, xoff=x_offset + (i * pitch), yoff=y_offset)
            plot_polygon(ax, shifted, color, lw=1.5, alpha=0.5)

        if carrier_width > 0 and 0 < pilot_dia < carrier_width:
            cx = pitch * (i + 0.5)
            cy = carrier_width / 2
            ax.add_patch(Circle((cx, cy), pilot_dia / 2, facecolor='white',
                                 edgecolor='black', linewidth=1.2, zorder=5))

        if i < total_stations - 1:
            ax.plot([pitch * (i + 1), pitch * (i + 1)], [0, strip_width],
                    color='black', linestyle=':', alpha=0.4)

    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    return fig


def render_strip_section(label, best, total_stations, margin, carrier_width, pilot_dia, colors):
    # ⭐ 수정: 텍스트에 표기되는 가로(L) 길이에도 동일한 산출 공식 적용
    all_geoms = best['parts'][0] if len(best['parts']) == 1 else unary_union(best['parts'])
    minx, miny, maxx, maxy = all_geoms.bounds
    part_length = maxx - minx
    
    l_val = (best['p'] * (total_stations - 1)) + part_length + (best['p'] * 0.4)
    
    st.info(f"📐 **{label} 금형 코어 최소 사이즈:** 가로(L) :blue[**{l_val:.1f} mm**] × "
            f"세로(W) :blue[**{best['w']:.1f} mm**] (캐리어 {carrier_width}mm 포함)  |  "
            f"피치(P) :blue[**{best['p']:.2f} mm**] × **{total_stations}**스테이션")
            
    part_zone = best['w'] - carrier_width * 2
    parts_and_colors = list(zip(best['parts'], colors))
    fig = plot_strip_layout(parts_and_colors, best['p'], part_zone, margin, carrier_width, pilot_dia, total_stations)
    st.pyplot(fig)


# ============================================================
# [5] 웹사이트 화면 및 메뉴 구성
# ============================================================
st.set_page_config(page_title="프레스 레이아웃 최적화기", layout="wide")
st.title("⚙️ 프로그레시브 금형 스트립 설계 시뮬레이터")

if KOREAN_FONT_FOUND is None:
    st.warning(
        "⚠️ 서버에 한글 폰트가 설치되어 있지 않아 아래 **Layout도 이미지(matplotlib) 안의 한글 텍스트**가 "
        "네모 박스로 깨져 보일 수 있습니다 (Streamlit UI 텍스트/표는 정상 표시됩니다).\n\n"
        "Streamlit Community Cloud에 배포한 경우, 저장소 루트에 `packages.txt` 파일을 만들고 "
        "`fonts-nanum` 한 줄을 추가한 뒤 재배포하면 해결됩니다."
    )

st.sidebar.header("📝 1. 소재 조건 입력")
mat_type = st.sidebar.radio("소재 특성 분류", ["일반 철강/연질 (SPCC, AL 등)", "고장력강/경질 (STS, SUS 등)"])
material_thickness = st.sidebar.number_input("소재 두께 (t)", value=1.2, step=0.1)
material_price = st.sidebar.number_input("단가 (원/kg)", value=1200, step=50)
material_density = st.sidebar.number_input("비중", value=7.85, step=0.01)

if "일반" in mat_type:
    rec_bridge = max(1.2, 1.2 * material_thickness)
    rec_margin = max(1.5, 1.5 * material_thickness)
else:
    rec_bridge = max(1.5, 1.5 * material_thickness)
    rec_margin = max(2.0, 1.8 * material_thickness)

st.sidebar.header("📏 2. 배열 간격 (다이 강도 고려)")
bridge = st.sidebar.number_input("최소 브릿지 (mm)", value=float(round(rec_bridge, 1)), step=0.1)
margin = st.sidebar.number_input("가장자리 마진 (mm)", value=float(round(rec_margin, 1)), step=0.1)

st.sidebar.header("🔧 3. 캐리어 & 파일럿 설계")
carrier_width = st.sidebar.number_input(
    "캐리어(스켈레톤) 폭 (mm)", value=float(round(max(4.0, 3 * material_thickness), 1)), step=0.5,
    help="부품을 다음 스테이션으로 이송시키는 스켈레톤 밴드 폭. 스트립 상/하단에 배치됩니다.")
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
apply_rolling_constraint = st.sidebar.checkbox(
    "벤딩 라인 - 압연방향 최소 이격각 적용", value=(st_form > 0),
    help="프레스 피드 방향(=압연방향, 도면 X축과 평행)과 벤딩 라인이 너무 나란하면 성형 시 크랙 위험이 커집니다.")
bend_line_angle = st.sidebar.number_input(
    "부품 좌표계 기준 벤딩 라인 각도 (°, 0=부품 X축과 평행)", value=0.0, step=5.0,
    disabled=not apply_rolling_constraint)
min_angle_from_rolling = st.sidebar.number_input(
    "최소 이격각 (°, 통상 30~45° 권장)", value=30.0, step=5.0,
    disabled=not apply_rolling_constraint)


# ============================================================
# [6] 메인 화면 동작 로직
# ============================================================
uploaded_file = st.file_uploader("DXF 전개도면을 업로드하세요.", type=['dxf'])

if uploaded_file is not None:
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
        else:
            st.caption(unit_msg)

            if part.geom_type == 'MultiPolygon':
                part = max(part.geoms, key=lambda a: a.area)

            part_area, pair_area = part.area, part.area * 2  

            if holes:
                hole_area_sum = sum(h.area for h in holes)
                st.info(f"🕳️ 내측 홀 **{len(holes)}개** 인식됨 (홀 면적 합계: {hole_area_sum:.1f} mm²) → "
                        f"소재이용율·원가 계산에 순단면적이 반영되었습니다.")

            common_kwargs = dict(
                bridge=bridge, margin=margin, carrier_width=carrier_width,
                material_thickness=material_thickness, material_density=material_density,
                material_price=material_price, check_rolling=apply_rolling_constraint,
                bend_line_angle=bend_line_angle, min_angle_from_rolling=min_angle_from_rolling,
            )

            # --- [Case 1] 단일 배열 ---
            single_results, best_s, s_fallback = analyze_case(
                base_parts=[part], origin='center', area_for_util=part_area, cost_divisor=1, **common_kwargs)

            # --- [Case 2] 180도 교차 배열 ---
            part_i_a, part_i_b, pair_i_geom = find_best_interlock(part, bridge)
            inter_results, best_i, i_fallback = (None, None, None)
            if pair_i_geom:
                inter_results, best_i, i_fallback = analyze_case(
                    base_parts=[part_i_a, part_i_b], origin=pair_i_geom.centroid,
                    area_for_util=pair_area, cost_divisor=2, **common_kwargs)

            # --- [Case 3] 다열 지그재그 배열 ---
            part_z_a, part_z_b, pair_z_geom = find_best_zigzag(part, bridge)
            zigzag_results, best_z, z_fallback = (None, None, None)
            if pair_z_geom:
                zigzag_results, best_z, z_fallback = analyze_case(
                    base_parts=[part_z_a, part_z_b], origin=pair_z_geom.centroid,
                    area_for_util=pair_area, cost_divisor=2, **common_kwargs)

            # --- [종합 판정] ---
            candidates = [('단일 배열', best_s)]
            if best_i: candidates.append(('180도 교차 배열', best_i))
            if best_z: candidates.append(('지그재그 배열', best_z))
            best_method_name, best_overall = min(candidates, key=lambda kv: kv[1]['cost'])
            saving_cost = int(best_s['cost'] - best_overall['cost'])

            st.success(f"🏆 분석 완료! 가장 훌륭한 배열은 **[{best_method_name}]**이며, 단일 배열 대비 1개당 "
                       f":blue[**{saving_cost:,}원**]을 절감합니다. (캐리어 폭 {carrier_width}mm 포함 기준)")

            # ==========================================
            # 화면 분리 1: 단위 배열 최적화 결과
            # ==========================================
            st.header("📊 [1단계] 단위 배열 최적화 결과")
            col1, col2, col3 = st.columns(3)

            render_case_column(col1, "[1] 단일 배열", single_results, best_s, s_fallback,
                                ['#004b87'], margin, carrier_width)

            if best_i:
                render_case_column(col2, "[2] 180도 교차 배열", inter_results, best_i, i_fallback,
                                    ['#004b87', '#007934'], margin, carrier_width)
            else:
                with col2:
                    st.subheader("[2] 180도 교차 배열")
                    st.warning("교차 배열 불가 형상")

            if best_z:
                render_case_column(col3, "[3] 지그재그 배열", zigzag_results, best_z, z_fallback,
                                    ['#004b87', '#d55e00'], margin, carrier_width)
            else:
                with col3:
                    st.subheader("[3] 지그재그 배열")
                    st.warning("지그재그 배열 불가 형상")

            # ==========================================
            # 화면 분리 2: Layout도 도면 및 금형 사이즈
            # ==========================================
            st.divider()
            st.header("🎞️ [2단계] 스트립 Layout도 및 금형 코어 사이즈 도출")
            st.markdown(f"좌측에서 입력하신 **총 {total_stations} 피치**, **캐리어 폭 {carrier_width}mm**, "
                        f"**파일럿홀 ⌀{pilot_dia}mm** 조건을 반영한 실제 금형 내부 작업 구간 설계 도면입니다.")

            st.subheader("◼️ [1] 단일 배열 Layout도")
            render_strip_section("단일 배열", best_s, total_stations, margin, carrier_width, pilot_dia, ['#004b87'])

            st.divider()
            st.subheader("◼️ [2] 180도 교차 배열 Layout도")
            if best_i:
                render_strip_section("180도 교차 배열", best_i, total_stations, margin, carrier_width, pilot_dia,
                                      ['#004b87', '#007934'])
            else:
                st.warning("이 부품은 180도 교차 배열이 불가능합니다.")

            st.divider()
            st.subheader("◼️ [3] 다열 지그재그 배열 Layout도")
            if best_z:
                render_strip_section("지그재그 배열", best_z, total_stations, margin, carrier_width, pilot_dia,
                                      ['#004b87', '#d55e00'])
            else:
                st.warning("이 부품은 지그재그 배열이 불가능합니다.")
