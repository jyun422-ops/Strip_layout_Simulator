import streamlit as st
import ezdxf
from ezdxf import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Circle, Rectangle
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from shapely.ops import unary_union
import tempfile
import os

# ============================================================
# [1] 핵심 알고리즘: 1D 슬라이딩 피치 계산 (형상 파고들기 + 반복 주기 검증)
# ============================================================
def calculate_1d_pitch(geom, bridge):
    """
    geom(단일 부품 또는 부품 쌍 전체)을 x축으로 무한 반복 배열했을 때
    자기 자신과 충돌하지 않는 최소 피치를 계산.
    buffered_geom.intersects(translate(geom, dx))로 '한 피치 옆의 나(=다음 스테이션)'와
    현재 형상이 겹치는지 검사하므로, 반복 배열 시 발생하는 간섭까지 자동으로 걸러진다.
    """
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
# [2] DXF 읽기: 외곽선 + 내측 홀(피어싱) 인식  ---  ⭐ 개선 2순위
# ============================================================
def read_part_with_holes(msp):
    """
    도면 안의 모든 닫힌 폴리라인을 읽어서, 면적이 가장 큰 것을 외곽선으로,
    나머지 중 외곽선 내부에 포함된 것들을 내측 홀(피어싱/파일럿홀 등)로 인식한다.
    반환: (외곽 shapely Polygon(홀 포함), 홀 Polygon 리스트, 순단면적)
    """
    candidates = []
    for entity in msp.query('LWPOLYLINE POLYLINE'):
        try:
            p = path.make_path(entity)
            coords = [(v.x, v.y) for v in p.flattening(distance=0.1)]
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
        return None, [], 0.0

    # 면적이 가장 큰 도형 = 부품 외곽선
    outer = max(candidates, key=lambda a: a.area)
    holes = []
    for c in candidates:
        if c is outer:
            continue
        # 외곽선 내부에 완전히 포함되는 도형만 '내측 홀'로 인정 (피어싱/파일럿홀/노칭 등)
        if outer.contains(c.buffer(-1e-6)):
            holes.append(c)

    if holes:
        net_part = Polygon(outer.exterior.coords, [h.exterior.coords for h in holes])
    else:
        net_part = outer

    return net_part, holes, net_part.area


def plot_polygon(ax, poly, color, lw=1.5, alpha=0.5):
    """홀이 있는 shapely Polygon을 올바르게(구멍이 뚫린 채로) 렌더링."""
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
# [3] 스트립 Layout도 렌더링  ---  ⭐ 개선 3순위: 캐리어 + 파일럿홀 반영
# ============================================================
def plot_strip_layout(parts_and_colors, pitch, part_zone_width, margin, carrier_width,
                       pilot_dia, total_stations):
    """
    part_zone_width: 부품 외곽(bounds) + 마진까지 포함한 순수 성형 구간 폭
    carrier_width  : 상/하 캐리어(스켈레톤) 밴드 폭 (파일럿홀이 위치하는 이송 구간)
    최종 스트립 전체 폭 = part_zone_width + carrier_width * 2
    """
    strip_width = part_zone_width + carrier_width * 2
    fig, ax = plt.subplots(figsize=(max(8, total_stations * 2), 4))
    total_length = pitch * total_stations

    # 금형 코어(스트립) 외곽 박스
    ax.plot([0, total_length, total_length, 0, 0], [0, 0, strip_width, strip_width, 0],
            color='red', linestyle='-', linewidth=2.5,
            label=f'금형 코어 최소 사이즈\n(가로: {total_length:.1f} x 세로: {strip_width:.1f})')

    # 상/하 캐리어(스켈레톤) 밴드 음영 처리
    if carrier_width > 0:
        ax.add_patch(Rectangle((0, 0), total_length, carrier_width,
                                facecolor='#999999', alpha=0.25, edgecolor='none'))
        ax.add_patch(Rectangle((0, strip_width - carrier_width), total_length, carrier_width,
                                facecolor='#999999', alpha=0.25, edgecolor='none',
                                label='캐리어(스켈레톤) 영역'))

    all_geoms = unary_union([p[0] for p in parts_and_colors])
    minx, miny, maxx, maxy = all_geoms.bounds

    y_offset = -miny + margin + carrier_width
    x_offset = -minx + (pitch * 0.2)

    for i in range(total_stations):
        for geom, color in parts_and_colors:
            shifted = translate(geom, xoff=x_offset + (i * pitch), yoff=y_offset)
            plot_polygon(ax, shifted, color, lw=1.5, alpha=0.5)

        # 파일럿 홀: 하단 캐리어 밴드 중앙에 스테이션마다 표시
        if carrier_width > 0 and pilot_dia > 0 and pilot_dia < carrier_width:
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


# ============================================================
# [4] 웹사이트 화면 및 메뉴 구성
# ============================================================
st.set_page_config(page_title="프레스 레이아웃 최적화기", layout="wide")
st.title("⚙️ 프로그레시브 금형 스트립 설계 시뮬레이터")

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


# ============================================================
# [5] 메인 화면 동작 로직
# ============================================================
uploaded_file = st.file_uploader("DXF 전개도면을 업로드하세요.", type=['dxf'])

if uploaded_file is not None:
    with st.spinner('내측 홀 인식, 안전 간격 적용 및 정밀 형상 맞춤(Nesting)을 분석 중입니다... (약 15초 소요)'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        doc = ezdxf.readfile(tmp_path)
        msp = doc.modelspace()
        os.remove(tmp_path)

        part, holes, net_area = read_part_with_holes(msp)

        if part is None:
            st.error("❌ 도면에서 다각형 폴리라인을 찾을 수 없습니다.")
        else:
            if part.geom_type == 'MultiPolygon':
                part = max(part.geoms, key=lambda a: a.area)

            part_area, pair_area = part.area, part.area * 2  # 홀이 반영된 순단면적

            if holes:
                hole_area_sum = sum(h.area for h in holes)
                st.info(f"🕳️ 내측 홀 **{len(holes)}개** 인식됨 (홀 면적 합계: {hole_area_sum:.1f} mm²) → "
                        f"소재이용율·원가 계산에 순단면적이 반영되었습니다.")

            # --- [Case 1] 단일 배열 ---
            single_results = []
            best_s_util, best_s_cost, best_s_angle, best_s_part = 0, float('inf'), 0, None
            best_s_w, best_s_p = 0, 0
            for angle in range(0, 180, 10):
                rot = rotate(part, angle, origin='center')
                p_val = calculate_1d_pitch(rot, bridge)
                minx, miny, maxx, maxy = rot.bounds
                w_val = (maxy - miny) + (margin * 2) + (carrier_width * 2)
                util = (part_area / (p_val * w_val)) * 100
                cost = (((p_val * w_val * material_thickness) * material_density) / 1000000) * material_price
                single_results.append({'각도': f"{angle}°", '피치(mm)': round(p_val, 2), '소재폭(mm)': round(w_val, 2),
                                        '소재이용율(%)': round(util, 2), '1개당 원가(원)': int(cost)})
                if util > best_s_util:
                    best_s_util, best_s_cost, best_s_angle, best_s_part, best_s_w, best_s_p = util, cost, angle, rot, w_val, p_val

            # --- [Case 2] 180도 교차 배열 ---
            part_i_a, part_i_b, pair_i_geom = find_best_interlock(part, bridge)
            inter_results = []
            best_i_util, best_i_cost, best_i_angle, best_i_pair = 0, float('inf'), 0, None
            best_i_part_a, best_i_part_b = None, None
            best_i_w, best_i_p = 0, 0
            if pair_i_geom:
                for angle in range(0, 180, 10):
                    rot_a = rotate(part_i_a, angle, origin=pair_i_geom.centroid)
                    rot_b = rotate(part_i_b, angle, origin=pair_i_geom.centroid)
                    rot_pair = unary_union([rot_a, rot_b])
                    p_val = calculate_1d_pitch(rot_pair, bridge)
                    minx, miny, maxx, maxy = rot_pair.bounds
                    w_val = (maxy - miny) + (margin * 2) + (carrier_width * 2)
                    util = (pair_area / (p_val * w_val)) * 100
                    cost = ((((p_val * w_val * material_thickness) * material_density) / 1000000) * material_price) / 2
                    inter_results.append({'각도': f"{angle}°", '피치(mm)': round(p_val, 2), '소재폭(mm)': round(w_val, 2),
                                           '소재이용율(%)': round(util, 2), '1개당 원가(원)': int(cost)})
                    if util > best_i_util:
                        best_i_util, best_i_cost, best_i_angle, best_i_pair, best_i_w, best_i_p = util, cost, angle, rot_pair, w_val, p_val
                        best_i_part_a, best_i_part_b = rot_a, rot_b

            # --- [Case 3] 다열 지그재그 배열 ---
            part_z_a, part_z_b, pair_z_geom = find_best_zigzag(part, bridge)
            zigzag_results = []
            best_z_util, best_z_cost, best_z_angle, best_z_pair = 0, float('inf'), 0, None
            best_z_part_a, best_z_part_b = None, None
            best_z_w, best_z_p = 0, 0
            if pair_z_geom:
                for angle in range(0, 180, 10):
                    rot_a = rotate(part_z_a, angle, origin=pair_z_geom.centroid)
                    rot_b = rotate(part_z_b, angle, origin=pair_z_geom.centroid)
                    rot_pair = unary_union([rot_a, rot_b])
                    p_val = calculate_1d_pitch(rot_pair, bridge)
                    minx, miny, maxx, maxy = rot_pair.bounds
                    w_val = (maxy - miny) + (margin * 2) + (carrier_width * 2)
                    util = (pair_area / (p_val * w_val)) * 100
                    cost = ((((p_val * w_val * material_thickness) * material_density) / 1000000) * material_price) / 2
                    zigzag_results.append({'각도': f"{angle}°", '피치(mm)': round(p_val, 2), '소재폭(mm)': round(w_val, 2),
                                            '소재이용율(%)': round(util, 2), '1개당 원가(원)': int(cost)})
                    if util > best_z_util:
                        best_z_util, best_z_cost, best_z_angle, best_z_pair, best_z_w, best_z_p = util, cost, angle, rot_pair, w_val, p_val
                        best_z_part_a, best_z_part_b = rot_a, rot_b

            # --- [종합 판정] ---
            best_overall_cost = min(best_s_cost, best_i_cost, best_z_cost)
            best_method_name = "180도 교차 배열" if best_overall_cost == best_i_cost else (
                "지그재그 배열" if best_overall_cost == best_z_cost else "단일 배열")
            saving_cost = int(best_s_cost - best_overall_cost)

            st.success(f"🏆 분석 완료! 가장 훌륭한 배열은 **[{best_method_name}]**이며, 단일 배열 대비 1개당 "
                       f":blue[**{saving_cost:,}원**]을 절감합니다. (캐리어 폭 {carrier_width}mm 포함 기준)")

            # ==========================================
            # 화면 분리 1: 단위 배열 최적화 결과
            # ==========================================
            st.header("📊 [1단계] 단위 배열 최적화 결과")
            format_dict = {'피치(mm)': '{:.2f}', '소재폭(mm)': '{:.2f}', '소재이용율(%)': '{:.2f}'}
            col1, col2, col3 = st.columns(3)

            def highlight_best(row, max_util):
                if row['소재이용율(%)'] == max_util:
                    return ['color: blue; font-weight: bold; background-color: #e6f2ff;'] * len(row)
                return [''] * len(row)

            with col1:
                st.subheader(f"[1] 단일 배열 ({best_s_angle}°)")
                st.caption(f"이용율: :blue[**{best_s_util:.2f}%**] | 단가: :blue[**{int(best_s_cost):,}원**]")
                fig1, ax1 = plt.subplots(figsize=(6, 6))
                plot_polygon(ax1, best_s_part, '#004b87')
                sx1 = best_s_part.bounds[0]
                sy1 = best_s_part.bounds[1] - margin - carrier_width
                sy2 = best_s_part.bounds[3] + margin + carrier_width
                ax1.plot([sx1, sx1 + best_s_p, sx1 + best_s_p, sx1, sx1], [sy1, sy1, sy2, sy2, sy1],
                         color='red', linestyle='--', linewidth=2.5)
                ax1.axis('equal'); ax1.set_xticks([]); ax1.set_yticks([])
                st.pyplot(fig1)
                df_single = pd.DataFrame(single_results)
                st.dataframe(df_single.style.apply(lambda r: highlight_best(r, df_single['소재이용율(%)'].max()), axis=1).format(format_dict), use_container_width=True)

            with col2:
                st.subheader(f"[2] 180도 교차 배열 ({best_i_angle}°)")
                st.caption(f"이용율: :blue[**{best_i_util:.2f}%**] | 단가: :blue[**{int(best_i_cost):,}원**]")
                if pair_i_geom:
                    fig2, ax2 = plt.subplots(figsize=(6, 6))
                    plot_polygon(ax2, best_i_part_a, '#004b87')
                    plot_polygon(ax2, best_i_part_b, '#007934')
                    sx1 = best_i_pair.bounds[0]
                    sy1 = best_i_pair.bounds[1] - margin - carrier_width
                    sy2 = best_i_pair.bounds[3] + margin + carrier_width
                    ax2.plot([sx1, sx1 + best_i_p, sx1 + best_i_p, sx1, sx1], [sy1, sy1, sy2, sy2, sy1],
                             color='red', linestyle='--', linewidth=2.5)
                    ax2.axis('equal'); ax2.set_xticks([]); ax2.set_yticks([])
                    st.pyplot(fig2)
                    df_inter = pd.DataFrame(inter_results)
                    st.dataframe(df_inter.style.apply(lambda r: highlight_best(r, df_inter['소재이용율(%)'].max()), axis=1).format(format_dict), use_container_width=True)
                else:
                    st.warning("교차 배열 불가 형상")

            with col3:
                st.subheader(f"[3] 지그재그 배열 ({best_z_angle}°)")
                st.caption(f"이용율: :blue[**{best_z_util:.2f}%**] | 단가: :blue[**{int(best_z_cost):,}원**]")
                if best_z_part_a is not None:
                    fig3, ax3 = plt.subplots(figsize=(6, 6))
                    plot_polygon(ax3, best_z_part_a, '#004b87')
                    plot_polygon(ax3, best_z_part_b, '#d55e00')
                    sx1 = best_z_pair.bounds[0]
                    sy1 = best_z_pair.bounds[1] - margin - carrier_width
                    sy2 = best_z_pair.bounds[3] + margin + carrier_width
                    ax3.plot([sx1, sx1 + best_z_p, sx1 + best_z_p, sx1, sx1], [sy1, sy1, sy2, sy2, sy1],
                             color='red', linestyle='--', linewidth=2.5)
                    ax3.axis('equal'); ax3.set_xticks([]); ax3.set_yticks([])
                    st.pyplot(fig3)
                    df_zigzag = pd.DataFrame(zigzag_results)
                    st.dataframe(df_zigzag.style.apply(lambda r: highlight_best(r, df_zigzag['소재이용율(%)'].max()), axis=1).format(format_dict), use_container_width=True)
                else:
                    st.warning("지그재그 배열 불가 형상")

            # ==========================================
            # 화면 분리 2: Layout도 도면 및 금형 사이즈 (캐리어 + 파일럿홀 포함)
            # ==========================================
            st.divider()
            st.header("🎞️ [2단계] 스트립 Layout도 및 금형 코어 사이즈 도출")
            st.markdown(f"좌측에서 입력하신 **총 {total_stations} 피치**, **캐리어 폭 {carrier_width}mm**, "
                        f"**파일럿홀 ⌀{pilot_dia}mm** 조건을 반영한 실제 금형 내부 작업 구간 설계 도면입니다.")

            st.subheader("◼️ [1] 단일 배열 Layout도")
            l_val_s = best_s_p * total_stations
            st.info(f"📐 **단일 배열 금형 코어 최소 사이즈:** 가로(L) :blue[**{l_val_s:.1f} mm**] × 세로(W) :blue[**{best_s_w:.1f} mm**] (캐리어 포함)")
            part_zone_s = best_s_w - carrier_width * 2
            fig_strip1 = plot_strip_layout([(best_s_part, '#004b87')], best_s_p, part_zone_s, margin,
                                            carrier_width, pilot_dia, total_stations)
            st.pyplot(fig_strip1)

            st.divider()
            st.subheader("◼️ [2] 180도 교차 배열 Layout도")
            if pair_i_geom:
                l_val_i = best_i_p * total_stations
                st.info(f"📐 **180도 교차 배열 금형 코어 최소 사이즈:** 가로(L) :blue[**{l_val_i:.1f} mm**] × 세로(W) :blue[**{best_i_w:.1f} mm**] (캐리어 포함)")
                part_zone_i = best_i_w - carrier_width * 2
                fig_strip2 = plot_strip_layout([(best_i_part_a, '#004b87'), (best_i_part_b, '#007934')],
                                                best_i_p, part_zone_i, margin, carrier_width, pilot_dia, total_stations)
                st.pyplot(fig_strip2)
            else:
                st.warning("이 부품은 180도 교차 배열이 불가능합니다.")

            st.divider()
            st.subheader("◼️ [3] 다열 지그재그 배열 Layout도")
            if best_z_part_a is not None:
                l_val_z = best_z_p * total_stations
                st.info(f"📐 **지그재그 배열 금형 코어 최소 사이즈:** 가로(L) :blue[**{l_val_z:.1f} mm**] × 세로(W) :blue[**{best_z_w:.1f} mm**] (캐리어 포함)")
                part_zone_z = best_z_w - carrier_width * 2
                fig_strip3 = plot_strip_layout([(best_z_part_a, '#004b87'), (best_z_part_b, '#d55e00')],
                                                best_z_p, part_zone_z, margin, carrier_width, pilot_dia, total_stations)
                st.pyplot(fig_strip3)
            else:
                st.warning("이 부품은 지그재그 배열이 불가능합니다.")
