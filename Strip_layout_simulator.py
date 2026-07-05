import streamlit as st
import ezdxf
from ezdxf import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from shapely.ops import unary_union
import tempfile
import os

# --- [1] 핵심 알고리즘: 정밀 네스팅 및 피치 계산 ---

def get_true_pitch(part, bridge):
    """단일 형상 또는 결합된 쌍(Pair)이 가로로 연속 배열될 때의 진짜 최소 피치 계산"""
    minx, miny, maxx, maxy = part.bounds
    w = maxx - minx
    # 조인 스타일(join_style=2, Mitre)을 적용해 코너에서도 브릿지 간격이 좁아지지 않도록 엄격히 보호
    buffered_part = part.buffer(bridge + 0.05, join_style=2) 
    
    dx = w + bridge
    step = w / 20
    # 1. 먼저 충돌할 때까지 좌측으로 이동
    while dx > 0:
        test = translate(part, xoff=dx, yoff=0)
        if buffered_part.intersects(test):
            break
        dx -= step
        
    if dx <= 0: return w + bridge # 예외 처리
    
    # 2. 이진 탐색(Binary Search)으로 0.01mm 단위의 완벽한 밀착 지점 찾기
    low = dx
    high = dx + step
    best_dx = high
    for _ in range(12):
        mid = (low + high) / 2
        test = translate(part, xoff=mid, yoff=0)
        if buffered_part.intersects(test):
            low = mid # 너무 가까움 (충돌)
        else:
            high = mid # 안전함
            best_dx = mid
            
    return best_dx

def find_best_interlock(part, bridge):
    """180도 교차 배열 시 최적의 상/하 맞물림 위치 찾기"""
    part_b_rotated = rotate(part, 180, origin='centroid')
    buffered_a = part.buffer(bridge + 0.05, join_style=2)
    minx, miny, maxx, maxy = part.bounds
    w, h = maxx - minx, maxy - miny
    
    best_pair_geom, best_part_b = None, None
    min_box_area = float('inf')
    
    for dy in np.linspace(-h*0.9, h*0.9, 40):
        dx = w * 2.0
        step = w / 20
        while dx > -w:
            test_b = translate(part_b_rotated, xoff=dx, yoff=dy)
            if buffered_a.intersects(test_b): break
            dx -= step
            
        if dx <= -w: continue
            
        low, high = dx, dx + step
        valid_dx = high
        for _ in range(10):
            mid = (low + high) / 2
            test_b = translate(part_b_rotated, xoff=mid, yoff=dy)
            if buffered_a.intersects(test_b): low = mid
            else: high, valid_dx = mid, mid
                
        test_b = translate(part_b_rotated, xoff=valid_dx, yoff=dy)
        pair = unary_union([part, test_b])
        p_minx, p_miny, p_maxx, p_maxy = pair.bounds
        box_area = (p_maxx - p_minx) * (p_maxy - p_miny)
        if box_area < min_box_area:
            min_box_area, best_pair_geom, best_part_b = box_area, pair, test_b
                
    return part, best_part_b, best_pair_geom

def find_best_zigzag(part, bridge, p_val):
    """진정한 다열 지그재그 배열: Row1의 피치를 기준으로 Row2를 최적으로 파고들게 배치"""
    buffered_a1 = part.buffer(bridge + 0.05, join_style=2)
    buffered_a2 = translate(buffered_a1, xoff=p_val, yoff=0)
    
    minx, miny, maxx, maxy = part.bounds
    h = maxy - miny
    best_pair_geom, best_part_b = None, None
    best_strip_w = float('inf')
    
    for dx in np.linspace(0, p_val, 25):
        dy = h * 1.5
        step = h / 20
        # 위에서 아래로 내리면서 Row 1과 충돌하는 지점 탐색
        while dy > -h:
            test_b = translate(part, xoff=dx, yoff=dy)
            if buffered_a1.intersects(test_b) or buffered_a2.intersects(test_b): break
            dy -= step
        
        if dy <= -h: continue
            
        low, high = dy, dy + step
        valid_dy = high
        for _ in range(10):
            mid = (low + high) / 2
            test_b = translate(part, xoff=dx, yoff=mid)
            if buffered_a1.intersects(test_b) or buffered_a2.intersects(test_b): low = mid
            else: high, valid_dy = mid, mid
        
        overall_miny = min(miny, miny + valid_dy)
        overall_maxy = max(maxy, maxy + valid_dy)
        strip_w = overall_maxy - overall_miny
        
        if strip_w < best_strip_w:
            best_strip_w = strip_w
            best_part_b = translate(part, xoff=dx, yoff=valid_dy)
            
    if best_part_b:
        best_pair_geom = unary_union([part, best_part_b])
        
    return part, best_part_b, best_pair_geom

# --- [도면 렌더링 함수] ---
def plot_strip_layout(parts_and_colors, pitch, strip_width, margin, total_stations):
    fig, ax = plt.subplots(figsize=(max(8, total_stations * 2), 3.5))
    total_length = pitch * total_stations + margin
    
    ax.plot([0, total_length, total_length, 0, 0], [0, 0, strip_width, strip_width, 0], 
            color='red', linestyle='-', linewidth=2.5, 
            label=f'금형 코어 최소 사이즈\n(가로: {total_length:.1f} x 세로: {strip_width:.1f})')
    
    all_geoms = unary_union([p[0] for p in parts_and_colors])
    minx, miny, maxx, maxy = all_geoms.bounds
    
    x_offset = -minx + margin
    y_offset = -miny + margin
    
    for i in range(total_stations):
        for geom, color in parts_and_colors:
            shifted = translate(geom, xoff=x_offset + (i * pitch), yoff=y_offset)
            ax.plot(*shifted.exterior.xy, color=color, linewidth=1.5)
            ax.fill(*shifted.exterior.xy, alpha=0.5, color=color)
        
        if i < total_stations - 1:
            ax.plot([margin + pitch * (i+1), margin + pitch * (i+1)], [0, strip_width], color='black', linestyle=':', alpha=0.4)
            
    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5)) 
    plt.tight_layout()
    return fig


# --- [2] 웹사이트 화면 및 메뉴 구성 ---
st.set_page_config(page_title="프레스 레이아웃 최적화기", layout="wide")
st.title("⚙️ 프로그레시브 금형 스트립 설계 시뮬레이터")

st.sidebar.header("📝 1. 소재 조건 입력")
mat_type = st.sidebar.radio("소재 특성 분류", ["일반 철강/연질 (SPCC, AL 등)", "고장력강/경질 (STS, SUS 등)"])
material_thickness = st.sidebar.number_input("소재 두께 (t)", value=1.2, step=0.1)
material_price = st.sidebar.number_input("단가 (원/kg)", value=1200, step=50)
material_density = st.sidebar.number_input("비중", value=7.85, step=0.01)

# 자동 추천 브릿지 로직
if "일반" in mat_type:
    rec_bridge = max(1.2, 1.2 * material_thickness)
    rec_margin = max(1.5, 1.5 * material_thickness)
else:
    rec_bridge = max(1.5, 1.5 * material_thickness)
    rec_margin = max(2.0, 1.8 * material_thickness)

st.sidebar.header("📏 2. 배열 간격 (다이 강도 고려)")
st.sidebar.caption("소재 두께를 기반으로 다이 파손을 막는 권장 최소값이 자동 세팅됩니다. 필요시 수정하세요.")
bridge = st.sidebar.number_input("최소 브릿지 (mm)", value=float(round(rec_bridge, 1)), step=0.1)
margin = st.sidebar.number_input("가장자리 마진 (mm)", value=float(round(rec_margin, 1)), step=0.1)

st.sidebar.header("🛠️ 3. Layout도 설계")
st_notch = st.sidebar.number_input("노칭 / 파이롯트 홀", value=1, step=1)
st_pierce = st.sidebar.number_input("피어싱 (내측 홀 타발)", value=1, step=1)
st_form = st.sidebar.number_input("벤딩 / 포밍", value=0, step=1)
st_blank = st.sidebar.number_input("블랭킹 (최종 낙하)", value=1, step=1)
st_idle = st.sidebar.number_input("아이들 피치 (빈 구간)", value=1, step=1)
st_simul = st.sidebar.number_input("➖ 동시 성형 (중복 차감)", value=0, step=1)

total_stations = max(1, int((st_notch + st_pierce + st_form + st_blank + st_idle) - st_simul))
st.sidebar.info(f"**총 예상 스테이션: {total_stations} 피치**")


# --- [3] 메인 화면 동작 로직 ---
uploaded_file = st.file_uploader("DXF 전개도면을 업로드하세요.", type=['dxf'])

if uploaded_file is not None:
    with st.spinner('다이 강도를 고려한 정밀 네스팅(Nesting) 최적화를 진행 중입니다... (약 15초 소요)'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        doc = ezdxf.readfile(tmp_path)
        msp = doc.modelspace()
        os.remove(tmp_path) 

        part_coords = []
        for entity in msp.query('LWPOLYLINE'):
            try:
                p = path.make_path(entity)
                part_coords = [(v.x, v.y) for v in p.flattening(distance=0.1)]
                break 
            except: continue

        if not part_coords:
            st.error("❌ 도면에서 다각형 폴리라인을 찾을 수 없습니다.")
        else:
            raw_part = Polygon(part_coords)
            part = raw_part.buffer(0)
            if part.geom_type == 'MultiPolygon': part = max(part.geoms, key=lambda a: a.area)
            part_area, pair_area = part.area, part.area * 2 
            
            # --- [Case 1] 단일 배열 ---
            single_results = []
            best_s_util, best_s_cost, best_s_angle, best_s_part = 0, float('inf'), 0, None
            best_s_w, best_s_p = 0, 0
            for angle in range(0, 180, 10):
                rot = rotate(part, angle, origin='center')
                p_val = get_true_pitch(rot, bridge) # 진짜 피치 계산 적용!
                minx, miny, maxx, maxy = rot.bounds
                w_val = (maxy - miny) + (margin * 2)
                util = (part_area / (p_val * w_val)) * 100
                cost = (((p_val * w_val * material_thickness) * material_density) / 1000000) * material_price
                single_results.append({'각도': f"{angle}°", '피치(mm)': round(p_val,2), '소재폭(mm)': round(w_val,2), '소재이용율(%)': round(util,2), '1개당 원가(원)': int(cost)})
                if util > best_s_util: best_s_util, best_s_cost, best_s_angle, best_s_part, best_s_w, best_s_p = util, cost, angle, rot, w_val, p_val

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
                    p_val = get_true_pitch(rot_pair, bridge) # 쌍 단위 진짜 피치 계산 적용!
                    minx, miny, maxx, maxy = rot_pair.bounds
                    w_val = (maxy - miny) + (margin * 2)
                    util = (pair_area / (p_val * w_val)) * 100
                    cost = ((((p_val * w_val * material_thickness) * material_density) / 1000000) * material_price) / 2
                    inter_results.append({'각도': f"{angle}°", '피치(mm)': round(p_val,2), '소재폭(mm)': round(w_val,2), '소재이용율(%)': round(util,2), '1개당 원가(원)': int(cost)})
                    if util > best_i_util: 
                        best_i_util, best_i_cost, best_i_angle, best_i_pair, best_i_w, best_i_p = util, cost, angle, rot_pair, w_val, p_val
                        best_i_part_a, best_i_part_b = rot_a, rot_b

            # --- [Case 3] 지그재그 배열 ---
            zigzag_results = []
            best_z_util, best_z_cost, best_z_angle = 0, float('inf'), 0
            best_z_part_a, best_z_part_b, best_z_pair = None, None, None
            best_z_w, best_z_p = 0, 0
            
            for angle in range(0, 180, 10):
                rot = rotate(part, angle, origin='center')
                p_val_single = get_true_pitch(rot, bridge) # 단일열 피치를 먼저 계산
                rot_z_a, rot_z_b, rot_pair = find_best_zigzag(rot, bridge, p_val_single)
                if rot_pair:
                    p_val = p_val_single # 지그재그 진행 피치는 단일열 피치와 동일
                    minx, miny, maxx, maxy = rot_pair.bounds
                    w_val = (maxy - miny) + (margin * 2)
                    util = (pair_area / (p_val * w_val)) * 100
                    cost = ((((p_val * w_val * material_thickness) * material_density) / 1000000) * material_price) / 2
                    zigzag_results.append({'각도': f"{angle}°", '피치(mm)': round(p_val,2), '소재폭(mm)': round(w_val,2), '소재이용율(%)': round(util,2), '1개당 원가(원)': int(cost)})
                    if util > best_z_util:
                        best_z_util, best_z_cost, best_z_angle = util, cost, angle
                        best_z_p, best_z_w = p_val, w_val
                        best_z_part_a, best_z_part_b, best_z_pair = rot_z_a, rot_z_b, rot_pair

            # --- [종합 판정] ---
            best_overall_cost = min(best_s_cost, best_i_cost, best_z_cost)
            best_method_name = "180도 교차 배열" if best_overall_cost == best_i_cost else ("지그재그 배열" if best_overall_cost == best_z_cost else "단일 배열")
            saving_cost = int(best_s_cost - best_overall_cost)

            st.success(f"🏆 분석 완료! 가장 훌륭한 배열은 **[{best_method_name}]**이며, 단일 배열 대비 1개당 :blue[**{saving_cost:,}원**]을 절감합니다.")

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
                ax1.plot(*best_s_part.exterior.xy, color='#004b87', linewidth=2); ax1.fill(*best_s_part.exterior.xy, alpha=0.5, color='#004b87')
                sx1, sx2 = best_s_part.bounds[0] - bridge/2, (best_s_part.bounds[0] - bridge/2) + best_s_p
                sy1, sy2 = best_s_part.bounds[1] - margin, best_s_part.bounds[3] + margin
                ax1.plot([sx1, sx2, sx2, sx1, sx1], [sy1, sy1, sy2, sy2, sy1], color='red', linestyle='--', linewidth=2.5)
                ax1.axis('equal'); ax1.set_xticks([]); ax1.set_yticks([])
                st.pyplot(fig1)
                df_single = pd.DataFrame(single_results)
                st.dataframe(df_single.style.apply(lambda r: highlight_best(r, df_single['소재이용율(%)'].max()), axis=1).format(format_dict), use_container_width=True)

            with col2:
                st.subheader(f"[2] 180도 교차 배열 ({best_i_angle}°)")
                st.caption(f"이용율: :blue[**{best_i_util:.2f}%**] | 단가: :blue[**{int(best_i_cost):,}원**]")
                if pair_i_geom:
                    fig2, ax2 = plt.subplots(figsize=(6, 6))
                    ax2.plot(*best_i_part_a.exterior.xy, color='#004b87', linewidth=2); ax2.fill(*best_i_part_a.exterior.xy, alpha=0.5, color='#004b87')
                    ax2.plot(*best_i_part_b.exterior.xy, color='#007934', linewidth=2); ax2.fill(*best_i_part_b.exterior.xy, alpha=0.5, color='#007934')
                    sx1, sx2 = best_i_pair.bounds[0] - bridge/2, (best_i_pair.bounds[0] - bridge/2) + best_i_p
                    sy1, sy2 = best_i_pair.bounds[1] - margin, best_i_pair.bounds[3] + margin
                    ax2.plot([sx1, sx2, sx2, sx1, sx1], [sy1, sy1, sy2, sy2, sy1], color='red', linestyle='--', linewidth=2.5)
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
                    ax3.plot(*best_z_part_a.exterior.xy, color='#004b87', linewidth=2); ax3.fill(*best_z_part_a.exterior.xy, alpha=0.5, color='#004b87')
                    ax3.plot(*best_z_part_b.exterior.xy, color='#d55e00', linewidth=2); ax3.fill(*best_z_part_b.exterior.xy, alpha=0.5, color='#d55e00')
                    sx1, sx2 = best_z_part_a.bounds[0] - bridge/2, (best_z_part_a.bounds[0] - bridge/2) + best_z_p
                    sy1, sy2 = best_z_pair.bounds[1] - margin, best_z_pair.bounds[3] + margin
                    ax3.plot([sx1, sx2, sx2, sx1, sx1], [sy1, sy1, sy2, sy2, sy1], color='red', linestyle='--', linewidth=2.5)
                    ax3.axis('equal'); ax3.set_xticks([]); ax3.set_yticks([])
                    st.pyplot(fig3)
                    df_zigzag = pd.DataFrame(zigzag_results)
                    st.dataframe(df_zigzag.style.apply(lambda r: highlight_best(r, df_zigzag['소재이용율(%)'].max()), axis=1).format(format_dict), use_container_width=True)
                else:
                    st.warning("지그재그 배열 불가 형상")

            # ==========================================
            # 화면 분리 2: Layout도 도면 및 금형 사이즈
            # ==========================================
            st.divider()
            st.header("🎞️ [2단계] 스트립 Layout도 및 금형 코어 사이즈 도출")
            st.markdown(f"좌측에서 설정한 **안전 브릿지({bridge}mm)**와 **마진({margin}mm)**이 완벽하게 적용된 1:1 비율 도면입니다.")
            
            # --- [1] 단일 배열 Layout도 ---
            st.subheader("◼️ [1] 단일 배열 Layout도")
            l_val_s = best_s_p * total_stations + margin
            st.info(f"📐 **단일 배열 금형 코어 최소 사이즈:** 가로(L) :blue[**{l_val_s:.1f} mm**] × 세로(W) :blue[**{best_s_w:.1f} mm**]")
            fig_strip1 = plot_strip_layout([(best_s_part, '#004b87')], best_s_p, best_s_w, margin, total_stations)
            st.pyplot(fig_strip1)

            # --- [2] 180도 교차 배열 Layout도 ---
            st.divider()
            st.subheader("◼️ [2] 180도 교차 배열 Layout도")
            if pair_i_geom:
                l_val_i = best_i_p * total_stations + margin
                st.info(f"📐 **180도 교차 배열 금형 코어 최소 사이즈:** 가로(L) :blue[**{l_val_i:.1f} mm**] × 세로(W) :blue[**{best_i_w:.1f} mm**]")
                fig_strip2 = plot_strip_layout([(best_i_part_a, '#004b87'), (best_i_part_b, '#007934')], best_i_p, best_i_w, margin, total_stations)
                st.pyplot(fig_strip2)
            else:
                st.warning("이 부품은 180도 교차 배열이 불가능합니다.")

            # --- [3] 지그재그 배열 Layout도 ---
            st.divider()
            st.subheader("◼️ [3] 지그재그 배열 Layout도")
            if best_z_pair is not None:
                l_val_z = best_z_p * total_stations + margin
                st.info(f"📐 **지그재그 배열 금형 코어 최소 사이즈:** 가로(L) :blue[**{l_val_z:.1f} mm**] × 세로(W) :blue[**{best_z_w:.1f} mm**]")
                fig_strip3 = plot_strip_layout([(best_z_part_a, '#004b87'), (best_z_part_b, '#d55e00')], best_z_p, best_z_w, margin, total_stations)
                st.pyplot(fig_strip3)
            else:
                st.warning("이 부품은 지그재그 배열이 불가능합니다.")
