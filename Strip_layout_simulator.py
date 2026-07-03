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

# --- [1] 핵심 알고리즘 함수 ---
def find_best_interlock(part, bridge):
    part_b_rotated = rotate(part, 180, origin='centroid')
    buffered_a = part.buffer(bridge, resolution=4)
    minx, miny, maxx, maxy = part.bounds
    w, h = maxx - minx, maxy - miny
    
    best_pair_geom, best_part_a, best_part_b = None, part, None
    min_box_area = float('inf')
    
    for dy in np.linspace(-h*0.8, h*0.8, 30): 
        dx, step = w * 1.5, w / 20 
        while dx > -w:
            test_b = translate(part_b_rotated, xoff=dx, yoff=dy)
            if buffered_a.intersects(test_b): dx += step; break
            dx -= step
        fine_step = step / 10
        while dx > -w:
            test_b = translate(part_b_rotated, xoff=dx, yoff=dy)
            if buffered_a.intersects(test_b): dx += fine_step; break
            dx -= fine_step
            
        test_b = translate(part_b_rotated, xoff=dx, yoff=dy)
        try:
            pair = unary_union([part, test_b])
            p_minx, p_miny, p_maxx, p_maxy = pair.bounds
            box_area = (p_maxx - p_minx) * (p_maxy - p_miny)
            if box_area < min_box_area:
                min_box_area, best_pair_geom, best_part_b = box_area, pair, test_b
        except: continue
    return best_part_a, best_part_b, best_pair_geom

def find_best_zigzag(part, bridge):
    part_b_same = part 
    buffered_a = part.buffer(bridge, resolution=4)
    minx, miny, maxx, maxy = part.bounds
    w, h = maxx - minx, maxy - miny
    
    best_pair_geom, best_part_a, best_part_b = None, part, None
    min_box_area = float('inf')
    
    for dy in np.linspace(-h*0.8, h*0.8, 30): 
        dx, step = w * 1.5, w / 20 
        while dx > -w:
            test_b = translate(part_b_same, xoff=dx, yoff=dy)
            if buffered_a.intersects(test_b): dx += step; break
            dx -= step
        fine_step = step / 10
        while dx > -w:
            test_b = translate(part_b_same, xoff=dx, yoff=dy)
            if buffered_a.intersects(test_b): dx += fine_step; break
            dx -= fine_step
            
        test_b = translate(part_b_same, xoff=dx, yoff=dy)
        try:
            pair = unary_union([part, test_b])
            p_minx, p_miny, p_maxx, p_maxy = pair.bounds
            box_area = (p_maxx - p_minx) * (p_maxy - p_miny)
            if box_area < min_box_area:
                min_box_area, best_pair_geom, best_part_b = box_area, pair, test_b
        except: continue
    return best_part_a, best_part_b, best_pair_geom


# --- [2] 웹사이트 화면 및 메뉴 구성 ---
st.set_page_config(page_title="프레스 레이아웃 최적화기", layout="wide")
st.title("⚙️ 프로그레시브 금형 스트립 설계 자동화기")
st.markdown("도면을 업로드하면 **최적의 다열 배열**을 찾고, **필요한 금형 길이(기장)**를 자동으로 계산합니다.")

# 사이드바 설정 영역
st.sidebar.header("📝 1. 소재 조건 입력")
mat_type = st.sidebar.radio("소재 특성 분류", ["일반 철강/연질 (SPCC, AL 등)", "고장력강/경질 (STS, SUS 등)"])
material_thickness = st.sidebar.number_input("소재 두께 (t)", value=1.2, step=0.1)
material_price = st.sidebar.number_input("단가 (원/kg)", value=1200, step=50)
material_density = st.sidebar.number_input("비중", value=7.85, step=0.01)

# 두께와 특성에 따른 안전 브릿지/마진 자동 계산 로직
if "일반" in mat_type:
    rec_bridge = max(1.2, 1.2 * material_thickness)
    rec_margin = max(1.5, 1.5 * material_thickness)
else:
    rec_bridge = max(1.5, 1.5 * material_thickness)
    rec_margin = max(2.0, 1.8 * material_thickness)

st.sidebar.header("📏 2. 배열 간격 (다이 강도 고려)")
bridge = st.sidebar.number_input("최소 브릿지 (mm)", value=float(round(rec_bridge, 1)), step=0.1)
margin = st.sidebar.number_input("가장자리 마진 (mm)", value=float(round(rec_margin, 1)), step=0.1)

st.sidebar.header("🛠️ 3. 공도도(Strip Layout) 설계")
st_notch = st.sidebar.number_input("노칭 / 파이롯트 홀", value=1, step=1, help="외곽을 자르거나 파일럿 핀용 홀을 뚫는 피치 수")
st_pierce = st.sidebar.number_input("피어싱 (내측 홀 타발)", value=1, step=1)
st_form = st.sidebar.number_input("벤딩 / 포밍", value=0, step=1)
st_blank = st.sidebar.number_input("블랭킹 (최종 낙하)", value=1, step=1)
st_idle = st.sidebar.number_input("아이들 피치 (빈 구간)", value=1, step=1, help="금형 강성을 위해 건너뛰는 빈 피치")
st_simul = st.sidebar.number_input("➖ 동시 성형 (중복 차감)", value=0, step=1, help="예: 피어싱과 벤딩을 한 피치에서 동시 수행 시 1 차감")

# 총 피치 수 계산
total_stations = (st_notch + st_pierce + st_form + st_blank + st_idle) - st_simul
st.sidebar.info(f"**총 예상 스테이션: {total_stations} 피치**")


# --- [3] 메인 화면 동작 로직 ---
uploaded_file = st.file_uploader("DXF 전개도면을 이곳에 드래그 앤 드롭 하세요.", type=['dxf'])

if uploaded_file is not None:
    with st.spinner('안전 간격 적용 및 3가지 배열 경우의 수를 분석 중입니다... (약 15초 소요)'):
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

            part_area = part.area
            pair_area = part_area * 2 
            
            # --- [Case 1] 단일 배열 ---
            single_results = []
            best_s_util, best_s_cost, best_s_angle, best_s_part = 0, float('inf'), 0, None
            best_s_w, best_s_p = 0, 0
            
            for angle in range(0, 180, 10):
                rot = rotate(part, angle, origin='center')
                minx, miny, maxx, maxy = rot.bounds
                p_val, w_val = (maxx - minx) + bridge, (maxy - miny) + (margin * 2)
                util = (part_area / (p_val * w_val)) * 100
                cost = (((p_val * w_val * material_thickness) * material_density) / 1000000) * material_price
                
                single_results.append({'각도': f"{angle}°", '피치(mm)': round(p_val,2), '소재폭(mm)': round(w_val,2), '소재이용율(%)': round(util,2), '1개당 원가(원)': int(cost)})
                if util > best_s_util: best_s_util, best_s_cost, best_s_angle, best_s_part, best_s_w, best_s_p = util, cost, angle, rot, w_val, p_val

            # --- [Case 2] 180도 교차 배열 ---
            part_i_a, part_i_b, pair_i_geom = find_best_interlock(part, bridge)
            inter_results = []
            best_i_util, best_i_cost, best_i_angle, best_i_pair = 0, float('inf'), 0, None
            best_i_w, best_i_p = 0, 0

            if pair_i_geom:
                for angle in range(0, 180, 10):
                    rot = rotate(pair_i_geom, angle, origin='center')
                    minx, miny, maxx, maxy = rot.bounds
                    p_val, w_val = (maxx - minx) + bridge, (maxy - miny) + (margin * 2)
                    util = (pair_area / (p_val * w_val)) * 100
                    cost = ((((p_val * w_val * material_thickness) * material_density) / 1000000) * material_price) / 2
                    
                    inter_results.append({'각도': f"{angle}°", '피치(mm)': round(p_val,2), '소재폭(mm)': round(w_val,2), '소재이용율(%)': round(util,2), '1개당 원가(원)': int(cost)})
                    if util > best_i_util: best_i_util, best_i_cost, best_i_angle, best_i_pair, best_i_w, best_i_p = util, cost, angle, rot, w_val, p_val

            # --- [Case 3] 지그재그 배열 ---
            part_z_a, part_z_b, pair_z_geom = find_best_zigzag(part, bridge)
            zigzag_results = []
            best_z_util, best_z_cost, best_z_angle, best_z_pair = 0, float('inf'), 0, None
            best_z_w, best_z_p = 0, 0

            if pair_z_geom:
                for angle in range(0, 180, 10):
                    rot = rotate(pair_z_geom, angle, origin='center')
                    minx, miny, maxx, maxy = rot.bounds
                    p_val, w_val = (maxx - minx) + bridge, (maxy - miny) + (margin * 2)
                    util = (pair_area / (p_val * w_val)) * 100
                    cost = ((((p_val * w_val * material_thickness) * material_density) / 1000000) * material_price) / 2
                    
                    zigzag_results.append({'각도': f"{angle}°", '피치(mm)': round(p_val,2), '소재폭(mm)': round(w_val,2), '소재이용율(%)': round(util,2), '1개당 원가(원)': int(cost)})
                    if util > best_z_util: best_z_util, best_z_cost, best_z_angle, best_z_pair, best_z_w, best_z_p = util, cost, angle, rot, w_val, p_val

            # --- [4] 종합 결과 및 금형 기장(길이) 산출 ---
            best_overall_cost = min(best_s_cost, best_i_cost, best_z_cost)
            
            # 최고 효율의 피치 값을 추출
            if best_overall_cost == best_i_cost:
                best_method_name = "180도 교차 배열"
                final_best_pitch = best_i_p
            elif best_overall_cost == best_z_cost:
                best_method_name = "지그재그 배열"
                final_best_pitch = best_z_p
            else:
                best_method_name = "단일 배열"
                final_best_pitch = best_s_p

            saving_cost = int(best_s_cost - best_overall_cost)
            est_die_length = total_stations * final_best_pitch

            st.success(f"🏆 가장 훌륭한 배열은 **[{best_method_name}]**이며, 단일 배열 대비 1개당 :blue[**{saving_cost:,}원**]을 절감합니다.")
            
            # 프로그레시브 스트립 스펙 요약 출력
            st.info(f"📏 **[프로그레시브 금형 예상 스펙]** 총 스테이션 수: **{total_stations} 피치** | 최적 이송 피치: **{final_best_pitch:.2f} mm** ➔ 예상 스트립 작업 구간(금형 기장): :blue[**{est_die_length:.2f} mm**]")
            
            format_dict = {'피치(mm)': '{:.2f}', '소재폭(mm)': '{:.2f}', '소재이용율(%)': '{:.2f}'}
            col1, col2, col3 = st.columns(3)
            
            def highlight_best(row, max_util):
                if row['소재이용율(%)'] == max_util:
                    return ['color: blue; font-weight: bold; background-color: #e6f2ff;'] * len(row)
                return [''] * len(row)

            # 1. 단일 배열
            with col1:
                st.subheader(f"[1] 단일 배열 ({best_s_angle}°)")
                st.caption(f"최고 이용율: :blue[**{best_s_util:.2f}%**] | 단가: :blue[**{int(best_s_cost):,}원**]")
                fig1, ax1 = plt.subplots(figsize=(6, 6))
                ax1.plot(*best_s_part.exterior.xy, color='#004b87', linewidth=2)
                ax1.fill(*best_s_part.exterior.xy, alpha=0.5, color='#004b87')
                minx, miny, maxx, maxy = best_s_part.bounds
                ax1.plot([minx, maxx, maxx, minx, minx], [miny, miny, maxy, maxy, miny], color='red', linestyle='--', linewidth=2.5)
                ax1.axis('equal'); ax1.set_xticks([]); ax1.set_yticks([])
                st.pyplot(fig1)
                
                df_single = pd.DataFrame(single_results)
                max_s = df_single['소재이용율(%)'].max()
                st.dataframe(df_single.style.apply(lambda r: highlight_best(r, max_s), axis=1).format(format_dict), use_container_width=True)

            # 2. 180도 교차 배열
            with col2:
                st.subheader(f"[2] 180도 교차 배열 ({best_i_angle}°)")
                st.caption(f"최고 이용율: :blue[**{best_i_util:.2f}%**] | 단가: :blue[**{int(best_i_cost):,}원**]")
                if pair_i_geom:
                    fig2, ax2 = plt.subplots(figsize=(6, 6))
                    rot_a = rotate(part_i_a, best_i_angle, origin=pair_i_geom.centroid)
                    rot_b = rotate(part_i_b, best_i_angle, origin=pair_i_geom.centroid)
                    ax2.plot(*rot_a.exterior.xy, color='#004b87', linewidth=2); ax2.fill(*rot_a.exterior.xy, alpha=0.5, color='#004b87')
                    ax2.plot(*rot_b.exterior.xy, color='#007934', linewidth=2); ax2.fill(*rot_b.exterior.xy, alpha=0.5, color='#007934')
                    minx, miny, maxx, maxy = best_i_pair.bounds
                    ax2.plot([minx, maxx, maxx, minx, minx], [miny, miny, maxy, maxy, miny], color='red', linestyle='--', linewidth=2.5)
                    ax2.axis('equal'); ax2.set_xticks([]); ax2.set_yticks([])
                    st.pyplot(fig2)
                    
                    df_inter = pd.DataFrame(inter_results)
                    max_i = df_inter['소재이용율(%)'].max()
                    st.dataframe(df_inter.style.apply(lambda r: highlight_best(r, max_i), axis=1).format(format_dict), use_container_width=True)
                else:
                    st.warning("교차 배열이 불가능한 형상입니다.")

            # 3. 지그재그 배열
            with col3:
                st.subheader(f"[3] 지그재그 배열 ({best_z_angle}°)")
                st.caption(f"최고 이용율: :blue[**{best_z_util:.2f}%**] | 단가: :blue[**{int(best_z_cost):,}원**]")
                if pair_z_geom:
                    fig3, ax3 = plt.subplots(figsize=(6, 6))
                    rot_a = rotate(part_z_a, best_z_angle, origin=pair_z_geom.centroid)
                    rot_b = rotate(part_z_b, best_z_angle, origin=pair_z_geom.centroid)
                    ax3.plot(*rot_a.exterior.xy, color='#004b87', linewidth=2); ax3.fill(*rot_a.exterior.xy, alpha=0.5, color='#004b87')
                    ax3.plot(*rot_b.exterior.xy, color='#d55e00', linewidth=2); ax3.fill(*rot_b.exterior.xy, alpha=0.5, color='#d55e00')
                    minx, miny, maxx, maxy = best_z_pair.bounds
                    ax3.plot([minx, maxx, maxx, minx, minx], [miny, miny, maxy, maxy, miny], color='red', linestyle='--', linewidth=2.5)
                    ax3.axis('equal'); ax3.set_xticks([]); ax3.set_yticks([])
                    st.pyplot(fig3)
                    
                    df_zigzag = pd.DataFrame(zigzag_results)
                    max_z = df_zigzag['소재이용율(%)'].max()
                    st.dataframe(df_zigzag.style.apply(lambda r: highlight_best(r, max_z), axis=1).format(format_dict), use_container_width=True)
                else:
                    st.warning("지그재그 배열이 불가능한 형상입니다.")